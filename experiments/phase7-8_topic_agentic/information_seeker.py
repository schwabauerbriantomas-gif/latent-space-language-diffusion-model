"""
InformationSeeker: 4th HRM head — agentic information retrieval & ingestion.

THE PROBLEM:
  The current pipeline (Generator → Reviewer → Editor) is "blind":
  - It generates from what it already knows
  - It cannot detect when it LACKS knowledge
  - It cannot QUERY SplatsDB for more information
  - It cannot TRIGGER external models to find and ingest new data

THE SOLUTION:
  InformationSeeker adds agentic capabilities to the pipeline:
  1. GAP DETECTION: Is the current knowledge sufficient for this topic?
  2. SPLATSDB QUERY: Search the latent space for relevant embeddings
  3. INGEST TRIGGER: If insufficient, formulate a retrieval query for
     an external model (LLM, web search, etc.) to find and ingest data

ARCHITECTURE:

  ┌──────────────────────────────────────────────────────────────┐
  │  InformationSeeker (4th HRM head)                           │
  │                                                              │
  │  topic_emb ──→ ConfidenceNet ──→ confidence [0,1]          │
  │                    │                                         │
  │                    │ low confidence?                         │
  │                    ▼                                         │
  │               QueryGenerator ──→ search_query (text)        │
  │                    │                                         │
  │                    ▼                                         │
  │               SplatsDB.query(query) ──→ results, density    │
  │                    │                                         │
  │                    │ density < threshold?                    │
  │                    ▼                                         │
  │               IngestTrigger ──→ {query, categories, reason} │
  │                    │                                         │
  │                    ▼                                         │
  │               [External Model: web_search, LLM, etc.]       │
  │                    │                                         │
  │                    ▼                                         │
  │               SplatsDB.ingest(new_text → bge-m3 embeddings) │
  │                    │                                         │
  └────────────────────┴─────────────────────────────────────────┘

WHY THIS MATTERS:
  - A closed system can only regurgitate what it was trained on
  - An open system can IDENTIFY what it doesn't know and SEEK it
  - This is the difference between a tool and an agent
  - SplatsDB becomes a LIVING knowledge base that grows on demand

TRAINING:
  ConfidenceNet learns to predict whether the Generator will produce
  high-quality output for a given topic. It's trained on:
  - Positive: topics where Generator + Reviewer score > threshold
  - Negative: topics where score < threshold (knowledge gap)

  QueryGenerator is a lightweight decoder that produces a natural
  language query from a topic embedding, for external retrieval.
"""
import math
import json
import sys
import time
import random
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RESULTS_DIR = REPO / "results"

from vocab_cfg import (
    build_vocab, CFGGenerator, tag_sequence, check_grammar,
    PAD, MASK, BOS, EOS, UNK, VOCAB, FUNC, ALL_NOUNS, ALL_ADJ,
)
from mdlm import forward_mask, encode_sequences, decode_tokens
from topic_mdlm import (
    TopicEncoder, TopicConditionedLayer, TopicMDLMTransformer,
    TopicReviewer, sample_topic_mdLM, build_category_topic_embeddings,
)


# ═══════════════════════════════════════════════════════════════════════════
# 1. ConfidenceNet — Detects Knowledge Gaps
# ═══════════════════════════════════════════════════════════════════════════

class ConfidenceNet(nn.Module):
    """Predicts whether the Generator can produce quality output for a topic.

    Input:  topic embedding [batch, 1024]
    Output: confidence score [0, 1] per sample

    Low confidence → knowledge gap → triggers retrieval/ingestion.

    Architecture: TopicEncoder → transformer layers → classifier
    """

    def __init__(self, topic_dim=1024, d_model=256, n_heads=4,
                 n_layers=3, dropout=0.1):
        super().__init__()
        self.topic_encoder = TopicEncoder(topic_dim, d_model, dropout)
        self.layers = nn.ModuleList([
            TopicConditionedLayer(d_model, n_heads, d_model * 4, dropout)
            for _ in range(n_layers)
        ])
        self.final_norm = nn.LayerNorm(d_model)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

    def forward(self, topic_emb):
        """topic_emb: [batch, 1024] → confidence logits [batch]"""
        topic_h = self.topic_encoder(topic_emb)  # [batch, d_model]
        x = topic_h.unsqueeze(1)  # [batch, 1, d_model]

        # Self-attention across the topic representation
        for layer in self.layers:
            x = layer(x, x)  # self-attention (topic attends to itself)

        x = self.final_norm(x[:, 0, :])  # [batch, d_model]
        return self.classifier(x).squeeze(-1)  # [batch]


# ═══════════════════════════════════════════════════════════════════════════
# 2. QueryGenerator — Produces Retrieval Queries
# ═══════════════════════════════════════════════════════════════════════════

class QueryGenerator(nn.Module):
    """Generates a natural language query from a topic embedding.

    Input:  topic embedding [batch, 1024]
    Output: token IDs for a retrieval query (e.g., "tell me about ocean animals")

    Architecture: TopicEncoder → transformer decoder → vocab projection

    This produces queries that an external system (LLM, web search)
    can use to retrieve relevant information for SplatsDB ingestion.
    """

    def __init__(self, vocab_size, topic_dim=1024, d_model=256,
                 n_heads=4, n_layers=4, max_query_len=12, dropout=0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_query_len = max_query_len
        self.d_model = d_model

        self.topic_encoder = TopicEncoder(topic_dim, d_model, dropout)
        self.token_emb = nn.Embedding(vocab_size, d_model, padding_idx=PAD)
        self.pos_emb = nn.Embedding(max_query_len, d_model)

        # Cross-attention decoder layers
        self.layers = nn.ModuleList([
            TopicConditionedLayer(d_model, n_heads, d_model * 4, dropout)
            for _ in range(n_layers)
        ])
        self.final_norm = nn.LayerNorm(d_model)
        self.output = nn.Linear(d_model, vocab_size)

    def forward(self, tokens, topic_emb):
        """Teacher-forced forward pass for training.

        Args:
            tokens: [batch, query_len] query token IDs (shifted right)
            topic_emb: [batch, 1024]
        Returns:
            logits: [batch, query_len, vocab_size]
        """
        batch, seq_len = tokens.shape

        topic_h = self.topic_encoder(topic_emb)
        topic_kv = topic_h.unsqueeze(1)  # [batch, 1, d_model]

        pos = torch.arange(seq_len, device=tokens.device).unsqueeze(0).expand(batch, -1)
        x = self.token_emb(tokens) + self.pos_emb(pos)

        pad_mask = (tokens == PAD)
        for layer in self.layers:
            x = layer(x, topic_kv, src_key_padding_mask=pad_mask)

        x = self.final_norm(x)
        return self.output(x)

    @torch.no_grad()
    def generate(self, topic_emb, tok2id, max_len=None, temperature=0.8):
        """Autoregressive query generation.

        Args:
            topic_emb: [batch, 1024] or [1024]
            max_len: max query length (default: self.max_query_len)

        Returns:
            token IDs: [batch, query_len]
        """
        self.eval()
        if topic_emb.dim() == 1:
            topic_emb = topic_emb.unsqueeze(0)
        batch = topic_emb.shape[0]
        max_len = max_len or self.max_query_len

        tokens = torch.full((batch, max_len), PAD, device=topic_emb.device)
        tokens[:, 0] = BOS

        topic_h = self.topic_encoder(topic_emb)
        topic_kv = topic_h.unsqueeze(1)

        for step in range(1, max_len):
            pos = torch.arange(max_len, device=tokens.device).unsqueeze(0).expand(batch, -1)
            x = self.token_emb(tokens) + self.pos_emb(pos)

            for layer in self.layers:
                x = layer(x, topic_kv)

            x = self.final_norm(x)
            logits = self.output(x)

            # Sample next token at current position
            next_logits = logits[:, step - 1, :] / temperature
            probs = F.softmax(next_logits, dim=-1)
            next_tok = torch.multinomial(probs, 1).squeeze(-1)
            tokens[:, step] = next_tok

            if (next_tok == EOS).all():
                break

        return tokens


# ═══════════════════════════════════════════════════════════════════════════
# 3. MockSplatsDB — Simulates SplatsDB's Vector Store
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class SplatsDBEntry:
    """A single entry in the mock SplatsDB vector store."""
    text: str
    embedding: torch.Tensor  # [1024]
    category: str
    source: str  # "initial", "ingested", "generated"


class MockSplatsDB:
    """Simulates SplatsDB's vector store for the agentic loop.

    In production, this would be the real SplatsDB via MCP/REST API.
    Here we simulate:
      - query(): find nearest entries by cosine similarity
      - ingest(): add new entries (from external retrieval)
      - density(): how many entries are near a given embedding
      - get_stats(): store statistics

    The mock starts with a partial knowledge base (some categories
    well-covered, others sparse) to simulate real-world knowledge gaps.
    """

    def __init__(self, cat_embeds, categories, device=DEVICE):
        self.cat_embeds = cat_embeds
        self.categories = categories
        self.device = device
        self.entries: List[SplatsDBEntry] = []
        self.ingest_count = 0

    def initialize(self, coverage: Optional[Dict[str, int]] = None):
        """Initialize with partial coverage (simulates knowledge gaps).

        Args:
            coverage: number of entries per category.
                      Default: some categories sparse, others rich.
        """
        if coverage is None:
            # Simulate real-world: some categories well-known, others sparse
            coverage = {}
            for i, cat in enumerate(self.categories):
                if i % 3 == 0:
                    coverage[cat] = 20  # rich
                elif i % 3 == 1:
                    coverage[cat] = 5   # sparse
                else:
                    coverage[cat] = 0   # empty (knowledge gap!)

        rng = random.Random(42)
        self.entries = []

        for cat in self.categories:
            n = coverage.get(cat, 0)
            if cat not in VOCAB:
                continue
            cat_words = VOCAB[cat]
            for i in range(n):
                word = cat_words[i % len(cat_words)]
                # Simulate embedding with noise around category centroid
                emb = self.cat_embeds[cat] + 0.3 * torch.randn(1024, device=self.device)
                emb = F.normalize(emb, dim=0)
                self.entries.append(SplatsDBEntry(
                    text=f"{word} ({cat})",
                    embedding=emb,
                    category=cat,
                    source="initial",
                ))

        print(f"SplatsDB initialized: {len(self.entries)} entries")
        for cat in self.categories:
            n = sum(1 for e in self.entries if e.category == cat)
            print(f"  {cat:15s}: {n:3d} entries")

    def query(self, topic_emb, k=5):
        """Find k nearest entries to topic embedding.

        Returns:
            (entries, similarities, density_score)
        """
        if not self.entries:
            return [], [], 0.0

        all_embs = torch.stack([e.embedding for e in self.entries])  # [N, 1024]
        topic_norm = F.normalize(topic_emb.unsqueeze(0), dim=-1)
        all_norm = F.normalize(all_embs, dim=-1)
        sims = (topic_norm @ all_norm.T).squeeze(0)  # [N]

        topk_sims, topk_idx = sims.topk(min(k, len(self.entries)))
        results = [self.entries[i] for i in topk_idx.tolist()]

        # Density: how concentrated the top-k similarities are
        density = topk_sims.mean().item()

        return results, topk_sims.tolist(), density

    def density(self, topic_emb, threshold=0.15):
        """How many entries are within similarity threshold of topic.

        Low density → knowledge gap → triggers retrieval.

        Note: threshold=0.15 is calibrated for 1024D block-separated
        category embeddings where intra-category cosine sim ≈ 0.08-0.20.
        """
        if not self.entries:
            return 0.0

        all_embs = torch.stack([e.embedding for e in self.entries])
        topic_norm = F.normalize(topic_emb.unsqueeze(0), dim=-1)
        all_norm = F.normalize(all_embs, dim=-1)
        sims = (topic_norm @ all_norm.T).squeeze(0)

        return (sims > threshold).float().mean().item()

    def ingest(self, text, embedding, category, source="ingested"):
        """Add a new entry to the vector store."""
        self.entries.append(SplatsDBEntry(
            text=text,
            embedding=F.normalize(embedding, dim=0),
            category=category,
            source=source,
        ))
        self.ingest_count += 1

    def ingest_batch(self, items: List[Tuple[str, torch.Tensor, str]]):
        """Ingest multiple entries at once."""
        for text, emb, cat in items:
            self.ingest(text, emb, cat)

    def get_category_density(self, category):
        """Count entries in a specific category."""
        return sum(1 for e in self.entries if e.category == category)

    def get_stats(self):
        """Return store statistics."""
        by_cat = {}
        by_source = {}
        for e in self.entries:
            by_cat[e.category] = by_cat.get(e.category, 0) + 1
            by_source[e.source] = by_source.get(e.source, 0) + 1
        return {
            "total_entries": len(self.entries),
            "by_category": by_cat,
            "by_source": by_source,
            "ingest_count": self.ingest_count,
        }


# ═══════════════════════════════════════════════════════════════════════════
# 4. IngestTrigger — Formulates Retrieval Requests
# ═══════════════════════════════════════════════════════════════/s═════════

@dataclass
class IngestRequest:
    """A request to ingest new data into SplatsDB."""
    query: str                    # Natural language query for external retrieval
    target_category: str          # Which category needs more data
    reason: str                   # Why ingestion is needed
    current_density: float        # Current density score
    confidence: float             # ConfidenceNet score
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()


class IngestTrigger:
    """Decides WHEN to trigger ingestion and FORMULATES the request.

    Logic:
      1. If ConfidenceNet score < threshold → knowledge gap detected
      2. Query SplatsDB for nearby entries
      3. If density < threshold → not enough data → trigger ingestion
      4. Generate a query for external retrieval
    """

    def __init__(self, confidence_threshold=0.5, density_threshold=0.1):
        self.confidence_threshold = confidence_threshold
        self.density_threshold = density_threshold

    def evaluate(self, topic_emb, category, confidence_score,
                 splatsdb: MockSplatsDB, query_text: str = "") -> Optional[IngestRequest]:
        """Decide whether to trigger ingestion.

        Returns:
            IngestRequest if ingestion needed, None otherwise.
        """
        # Use category count as primary density signal (more reliable than
        # cosine density for block-separated embeddings)
        cat_count = splatsdb.get_category_density(category)
        # Normalize: density_threshold is interpreted as "minimum entries"
        min_entries = max(1, int(self.density_threshold * 100))

        density = splatsdb.density(topic_emb)  # cosine density (for reporting)

        needs_ingestion = (
            confidence_score < self.confidence_threshold or
            cat_count < min_entries
        )

        if not needs_ingestion:
            return None

        reason_parts = []
        if confidence_score < self.confidence_threshold:
            reason_parts.append(f"low confidence ({confidence_score:.2f} < {self.confidence_threshold})")
        if cat_count < min_entries:
            reason_parts.append(f"low coverage ({cat_count} entries < {min_entries} minimum)")

        return IngestRequest(
            query=query_text or f"retrieve information about {category}",
            target_category=category,
            reason="; ".join(reason_parts),
            current_density=density,
            confidence=confidence_score,
        )


# ═════════════════════════════════════════════════════════ SplatsDB simulation

class MockExternalRetriever:
    """Simulates an external model (LLM, web search) that retrieves data.

    In production, this would be:
      - A web search API (Google, Bing, Brave)
      - An LLM with internet access (Perplexity, GPT-4, Claude)
      - A specialized database (Wikipedia API, arxiv, etc.)

    Here we simulate it by generating relevant text from the vocabulary.
    """

    def __init__(self, cat_embeds, categories, device=DEVICE):
        self.cat_embeds = cat_embeds
        self.categories = categories
        self.device = device

    def retrieve(self, query: str, target_category: str,
                 n_items: int = 10) -> List[Tuple[str, torch.Tensor, str]]:
        """Simulate external retrieval.

        Returns list of (text, embedding, category) tuples for ingestion.
        """
        if target_category not in VOCAB or target_category not in self.cat_embeds:
            return []

        cat_words = VOCAB[target_category]
        results = []

        for i in range(min(n_items, len(cat_words))):
            word = cat_words[(i + 5) % len(cat_words)]  # offset to avoid duplicates
            # Generate embedding near category centroid
            emb = self.cat_embeds[target_category] + 0.2 * torch.randn(1024, device=self.device)
            emb = F.normalize(emb, dim=0)
            text = f"{word} ({target_category}) [retrieved]"
            results.append((text, emb, target_category))

        return results


# ═══════════════.load_state_dict.load_state_dict══════════════════════════════════
# 5. AgenticPipeline — Full Agentic HRM Loop
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class AgenticConfig:
    """Configuration for the agentic pipeline."""
    # ConfidenceNet
    conf_d_model: int = 256
    conf_n_layers: int = 3
    conf_epochs: int = 1500

    # QueryGenerator
    qg_d_model: int = 256
    qg_n_layers: int = 4
    qg_max_query_len: int = 8
    qg_epochs: int = 2000

    # SplatsDB simulation
    initial_coverage: Optional[Dict[str, int]] = None

    # Agentic loop
    confidence_threshold: float = 0.5
    density_threshold: float = 0.1
    n_retrieval_items: int = 10


class AgenticPipeline:
    """Full agentic HRM pipeline with information seeking.

    Pipeline flow:
      1. Receive topic request
      2. ConfidenceNet evaluates: can we handle this?
      3. If low confidence → query SplatsDB
      4. If SplatsDB sparse → IngestTrigger → external retrieval → ingest
      5. Generator produces text (topic-conditioned)
      6. Reviewer scores → Editor refines → output

    Usage:
        pipeline = AgenticPipeline()
        pipeline.train_information_seeker()
        result = pipeline.process_topic("animals", generate=True)
    """

    def __init__(self, config: Optional[AgenticConfig] = None,
                 generator=None, reviewer=None, editor=None):
        self.config = config or AgenticConfig()
        self.tok2id, self.id2tok, self.all_words = build_vocab()
        self.vocab_size = len(self.all_words)
        self.cfg_gen = CFGGenerator(seed=42)

        # Reuse existing topic-conditioned models if provided
        self.generator = generator
        self.reviewer = reviewer
        self.editor = editor

        # New components
        self.confidence_net: Optional[ConfidenceNet] = None
        self.query_generator: Optional[QueryGenerator] = None
        self.splatsdb: Optional[MockSplatsDB] = None
        self.ingest_trigger = IngestTrigger(
            self.config.confidence_threshold,
            self.config.density_threshold,
        )
        self.external_retriever: Optional[MockExternalRetriever] = None

        # Build topic embeddings
        self.cat_embeds, self.categories, self.mixed = \
            build_category_topic_embeddings()

    # ── Training ──────────────────────────────────────────────────────

    def train_confidence_net(self, generator, reviewer, verbose=True):
        """Train ConfidenceNet to predict generation quality.

        Generates sequences for each category, scores them with the
        reviewer, and trains ConfidenceNet to predict the score
        from the topic embedding alone.
        """
        cfg = self.config

        # Generate training data: (topic_emb, quality_score) pairs
        if verbose:
            print(f"\n{'='*60}")
            print("TRAINING CONFIDENCE NET (generating quality labels)")
            print(f"{'='*60}")

        # For each category, generate samples and get reviewer scores
        training_topics = []
        training_scores = []

        for round_i in range(20):
            for cat in self.categories:
                topic = self.cat_embeds[cat]
                # Generate with noise (simulate variety)
                noisy_topic = topic + 0.1 * torch.randn(1024, device=DEVICE)
                noisy_topic = F.normalize(noisy_topic, dim=0)

                # Generate samples
                samples = sample_topic_mdLM(
                    generator, seq_len=20, topic_emb=noisy_topic,
                    tok2id=self.tok2id, n_samples=5, n_steps=30,
                    temperature=0.7,
                )
                topic_batch = noisy_topic.unsqueeze(0).expand(5, -1).to(DEVICE)
                scores = torch.sigmoid(reviewer(samples, topic_batch))
                mean_score = scores.mean().item()

                training_topics.append(noisy_topic)
                training_scores.append(mean_score)

        # Also add mixed/low-confidence examples
        for _ in range(20):
            # Random topic (should have low confidence)
            rand_topic = F.normalize(torch.randn(1024, device=DEVICE), dim=0)
            noisy_mixed = self.mixed + 0.3 * torch.randn(1024, device=DEVICE)
            noisy_mixed = F.normalize(noisy_mixed, dim=0)
            training_topics.append(noisy_mixed)
            training_scores.append(0.2 + 0.1 * random.random())

        training_topics = torch.stack(training_topics)  # [N, 1024]
        training_scores = torch.tensor(training_scores, device=DEVICE)  # [N]

        if verbose:
            print(f"  Training data: {len(training_topics)} (topic, score) pairs")
            print(f"  Score distribution: min={training_scores.min():.2f} "
                  f"max={training_scores.max():.2f} mean={training_scores.mean():.2f}")

        # Train ConfidenceNet
        model = ConfidenceNet(
            topic_dim=1024, d_model=cfg.conf_d_model,
            n_heads=4, n_layers=cfg.conf_n_layers, dropout=0.1,
        ).to(DEVICE)

        n_params = sum(p.numel() for p in model.parameters())
        if verbose:
            print(f"  ConfidenceNet: {n_params:,} params ({n_params/1e6:.1f}M)")

        opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
        n_train = len(training_topics)
        t0 = time.time()

        for epoch in range(cfg.conf_epochs):
            idx = torch.randint(0, n_train, (min(64, n_train),))
            batch_topics = training_topics[idx]
            batch_scores = training_scores[idx]

            opt.zero_grad()
            pred = model(batch_topics)
            loss = F.mse_loss(pred, batch_scores)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            if epoch % 300 == 0 or epoch == cfg.conf_epochs - 1:
                with torch.no_grad():
                    all_pred = torch.sigmoid(model(training_topics))
                    # Bin into confident vs not
                    high_conf = (training_scores > 0.5).float()
                    pred_high = (all_pred > 0).float()
                    acc = (pred_high == high_conf).float().mean().item()
                if verbose:
                    print(f"  ep {epoch:4d}: loss={loss.item():.4f} acc={acc:.1%} "
                          f"({time.time()-t0:.0f}s)")

        self.confidence_net = model
        return model

    def train_query_generator(self, verbose=True):
        """Train QueryGenerator to produce retrieval queries from topics.

        Training data: (topic_emb, query_text) pairs.
        Queries are templated: "tell me about {category}"
        "what are {category}"
        "find information on {category}"
        """
        cfg = self.config

        # Generate training queries
        training_topics = []
        training_queries = []

        query_templates = [
            lambda cat: f"tell me about {cat}",
            lambda cat: f"what are {cat}",
            lambda cat: f"find {cat} information",
            lambda cat: f"search for {cat}",
            lambda cat: f"learn about {cat}",
        ]

        for cat in self.categories:
            topic = self.cat_embeds[cat]
            for tmpl in query_templates:
                query = tmpl(cat)
                words = query.split()
                ids = [BOS] + [self.tok2id.get(w, UNK) for w in words] + [EOS]
                while len(ids) < cfg.qg_max_query_len:
                    ids.append(PAD)
                training_topics.append(topic)
                training_queries.append(ids)

        training_topics = torch.stack(training_topics).to(DEVICE)
        training_queries = torch.tensor(training_queries, device=DEVICE)

        if verbose:
            print(f"\n{'='*60}")
            print("TRAINING QUERY GENERATOR")
            print(f"{'='*60}")
            print(f"  Training data: {len(training_topics)} (topic, query) pairs")

        model = QueryGenerator(
            vocab_size=self.vocab_size, topic_dim=1024,
            d_model=cfg.qg_d_model, n_heads=4, n_layers=cfg.qg_n_layers,
            max_query_len=cfg.qg_max_query_len, dropout=0.1,
        ).to(DEVICE)

        n_params = sum(p.numel() for p in model.parameters())
        if verbose:
            print(f"  QueryGenerator: {n_params:,} params ({n_params/1e6:.1f}M)")

        opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
        n_train = len(training_topics)
        t0 = time.time()

        for epoch in range(cfg.qg_epochs):
            idx = torch.randint(0, n_train, (min(64, n_train),))
            batch_topics = training_topics[idx]
            batch_queries = training_queries[idx]

            opt.zero_grad()
            logits = model(batch_queries, batch_topics)  # [batch, seq, vocab]

            # Cross-entropy over non-pad positions
            # Shift: predict token[i] from token[i-1]
            shift_logits = logits[:, :-1, :].contiguous()
            shift_targets = batch_queries[:, 1:].contiguous()

            loss = F.cross_entropy(
                shift_logits.reshape(-1, self.vocab_size),
                shift_targets.reshape(-1),
                ignore_index=PAD,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            if epoch % 300 == 0 or epoch == cfg.qg_epochs - 1:
                if verbose:
                    print(f"  ep {epoch:4d}: loss={loss.item():.4f} ({time.time()-t0:.0f}s)")

        self.query_generator = model

        # Show sample queries
        if verbose:
            print(f"\n  Sample generated queries:")
            for cat in self.categories[:5]:
                topic = self.cat_embeds[cat]
                tokens = model.generate(topic, self.tok2id, temperature=0.5)
                decoded = decode_tokens(tokens, self.id2tok)
                print(f"    {cat:15s} → {decoded[0]}")

        return model

    # ── SplatsDB Setup ────────────────────────────────────────────────

    def setup_splatsdb(self, coverage: Optional[Dict[str, int]] = None):
        """Initialize the mock SplatsDB with partial coverage."""
        self.splatsdb = MockSplatsDB(self.cat_embeds, self.categories)
        self.splatsdb.initialize(coverage or self.config.initial_coverage)
        self.external_retriever = MockExternalRetriever(self.cat_embeds, self.categories)

    # ── Agentic Loop ──────────────────────────────────────────────────

    def process_topic(self, category: str, verbose=True) -> Dict:
        """Full agentic processing of a topic request.

        Steps:
          1. ConfidenceNet evaluates topic
          2. Query SplatsDB
          3. If needed: IngestTrigger → retrieve → ingest
          4. Generate text
          5. Reviewer scores
          6. Editor refines

        Returns dict with all steps' results.
        """
        topic = self.cat_embeds[category]

        result = {
            "category": category,
            "steps": [],
        }

        # ── Step 1: Confidence Assessment ─────────────────────────
        with torch.no_grad():
            confidence = torch.sigmoid(self.confidence_net(topic.unsqueeze(0))).item()

        result["steps"].append({
            "step": "confidence_assessment",
            "confidence": confidence,
            "verdict": "high" if confidence > self.config.confidence_threshold else "low",
        })

        if verbose:
            print(f"\n  [{category}] Step 1: Confidence = {confidence:.3f} "
                  f"({'HIGH' if confidence > self.config.confidence_threshold else 'LOW — gap detected'})")

        # ── Step 2: SplatsDB Query ────────────────────────────────
        entries, sims, density = self.splatsdb.query(topic, k=5)
        cat_density = self.splatsdb.get_category_density(category)

        result["steps"].append({
            "step": "splatsdb_query",
            "n_nearby": len(entries),
            "density": density,
            "category_count": cat_density,
            "top_results": [e.text for e in entries[:3]],
        })

        if verbose:
            print(f"  [{category}] Step 2: SplatsDB has {cat_density} entries, "
                  f"density={density:.3f}")
            if entries:
                print(f"    Top: {entries[0].text} (sim={sims[0]:.3f})")

        # ── Step 3: IngestTrigger (if needed) ─────────────────────
        # Generate query text
        query_text = f"retrieve information about {category}"
        if self.query_generator is not None:
            with torch.no_grad():
                q_tokens = self.query_generator.generate(
                    topic, self.tok2id, temperature=0.5,
                )
                query_decoded = decode_tokens(q_tokens, self.id2tok)
                query_text = query_decoded[0] if query_decoded else query_text

        ingest_req = self.ingest_trigger.evaluate(
            topic_emb=topic,
            category=category,
            confidence_score=confidence,
            splatsdb=self.splatsdb,
            query_text=query_text,
        )

        if ingest_req is not None:
            result["steps"].append({
                "step": "ingest_triggered",
                "query": ingest_req.query,
                "reason": ingest_req.reason,
                "target_category": ingest_req.target_category,
            })

            if verbose:
                print(f"  [{category}] Step 3: ⚡ INGEST TRIGGERED")
                print(f"    Query: \"{ingest_req.query}\"")
                print(f"    Reason: {ingest_req.reason}")

            # Simulate external retrieval
            retrieved = self.external_retriever.retrieve(
                ingest_req.query, ingest_req.target_category,
                n_items=self.config.n_retrieval_items,
            )

            # Ingest into SplatsDB
            self.splatsdb.ingest_batch(retrieved)

            # Re-query after ingestion
            entries_after, sims_after, density_after = self.splatsdb.query(topic, k=5)
            cat_density_after = self.splatsdb.get_category_density(category)

            result["steps"].append({
                "step": "data_ingested",
                "n_ingested": len(retrieved),
                "density_before": density,
                "density_after": density_after,
                "category_count_before": cat_density,
                "category_count_after": cat_density_after,
            })

            if verbose:
                print(f"  [{category}] Step 3b: ✅ Ingested {len(retrieved)} items")
                print(f"    Density: {density:.3f} → {density_after:.3f}")
                print(f"    Category entries: {cat_density} → {cat_density_after}")
        else:
            result["steps"].append({
                "step": "no_ingestion_needed",
                "confidence": confidence,
                "density": density,
            })
            if verbose:
                print(f"  [{category}] Step 3: ✓ No ingestion needed "
                      f"(confidence={confidence:.2f}, density={density:.3f})")

        # ── Step 4: Generate (if models available) ────────────────
        if self.generator is not None and self.reviewer is not None:
            samples = sample_topic_mdLM(
                self.generator, seq_len=20, topic_emb=topic,
                tok2id=self.tok2id, n_samples=5, n_steps=30,
                temperature=0.7,
            )
            topic_batch = topic.unsqueeze(0).expand(5, -1).to(DEVICE)
            scores = torch.sigmoid(self.reviewer(samples, topic_batch))
            decoded = decode_tokens(samples, self.id2tok)

            result["generation"] = {
                "samples": decoded,
                "scores": scores.tolist(),
                "mean_score": scores.mean().item(),
            }

            if verbose:
                print(f"  [{category}] Step 4: Generated text (score={scores.mean().item():.3f})")
                for i in range(min(3, len(decoded))):
                    print(f"    [{scores[i].item():.2f}] {decoded[i]}")

        return result


# ═══════════════════════════════════════════════════════════════════════════
# 6. Experiment Runner
# ═══════════════════════════════════════════════════════════════════════════

def run_agentic_experiment():
    """Run the full agentic pipeline experiment.

    Tests the InformationSeeker's ability to:
      1. Detect knowledge gaps (sparse categories)
      2. Trigger ingestion
      3. Fill gaps via external retrieval
      4. Verify knowledge base grew correctly
    """
    print("=" * 70)
    print("AGENTIC PIPELINE: InformationSeeker (4th HRM head)")
    print("=" * 70)

    # ── Load Phase 7 models ───────────────────────────────────────────
    print("\nLoading Phase 7 topic-conditioned models...")
    ckpt = torch.load(
        RESULTS_DIR / "topic_conditioned_models.pt",
        map_location=DEVICE, weights_only=False,
    )

    tok2id = ckpt["tok2id"]
    id2tok = ckpt["id2tok"]
    vocab_size = len(tok2id)
    cat_embeds = ckpt["cat_embeds"]
    categories = ckpt["categories"]

    # Rebuild models from checkpoint
    generator = TopicMDLMTransformer(
        vocab_size, topic_dim=1024, d_model=384,
        n_heads=6, n_layers=6, max_seq_len=20, dropout=0.1,
    ).to(DEVICE)
    generator.load_state_dict(ckpt["generator"])
    generator.eval()

    reviewer = TopicReviewer(
        vocab_size, topic_dim=1024, d_model=256,
        n_heads=4, n_layers=4, max_seq_len=20, dropout=0.1,
    ).to(DEVICE)
    reviewer.load_state_dict(ckpt["reviewer"])
    reviewer.eval()

    print(f"  Generator + Reviewer loaded ({vocab_size} vocab, {len(categories)} categories)")

    # ── Build Agentic Pipeline ────────────────────────────────────────
    config = AgenticConfig(
        conf_d_model=256, conf_n_layers=3, conf_epochs=1500,
        qg_d_model=256, qg_n_layers=4, qg_max_query_len=8, qg_epochs=2000,
        confidence_threshold=0.5,
        density_threshold=0.05,  # 5 minimum entries per category for sufficient coverage
        n_retrieval_items=10,
    )

    pipeline = AgenticPipeline(
        config=config,
        generator=generator,
        reviewer=reviewer,
    )

    # ── Train InformationSeeker components ────────────────────────────
    pipeline.train_confidence_net(generator, reviewer, verbose=True)
    pipeline.train_query_generator(verbose=True)

    # ── Setup SplatsDB with DELIBERATE KNOWLEDGE GAPS ─────────────────
    print(f"\n{'='*60}")
    print("SETTING UP SPLATSDB WITH KNOWLEDGE GAPS")
    print(f"{'='*60}")

    # Deliberately make some categories empty/sparse to test gap detection
    coverage = {}
    for i, cat in enumerate(categories):
        if i % 4 == 0:
            coverage[cat] = 15  # rich
        elif i % 4 == 1:
            coverage[cat] = 3   # sparse
        else:
            coverage[cat] = 0   # EMPTY (knowledge gap!)

    pipeline.setup_splatsdb(coverage=coverage)

    # ── Run Agentic Loop for Each Category ────────────────────────────
    print(f"\n{'='*60}")
    print("AGENTIC LOOP: Processing each category")
    print(f"{'='*60}")

    all_results = {}
    for cat in categories:
        result = pipeline.process_topic(cat, verbose=True)
        all_results[cat] = result

    # ── Summary ───────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("AGENTIC PIPELINE SUMMARY")
    print(f"{'='*70}")

    stats = pipeline.splatsdb.get_stats()

    print(f"\n  SplatsDB Statistics:")
    print(f"    Total entries: {stats['total_entries']}")
    print(f"    Total ingested: {stats['ingest_count']}")
    print(f"    By source: {stats['by_source']}")

    print(f"\n  Category Coverage (before → after):")
    initial_coverage = coverage
    for cat in categories:
        before = initial_coverage.get(cat, 0)
        after = stats["by_category"].get(cat, 0)
        gap_filled = after - before
        marker = "⚡" if gap_filled > 0 else " "
        print(f"    {marker} {cat:15s}: {before:3d} → {after:3d} (+{gap_filled})")

    # Ingestion decisions
    n_triggered = sum(1 for r in all_results.values()
                      if any(s["step"] == "ingest_triggered" for s in r["steps"]))
    n_skipped = sum(1 for r in all_results.values()
                    if any(s["step"] == "no_ingestion_needed" for s in r["steps"]))

    print(f"\n  Ingestion Decisions:")
    print(f"    Triggered: {n_triggered}/{len(categories)} categories")
    print(f"    Skipped (sufficient): {n_skipped}/{len(categories)} categories")
    print(f"    Total items ingested: {stats['ingest_count']}")

    # Save results
    result = {
        "experiment": "agentic_information_seeker",
        "timestamp": datetime.now().isoformat(),
        "splatsdb_stats": stats,
        "initial_coverage": initial_coverage,
        "n_categories_triggered": n_triggered,
        "n_categories_skipped": n_skipped,
        "category_results": {
            cat: {
                "confidence": r["steps"][0]["confidence"] if r["steps"] else None,
                "steps": r["steps"],
                "generation": r.get("generation"),
            }
            for cat, r in all_results.items()
        },
    }

    out = RESULTS_DIR / "agentic_results.json"
    with open(out, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"\nResults saved to {out}")

    # Save InformationSeeker models
    torch.save({
        "confidence_net": pipeline.confidence_net.state_dict(),
        "query_generator": pipeline.query_generator.state_dict(),
        "tok2id": tok2id,
        "id2tok": id2tok,
    }, RESULTS_DIR / "information_seeker_models.pt")
    print(f"Models saved to {RESULTS_DIR / 'information_seeker_models.pt'}")


if __name__ == "__main__":
    run_agentic_experiment()
