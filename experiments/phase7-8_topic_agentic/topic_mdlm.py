"""
Latent-Conditioned MDLM: integrates SplatsDB's latent diffusion with MDLM.

This module closes the loop:
  1. SplatsDB stores text as bge-m3 embeddings (1024D)
  2. Latent diffusion (PCA+SVGD) samples new topic embeddings from that space
  3. TopicEncoder projects 1024D → d_model
  4. Topic cross-attention conditions every MDLM transformer layer
  5. MDLM generates tokens semantically consistent with the sampled topic

Architecture:
                         SplatsDB (bge-m3, 1024D)
                              │
                    ┌─────────▼──────────┐
                    │  Latent Diffusion   │
                    │  PCA(k) + SVGD      │
                    │  → topic_e (1024D)  │
                    └─────────┬──────────┘
                              │
                    ┌─────────▼──────────┐
                    │  TopicEncoder       │
                    │  1024 → d_model     │
                    │  → topic_h (d_model)│
                    └─────────┬──────────┘
                              │
    ┌─────────────────────────▼──────────────────────────┐
    │  Topic-Conditioned MDLM Transformer                │
    │                                                    │
    │  [M] [M] [M] [M]  +  topic_h                       │
    │       │                    │                       │
    │  Token + Pos Emb      Cross-Attention              │
    │       │              (K,V from topic)              │
    │       ▼                    ▼                       │
    │  ┌─────────────────────────────┐                   │
    │  │ Self-Attention (tokens)     │                   │
    │  │ Cross-Attention (← topic)   │  × N layers       │
    │  │ FFN                         │                   │
    │  └─────────────────────────────┘                   │
    │       │                                            │
    │       ▼                                            │
    │  Logits over vocabulary                            │
    └────────────────────────────────────────────────────┘

WHY THIS MATTERS:
  - Phase 5-6 MDLM generates grammatical text but SEMANTICALLY RANDOM
    ("the road swims at an gray hand")
  - With topic conditioning, the model should generate text about
    the topic: if topic ≈ "ocean", text should mention water, fish, waves
  - This is the missing link between SplatsDB's semantic memory
    and the text generation pipeline
"""
import math
import json
import sys
import time
import random
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple, Dict

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
    grammar_score, PAD, MASK, BOS, EOS, UNK,
    ALL_NOUNS, ALL_ADJ, FUNC, VOCAB,
)
from mdlm import forward_mask, encode_sequences, decode_tokens


# ═══════════════════════════════════════════════════════════════════════════
# 1. TopicEncoder: 1024D → d_model
# ═══════════════════════════════════════════════════════════════════════════

class TopicEncoder(nn.Module):
    """Projects a bge-m3 topic embedding (1024D) to the transformer's d_model.

    Architecture: Linear → LayerNorm → GELU → Linear → LayerNorm
    """

    def __init__(self, input_dim=1024, d_model=384, dropout=0.1):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(input_dim, d_model * 2),
            nn.LayerNorm(d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
        )

    def forward(self, topic_emb):
        """topic_emb: [batch, 1024] → [batch, d_model]"""
        return self.proj(topic_emb)


# ═══════════════════════════════════════════════════════════════════════════
# 2. Topic-Conditioned Transformer Layer
# ═══════════════════════════════════════════════════════════════════════════

class TopicConditionedLayer(nn.Module):
    """Transformer encoder layer with cross-attention to topic embedding.

    Flow:
      x → Self-Attention → Add+Norm → Cross-Attention(topic) → Add+Norm → FFN → Add+Norm

    The cross-attention lets each token position attend to the topic,
    enabling semantically-conditioned token prediction.
    """

    def __init__(self, d_model, n_heads, dim_feedforward, dropout=0.1):
        super().__init__()
        # Self-attention (token ↔ token)
        self.self_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True,
        )
        self.norm1 = nn.LayerNorm(d_model)

        # Cross-attention (token ← topic)
        self.cross_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True,
        )
        self.norm2 = nn.LayerNorm(d_model)

        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
        )
        self.norm3 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, topic_kv, src_key_padding_mask=None):
        """
        Args:
            x: [batch, seq_len, d_model]
            topic_kv: [batch, 1, d_model] (topic representation)
            src_key_padding_mask: [batch, seq_len] True=pad
        """
        # Self-attention
        residual = x
        x_norm = self.norm1(x)
        x2, _ = self.self_attn(x_norm, x_norm, x_norm,
                                key_padding_mask=src_key_padding_mask)
        x = residual + self.dropout(x2)

        # Cross-attention to topic (query=token, key/value=topic)
        residual = x
        x_norm = self.norm2(x)
        x2, _ = self.cross_attn(x_norm, topic_kv, topic_kv)  # no padding mask for topic
        x = residual + self.dropout(x2)

        # FFN
        residual = x
        x_norm = self.norm3(x)
        x2 = self.ffn(x_norm)
        x = residual + self.dropout(x2)

        return x


# ═══════════════════════════════════════════════════════════════════════════
# 3. Topic-Conditioned MDLM Transformer
# ═══════════════════════════════════════════════════════════════════════════

class TopicMDLMTransformer(nn.Module):
    """MDLM transformer with topic conditioning via cross-attention.

    Compared to MDLMTransformer (from mdlm.py):
      - Adds TopicEncoder (1024D → d_model)
      - Each layer has cross-attention to the topic
      - Sampling is conditioned on a topic embedding
    """

    def __init__(self, vocab_size, topic_dim=1024, d_model=384,
                 n_heads=6, n_layers=6, max_seq_len=20, dropout=0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model

        # Embeddings
        self.token_emb = nn.Embedding(vocab_size, d_model, padding_idx=PAD)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.time_mlp = nn.Sequential(
            nn.Linear(d_model, d_model), nn.SiLU(), nn.Linear(d_model, d_model),
        )

        # Topic encoder
        self.topic_encoder = TopicEncoder(topic_dim, d_model, dropout)

        # Transformer layers with cross-attention
        self.layers = nn.ModuleList([
            TopicConditionedLayer(d_model, n_heads, d_model * 4, dropout)
            for _ in range(n_layers)
        ])
        self.final_norm = nn.LayerNorm(d_model)

        # Output head
        self.output = nn.Linear(d_model, vocab_size)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def _timestep_embedding(self, t, max_period=10000):
        half = self.d_model // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(half, device=t.device) / half
        )
        args = t[:, None] * freqs[None, :]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if emb.shape[-1] < self.d_model:
            emb = F.pad(emb, (0, self.d_model - emb.shape[-1]))
        return self.time_mlp(emb)

    def forward(self, tokens, t, topic_emb):
        """
        Args:
            tokens: [batch, seq_len]
            t: [batch] diffusion timestep
            topic_emb: [batch, 1024] bge-m3 topic embedding

        Returns:
            logits: [batch, seq_len, vocab_size]
        """
        batch, seq_len = tokens.shape

        # Embed tokens + position + time
        pos = torch.arange(seq_len, device=tokens.device).unsqueeze(0).expand(batch, -1)
        x = self.token_emb(tokens) + self.pos_emb(pos)

        t_emb = self._timestep_embedding(t.float())
        x = x + t_emb.unsqueeze(1)

        # Encode topic
        topic_h = self.topic_encoder(topic_emb)  # [batch, d_model]
        topic_kv = topic_h.unsqueeze(1)  # [batch, 1, d_model]

        # Padding mask
        pad_mask = (tokens == PAD)

        # Pass through layers
        for layer in self.layers:
            x = layer(x, topic_kv, src_key_padding_mask=pad_mask)

        x = self.final_norm(x)
        logits = self.output(x)
        return logits


# ═══════════════════════════════════════════════════════════════════════════
# 4. Topic-Conditioned Reviewer
# ═══════════════════════════════════════════════════════════════════════════

class TopicReviewer(nn.Module):
    """Reviewer that checks both grammaticality AND topic consistency.

    Scores a sequence given a topic:
      - High score = grammatical AND on-topic
      - Low score = ungrammatical OR off-topic

    This is more selective than the Phase 6 reviewer (grammar only).
    """

    def __init__(self, vocab_size, topic_dim=1024, d_model=256,
                 n_heads=4, n_layers=4, max_seq_len=20, dropout=0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model

        self.token_emb = nn.Embedding(vocab_size, d_model, padding_idx=PAD)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)

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

    def forward(self, tokens, topic_emb):
        """Returns logits [batch] (positive = good+on-topic)."""
        batch, seq_len = tokens.shape
        pos = torch.arange(seq_len, device=tokens.device).unsqueeze(0).expand(batch, -1)
        x = self.token_emb(tokens) + self.pos_emb(pos)

        topic_h = self.topic_encoder(topic_emb)
        topic_kv = topic_h.unsqueeze(1)

        pad_mask = (tokens == PAD)
        for layer in self.layers:
            x = layer(x, topic_kv, src_key_padding_mask=pad_mask)

        cls_repr = self.final_norm(x[:, 0, :])
        return self.classifier(cls_repr).squeeze(-1)


# ═══════════════════════════════════════════════════════════════════════════
# 5. Loss Functions
# ═══════════════════════════════════════════════════════════════════════════

def topic_mdlm_loss(model, tokens, topic_emb):
    """MDLM loss conditioned on topic."""
    batch = tokens.shape[0]
    t = torch.rand(batch, device=tokens.device)
    masked_tokens, mask_positions = forward_mask(tokens, t)
    logits = model(masked_tokens, t, topic_emb)

    mask_flat = mask_positions.reshape(-1)
    logits_flat = logits.reshape(-1, logits.shape[-1])
    tokens_flat = tokens.reshape(-1)

    if mask_flat.sum() == 0:
        return torch.tensor(0.0, device=tokens.device)

    return F.cross_entropy(logits_flat[mask_flat], tokens_flat[mask_flat])


# ═══════════════════════════════════════════════════════════════════════════
# 6. Topic-Conditioned Sampling
# ═══════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def sample_topic_mdLM(model, seq_len, topic_emb, tok2id,
                      n_samples=20, n_steps=40, temperature=0.7, device=DEVICE):
    """Generate sequences conditioned on a topic embedding."""
    model.eval()
    batch = n_samples

    # Expand topic to batch
    if topic_emb.dim() == 1:
        topic_emb = topic_emb.unsqueeze(0)
    topic_batch = topic_emb.expand(batch, -1).to(device)

    tokens = torch.full((batch, seq_len), MASK, device=device)
    tokens_per_step = max(1, seq_len // n_steps)

    for step in range(n_steps):
        t_val = max(0.01, 1.0 - step / n_steps)
        t = torch.full((batch,), t_val, device=device)

        logits = model(tokens, t, topic_batch)

        mask_positions = (tokens == MASK)

        for b in range(batch):
            masked_idx = mask_positions[b].nonzero(as_tuple=True)[0]
            if len(masked_idx) == 0:
                continue

            pos_logits = logits[b, masked_idx] / temperature
            probs = F.softmax(pos_logits, dim=-1)
            sampled = torch.multinomial(probs, num_samples=1).squeeze(-1)

            confidence = probs.max(dim=-1)[0]
            n_keep = min(tokens_per_step, len(masked_idx))
            top_confident = confidence.topk(n_keep)[1]

            for idx in top_confident:
                pos = masked_idx[idx]
                tokens[b, pos] = sampled[idx]

    return tokens


@torch.no_grad()
def editor_refine_topic(model, tokens, topic_emb, reviewer,
                         tok2id, id2tok, n_steps=20,
                         temperature=0.5, device=DEVICE):
    """Editor refinement conditioned on topic."""
    model.eval()
    reviewer.eval()
    batch = tokens.shape[0]

    if topic_emb.dim() == 1:
        topic_emb = topic_emb.unsqueeze(0)
    topic_batch = topic_emb.expand(batch, -1).to(device)

    scores = torch.sigmoid(reviewer(tokens, topic_batch))

    for step in range(n_steps):
        needs_edit = scores < 0.5
        if not needs_edit.any():
            break

        edited = tokens.clone()
        for b in range(batch):
            if not needs_edit[b]:
                continue
            valid_pos = []
            for pos in range(edited.shape[1]):
                tok = edited[b, pos].item()
                tok_str = id2tok.get(tok, "")
                if tok_str not in ("<pad>", "<bos>", "<eos>"):
                    valid_pos.append(pos)
            if len(valid_pos) >= 2:
                n_mask = min(2, len(valid_pos))
                mask_pos = random.sample(valid_pos, n_mask)
                for p in mask_pos:
                    edited[b, p] = MASK

        t = torch.full((batch,), 0.2, device=device)
        logits = model(edited, t, topic_batch)

        for b in range(batch):
            if not needs_edit[b]:
                continue
            for pos in range(edited.shape[1]):
                if edited[b, pos] == MASK:
                    pos_logits = logits[b, pos] / temperature
                    probs = F.softmax(pos_logits, dim=-1)
                    sampled = torch.multinomial(probs, 1)
                    edited[b, pos] = sampled

        new_scores = torch.sigmoid(reviewer(edited, topic_batch))
        improved = new_scores > scores
        tokens[improved] = edited[improved]
        scores[improved] = new_scores[improved]

    return tokens, scores


# ═══════════════════════════════════════════════════════════════════════════
# 7. Semantic Category Topics (for training)
# ═══════════════════════════════════════════════════════════════════════════

def build_category_topic_embeddings(device=DEVICE):
    """Build synthetic topic embeddings for each vocabulary category.

    During training, each sentence is paired with the topic embedding
    of its dominant category (animals, food, nature, etc.).

    For this experiment, we use one-hot-like embeddings in a 1024D space:
      - Each category gets a unique region of the embedding space
      - This simulates how bge-m3 would embed "animals" vs "food"

    In production, these would come from real bge-m3 embeddings of
    the actual text, via SplatsDB's latent diffusion pipeline.
    """
    categories = list(VOCAB.keys())  # 13 categories
    n_cat = len(categories)

    # Create well-separated embeddings in 1024D
    # Each category gets a random direction, with enough dimensionality
    # for the TopicEncoder to learn meaningful projections
    torch.manual_seed(42)
    cat_embeds = {}

    # Use orthogonal-ish vectors for clear separation
    for i, cat in enumerate(categories):
        emb = torch.zeros(1024, device=device)
        # Assign a block of ~78 dims per category (1024/13≈78)
        block_start = i * (1024 // n_cat)
        block_end = block_start + (1024 // n_cat)
        emb[block_start:block_end] = torch.randn(block_end - block_start, device=device)
        # Add some noise to the rest for richness
        emb += 0.1 * torch.randn(1024, device=device)
        # Normalize (like bge-m3)
        emb = F.normalize(emb, dim=0)
        cat_embeds[cat] = emb

    # Also create a "mixed" embedding (average of all)
    mixed = sum(cat_embeds.values()) / n_cat
    mixed = F.normalize(mixed, dim=0)

    return cat_embeds, categories, mixed


def sentence_to_topic(words: List[str], cat_embeds, categories, mixed,
                       device=DEVICE) -> torch.Tensor:
    """Determine which topic embedding to assign to a sentence.

    Finds the dominant category (most words from that category) and
    returns its embedding. If no clear winner, returns mixed.
    """
    cat_counts = {cat: 0 for cat in categories}
    for w in words:
        for cat, cat_words in VOCAB.items():
            if w in cat_words:
                cat_counts[cat] = cat_counts.get(cat, 0) + 1

    max_cat = max(cat_counts, key=cat_counts.get)
    if cat_counts[max_cat] > 0:
        return cat_embeds[max_cat]

    return mixed


# ═══════════════════════════════════════════════════════════════════════════
# 8. Topic-Conditioned HRM Pipeline
# ═══════════════════════════════════════════════════════════════════════════

def run_topic_conditioned_experiment():
    """Full topic-conditioned HRM experiment.

    Tests whether topic conditioning via cross-attention produces
    text that is semantically consistent with the sampled topic.
    """
    print("=" * 70)
    print("TOPIC-CONDITIONED MDLM (SplatsDB Latent → Cross-Attention → HRM)")
    print("=" * 70)

    # Build vocab and topic embeddings
    tok2id, id2tok, all_words = build_vocab()
    vocab_size = len(all_words)
    cat_embeds, categories, mixed = build_category_topic_embeddings()
    print(f"Vocabulary: {vocab_size} tokens")
    print(f"Categories: {len(categories)} ({', '.join(categories[:5])}, ...)")

    # Generate training data
    cfg_gen = CFGGenerator(seed=42)
    sequences = cfg_gen.generate_dataset(n=5000, seed=42)
    print(f"Training sequences: {len(sequences)}")

    # Assign topic embeddings to each sequence
    topic_embs = torch.stack([
        sentence_to_topic(seq, cat_embeds, categories, mixed)
        for seq in sequences
    ])
    print(f"Topic embeddings: {topic_embs.shape}")

    # Encode sequences
    max_seq_len = 18
    encoded = encode_sequences(sequences, tok2id, max_seq_len + 2).to(DEVICE)
    topic_embs = topic_embs.to(DEVICE)

    # Train/test split
    n_train = int(0.9 * len(encoded))
    train_tokens = encoded[:n_train]
    train_topics = topic_embs[:n_train]
    test_tokens = encoded[n_train:]
    test_topics = topic_embs[n_train:]

    # ── Build models ──────────────────────────────────────────────────
    d_model = 384
    n_heads = 6
    n_layers = 6

    generator = TopicMDLMTransformer(
        vocab_size, topic_dim=1024, d_model=d_model,
        n_heads=n_heads, n_layers=n_layers,
        max_seq_len=max_seq_len + 2, dropout=0.1,
    ).to(DEVICE)

    reviewer = TopicReviewer(
        vocab_size, topic_dim=1024, d_model=256,
        n_heads=4, n_layers=4,
        max_seq_len=max_seq_len + 2, dropout=0.1,
    ).to(DEVICE)

    editor = TopicMDLMTransformer(
        vocab_size, topic_dim=1024, d_model=d_model,
        n_heads=n_heads, n_layers=n_layers,
        max_seq_len=max_seq_len + 2, dropout=0.1,
    ).to(DEVICE)

    gen_params = sum(p.numel() for p in generator.parameters())
    rev_params = sum(p.numel() for p in reviewer.parameters())
    edit_params = sum(p.numel() for p in editor.parameters())
    total = gen_params + rev_params + edit_params
    print(f"\nModels:")
    print(f"  Generator: {gen_params:,} params ({gen_params/1e6:.1f}M)")
    print(f"  Reviewer:  {rev_params:,} params ({rev_params/1e6:.1f}M)")
    print(f"  Editor:    {edit_params:,} params ({edit_params/1e6:.1f}M)")
    print(f"  TOTAL:     {total:,} params ({total/1e6:.1f}M)")

    # ── Train Generator ───────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"TRAINING TOPIC-CONDITIONED GENERATOR ({gen_params:,} params)")
    print(f"{'='*60}")

    opt_g = torch.optim.AdamW(generator.parameters(), lr=3e-4, weight_decay=0.01)
    n_epochs = 2500
    batch_size = 256
    best_test = float('inf')
    best_state = None
    t0 = time.time()

    for epoch in range(n_epochs):
        idx = torch.randint(0, n_train, (batch_size,))
        batch_t = train_tokens[idx]
        batch_e = train_topics[idx]

        opt_g.zero_grad()
        loss = topic_mdlm_loss(generator, batch_t, batch_e)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(generator.parameters(), 1.0)
        opt_g.step()

        if epoch % 300 == 0 or epoch == n_epochs - 1:
            generator.eval()
            with torch.no_grad():
                test_loss = topic_mdlm_loss(generator, test_tokens, test_topics)
            generator.train()
            if test_loss.item() < best_test:
                best_test = test_loss.item()
                best_state = {k: v.clone() for k, v in generator.state_dict().items()}
            print(f"  ep {epoch:4d}: train={loss.item():.4f} test={test_loss.item():.4f} "
                  f"({time.time()-t0:.0f}s)")

    if best_state:
        generator.load_state_dict(best_state)
    print(f"Best generator test loss: {best_test:.4f}")

    # ── Train Reviewer ────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"TRAINING TOPIC-CONDITIONED REVIEWER ({rev_params:,} params)")
    print(f"{'='*60}")

    # Generate negative examples (wrong topic + corrupted)
    rng = random.Random(123)
    # For negatives: pair sentences with WRONG category topics
    neg_topics = []
    for seq in sequences:
        correct_cat = None
        for w in seq:
            for cat, cat_words in VOCAB.items():
                if w in cat_words:
                    correct_cat = cat
                    break
            if correct_cat:
                break
        # Pick a wrong category
        wrong_cats = [c for c in categories if c != correct_cat]
        wrong_cat = rng.choice(wrong_cats)
        neg_topics.append(cat_embeds[wrong_cat])

    neg_topic_embs = torch.stack(neg_topics).to(DEVICE)

    opt_r = torch.optim.AdamW(reviewer.parameters(), lr=3e-4, weight_decay=0.01)
    best_rev_acc = 0.0
    best_rev_state = None
    t0 = time.time()

    for epoch in range(2000):
        # Positive: correct topic
        idx = torch.randint(0, n_train, (batch_size,))
        pos_t = train_tokens[idx]
        pos_e = train_topics[idx]

        # Negative: wrong topic
        neg_idx = torch.randint(0, n_train, (batch_size,))
        neg_t = train_tokens[neg_idx]
        neg_e = neg_topic_embs[neg_idx]

        all_t = torch.cat([pos_t, neg_t])
        all_e = torch.cat([pos_e, neg_e])
        all_y = torch.cat([torch.ones(batch_size), torch.zeros(batch_size)]).to(DEVICE)

        opt_r.zero_grad()
        logits = reviewer(all_t, all_e)
        loss = F.binary_cross_entropy_with_logits(logits, all_y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(reviewer.parameters(), 1.0)
        opt_r.step()

        if epoch % 300 == 0 or epoch == 1999:
            reviewer.eval()
            with torch.no_grad():
                # Test on held-out
                test_pos = reviewer(test_tokens[:50], test_topics[:50])
                test_neg = reviewer(test_tokens[:50], neg_topic_embs[:50])
                test_logits = torch.cat([test_pos, test_neg])
                test_y = torch.cat([torch.ones(50), torch.zeros(50)]).to(DEVICE)
                test_loss = F.binary_cross_entropy_with_logits(test_logits, test_y)
                preds = (test_logits > 0).float()
                acc = (preds == test_y).float().mean().item()
            reviewer.train()
            if acc > best_rev_acc:
                best_rev_acc = acc
                best_rev_state = {k: v.clone() for k, v in reviewer.state_dict().items()}
            print(f"  ep {epoch:4d}: train={loss.item():.4f} test={test_loss.item():.4f} "
                  f"acc={acc:.1%} ({time.time()-t0:.0f}s)")

    if best_rev_state:
        reviewer.load_state_dict(best_rev_state)
    print(f"Best reviewer accuracy: {best_rev_acc:.1%}")

    # ── Train Editor ──────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"TRAINING TOPIC-CONDITIONED EDITOR ({edit_params:,} params)")
    print(f"{'='*60}")

    editor_seqs = cfg_gen.generate_dataset(n=5000, seed=99)
    editor_encoded = encode_sequences(editor_seqs, tok2id, max_seq_len + 2).to(DEVICE)
    editor_topics = torch.stack([
        sentence_to_topic(seq, cat_embeds, categories, mixed)
        for seq in editor_seqs
    ]).to(DEVICE)

    n_edit_train = int(0.9 * len(editor_encoded))
    edit_train_t = editor_encoded[:n_edit_train]
    edit_train_e = editor_topics[:n_edit_train]
    edit_test_t = editor_encoded[n_edit_train:]
    edit_test_e = editor_topics[n_edit_train:]

    opt_e = torch.optim.AdamW(editor.parameters(), lr=3e-4, weight_decay=0.01)
    best_edit = float('inf')
    best_edit_state = None
    t0 = time.time()

    for epoch in range(2000):
        idx = torch.randint(0, n_edit_train, (batch_size,))
        batch_t = edit_train_t[idx]
        batch_e = edit_train_e[idx]

        opt_e.zero_grad()
        loss = topic_mdlm_loss(editor, batch_t, batch_e)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(editor.parameters(), 1.0)
        opt_e.step()

        if epoch % 300 == 0 or epoch == 1999:
            editor.eval()
            with torch.no_grad():
                test_loss = topic_mdlm_loss(editor, edit_test_t, edit_test_e)
            editor.train()
            if test_loss.item() < best_edit:
                best_edit = test_loss.item()
                best_edit_state = {k: v.clone() for k, v in editor.state_dict().items()}
            print(f"  ep {epoch:4d}: train={loss.item():.4f} test={test_loss.item():.4f} "
                  f"({time.time()-t0:.0f}s)")

    if best_edit_state:
        editor.load_state_dict(best_edit_state)
    print(f"Best editor test loss: {best_edit:.4f}")

    # ── EVALUATION: Topic-Conditioned Generation ─────────────────────
    print(f"\n{'='*70}")
    print(f"EVALUATION: TOPIC-CONDITIONED GENERATION")
    print(f"{'='*70}")

    # For each category, generate 10 sequences conditioned on that topic
    # and check how many words belong to the target category
    results_by_cat = {}
    seq_len = max_seq_len + 2

    for target_cat in categories:
        topic = cat_embeds[target_cat]

        # Generate
        samples = sample_topic_mdLM(
            generator, seq_len, topic, tok2id,
            n_samples=10, n_steps=40, temperature=0.7,
        )

        # Review
        topic_batch = topic.unsqueeze(0).expand(10, -1).to(DEVICE)
        scores = torch.sigmoid(reviewer(samples, topic_batch))

        # Decode and analyze
        decoded = decode_tokens(samples, id2tok)
        target_words = set(VOCAB[target_cat])

        on_topic_count = 0
        total_content_words = 0

        for seq_str in decoded:
            words = [w for w in seq_str.split() if w != "[M]"]
            for w in words:
                # Check if word is a content word (not function word)
                is_content = any(w in cat_words for cat_words in VOCAB.values())
                if is_content:
                    total_content_words += 1
                    if w in target_words:
                        on_topic_count += 1

        on_topic_rate = on_topic_count / max(1, total_content_words)
        results_by_cat[target_cat] = {
            "on_topic_rate": on_topic_rate,
            "n_content_words": total_content_words,
            "n_on_topic": on_topic_count,
            "mean_score": scores.mean().item(),
            "samples": decoded[:3],
        }

        # Print sample
        print(f"\n  ── {target_cat.upper()} ──")
        print(f"    On-topic rate: {on_topic_rate:.1%} ({on_topic_count}/{total_content_words} content words)")
        print(f"    Reviewer score: {scores.mean().item():.3f}")
        for i, s in enumerate(decoded[:3]):
            print(f"    [{scores[i].item():.2f}] {s}")

    # ── BASELINE: No topic (unconditioned) ───────────────────────────
    print(f"\n  ── BASELINE (unconditioned, mixed topic) ──")
    baseline_samples = sample_topic_mdLM(
        generator, seq_len, mixed, tok2id,
        n_samples=30, n_steps=40, temperature=0.7,
    )
    baseline_decoded = decode_tokens(baseline_samples, id2tok)

    # Count category distribution in baseline
    baseline_cat_counts = {cat: 0 for cat in categories}
    total_words = 0
    for seq_str in baseline_decoded:
        for w in seq_str.split():
            if w == "[M]":
                continue
            for cat, cat_words in VOCAB.items():
                if w in cat_words:
                    baseline_cat_counts[cat] += 1
                    total_words += 1
                    break

    print(f"    Word distribution across categories:")
    for cat in categories:
        pct = baseline_cat_counts[cat] / max(1, total_words)
        print(f"      {cat:15s}: {baseline_cat_counts[cat]:3d} ({pct:.1%})")

    # ── SUMMARY ──────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"SUMMARY: TOPIC CONDITIONING EFFECT")
    print(f"{'='*70}")

    # Expected on-topic rate without conditioning: uniform = 1/13 ≈ 7.7%
    uniform_rate = 1.0 / len(categories)

    print(f"\n  Without conditioning (uniform baseline): {uniform_rate:.1%}")
    print(f"\n  With topic conditioning:")
    for cat in categories:
        r = results_by_cat[cat]
        improvement = r["on_topic_rate"] / uniform_rate
        print(f"    {cat:15s}: {r['on_topic_rate']:.1%} "
              f"({improvement:.1f}x uniform) "
              f"[score={r['mean_score']:.3f}]")

    # Average improvement
    avg_on_topic = np.mean([results_by_cat[c]["on_topic_rate"] for c in categories])
    avg_improvement = avg_on_topic / uniform_rate
    print(f"\n  Average on-topic rate: {avg_on_topic:.1%} ({avg_improvement:.1f}x uniform)")

    # Save results
    result = {
        "experiment": "topic_conditioned_mdlm",
        "timestamp": datetime.now().isoformat(),
        "vocab_size": vocab_size,
        "n_categories": len(categories),
        "model_params": {
            "generator": gen_params,
            "reviewer": rev_params,
            "editor": edit_params,
            "total": total,
        },
        "uniform_baseline_rate": uniform_rate,
        "avg_on_topic_rate": float(avg_on_topic),
        "avg_improvement_x": float(avg_improvement),
        "results_by_category": {
            cat: {
                "on_topic_rate": r["on_topic_rate"],
                "n_content_words": r["n_content_words"],
                "n_on_topic": r["n_on_topic"],
                "mean_score": r["mean_score"],
                "samples": r["samples"],
            }
            for cat, r in results_by_cat.items()
        },
        "baseline_distribution": baseline_cat_counts,
    }

    out = RESULTS_DIR / "topic_conditioned_results.json"
    with open(out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nResults saved to {out}")

    # Save models
    torch.save({
        "generator": generator.state_dict(),
        "reviewer": reviewer.state_dict(),
        "editor": editor.state_dict(),
        "tok2id": tok2id,
        "id2tok": id2tok,
        "cat_embeds": cat_embeds,
        "categories": categories,
    }, RESULTS_DIR / "topic_conditioned_models.pt")
    print(f"Models saved to {RESULTS_DIR / 'topic_conditioned_models.pt'}")


if __name__ == "__main__":
    run_topic_conditioned_experiment()
