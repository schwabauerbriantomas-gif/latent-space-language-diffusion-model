"""
HRM (Hierarchical Role Model) Pipeline for Text Generation & Editing.

Three specialized transformer models working in sequence:

  1. GENERATOR: MDLM that generates text from scratch (all-MASK → text)
  2. REVIEWER: Classifier that scores grammaticality + coherence [0,1]
  3. EDITOR:   MDLM that takes a partially-masked sequence and fills/refines it

Pipeline:
  GENERATOR → candidate sentences
      ↓
  REVIEWER  → score each candidate [0,1]
      ↓
  EDITOR    → take low-score candidates, mask worst positions, regenerate
      ↓
  Final output: highest-scoring sentences after up to N editing rounds

This mirrors a human writing process:
  Draft → Review → Edit → Final

ARCHITECTURE:
  All three models share the same MDLMTransformer architecture
  (from mdlm.py), but are trained on different objectives:

  - Generator: standard MDLM loss (predict masked tokens)
  - Reviewer:  binary classification (grammatical vs ungrammatical)
  - Editor:    partial-mask loss (only mask positions that need fixing)

WHY HRM:
  A single MDLM can generate but cannot self-correct.
  The Reviewer provides a learned signal of quality.
  The Editor specializes in refinement, not generation from scratch.
  This separation of concerns produces higher-quality output.
"""
import math
import json
import sys
import time
import random
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RESULTS_DIR = REPO / "results"

# Import from existing modules
from vocab_cfg import (
    build_vocab, CFGGenerator, tag_sequence, check_grammar,
    grammar_score, get_pos, PAD, MASK, BOS, EOS, UNK,
    ALL_NOUNS, ALL_ADJ, FUNC, VOCAB,
)
from mdlm import MDLMTransformer, forward_mask, encode_sequences, decode_tokens


# ═══════════════════════════════════════════════════════════════════════════
# 1. REVIEWER MODEL — Grammaticality Classifier
# ═══════════════════════════════════════════════════════════════════════════

class ReviewerModel(nn.Module):
    """BERT-style classifier that scores grammaticality of a sentence.

    Architecture: same transformer encoder as MDLM, but with a
    classification head on the [BOS] token (like BERT's [CLS]).

    Input:  token sequence
    Output: single score [0, 1] (1 = grammatical, 0 = ungrammatical)

    Training data: positive examples from CFG, negative from corrupted CFG.
    """

    def __init__(self, vocab_size, d_model=256, n_heads=4, n_layers=4,
                 max_seq_len=32, dropout=0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model

        self.token_emb = nn.Embedding(vocab_size, d_model, padding_idx=PAD)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4,
            dropout=dropout, activation="gelu", batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, n_layers)
        self.ln_f = nn.LayerNorm(d_model)

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

    def forward(self, tokens):
        """Returns logits [batch] (positive = grammatical)."""
        batch, seq_len = tokens.shape
        pos = torch.arange(seq_len, device=tokens.device).unsqueeze(0).expand(batch, -1)
        x = self.token_emb(tokens) + self.pos_emb(pos)

        pad_mask = (tokens == PAD)
        x = self.transformer(x, src_key_padding_mask=pad_mask)

        # Use BOS token representation for classification
        cls_repr = self.ln_f(x[:, 0, :])  # [batch, d_model]
        logits = self.classifier(cls_repr).squeeze(-1)  # [batch]
        return logits


def generate_negative_examples(positive_seqs: List[List[str]],
                                rng: random.Random) -> List[List[str]]:
    """Generate ungrammatical sequences by corrupting positive examples.

    Corruption strategies:
      1. Swap two random tokens (breaks word order)
      2. Delete a function word (breaks structure)
      3. Insert a random word in wrong position
      4. Replace a verb with a noun (POS violation)
      5. Shuffle a random subsequence
    """
    negatives = []
    all_words = []
    for cat_words in VOCAB.values():
        all_words.extend(cat_words)
    for func_words in FUNC.values():
        all_words.extend(func_words)
    all_words = list(set(all_words))

    nouns = ALL_NOUNS
    adjs = ALL_ADJ
    verbs = FUNC["verbs"]
    dets = FUNC["determiners"]

    for seq in positive_seqs:
        if len(seq) < 4:
            continue
        corrupted = list(seq)
        strategy = rng.randint(0, 4)

        if strategy == 0 and len(corrupted) > 3:
            # Swap two random positions
            i, j = rng.sample(range(len(corrupted)), 2)
            corrupted[i], corrupted[j] = corrupted[j], corrupted[i]

        elif strategy == 1:
            # Delete a random function word
            func_indices = [k for k, w in enumerate(corrupted)
                           if w in FUNC["determiners"] or w in FUNC["prepositions"]]
            if func_indices:
                del_idx = rng.choice(func_indices)
                corrupted.pop(del_idx)

        elif strategy == 2:
            # Insert random word at random position
            pos_insert = rng.randint(1, len(corrupted) - 1)
            corrupted.insert(pos_insert, rng.choice(all_words))

        elif strategy == 3:
            # Replace a verb with a noun (POS violation)
            verb_indices = [k for k, w in enumerate(corrupted) if w in verbs]
            if verb_indices:
                idx = rng.choice(verb_indices)
                corrupted[idx] = rng.choice(nouns)

        else:
            # Shuffle a subsequence of length 3
            if len(corrupted) >= 5:
                start = rng.randint(0, len(corrupted) - 3)
                sub = corrupted[start:start+3]
                rng.shuffle(sub)
                corrupted[start:start+3] = sub

        negatives.append(corrupted)
    return negatives


# ═══════════════════════════════════════════════════════════════════════════
# 2. EDITOR MODEL — Partial Refinement
# ═══════════════════════════════════════════════════════════════════════════

def editor_loss(model, tokens, tok2id, corruption_rate=0.3):
    """Editor training loss: mask only suspicious positions and predict.

    Unlike the generator (which masks uniformly), the editor learns to fix
    specific positions. Training: mask random positions (simulating errors),
    predict the original.
    """
    batch, seq_len = tokens.shape
    t = torch.full((batch,), corruption_rate, device=tokens.device)

    masked_tokens, mask_positions = forward_mask(tokens, t)
    logits = model(masked_tokens, t)

    mask_flat = mask_positions.reshape(-1)
    logits_flat = logits.reshape(-1, logits.shape[-1])
    tokens_flat = tokens.reshape(-1)

    if mask_flat.sum() == 0:
        return torch.tensor(0.0, device=tokens.device)

    return F.cross_entropy(logits_flat[mask_flat], tokens_flat[mask_flat])


# ═══════════════════════════════════════ edit sampling ═════════════════════

@torch.no_grad()
def editor_refine(model, tokens, reviewer, tok2id, id2tok,
                  n_steps=20, temperature=0.5, device=DEVICE):
    """Use the editor to refine a batch of sequences.

    Process:
      1. Score each sequence with reviewer
      2. For low-scoring sequences, mask the worst positions
      3. Regenerate masked positions
      4. Re-score
    """
    model.eval()
    reviewer.eval()
    batch = tokens.shape[0]

    # Initial score
    scores = torch.sigmoid(reviewer(tokens))  # [batch]

    for step in range(n_steps):
        # Find sequences below threshold
        needs_edit = scores < 0.5
        if not needs_edit.any():
            break

        # Mask 1-2 random positions in low-scoring sequences
        edited = tokens.clone()
        for b in range(batch):
            if not needs_edit[b]:
                continue
            # Mask 1-2 non-special positions
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

        # Regenerate masked positions
        t = torch.full((batch,), 0.2, device=device)
        logits = model(edited, t)

        for b in range(batch):
            if not needs_edit[b]:
                continue
            for pos in range(edited.shape[1]):
                if edited[b, pos] == MASK:
                    pos_logits = logits[b, pos] / temperature
                    probs = F.softmax(pos_logits, dim=-1)
                    sampled = torch.multinomial(probs, 1)
                    edited[b, pos] = sampled

        # Re-score
        new_scores = torch.sigmoid(reviewer(edited))
        # Accept only if improved
        improved = new_scores > scores
        tokens[improved] = edited[improved]
        scores[improved] = new_scores[improved]

    return tokens, scores


# ═══════════════════════════════════════════════════════════════════════════
# 3. HRM PIPELINE — Orchestration
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class HRMConfig:
    """Configuration for the HRM pipeline."""
    # Generator
    gen_d_model: int = 384
    gen_n_heads: int = 6
    gen_n_layers: int = 6
    gen_epochs: int = 3000
    gen_batch: int = 256

    # Reviewer
    rev_d_model: int = 256
    rev_n_heads: int = 4
    rev_n_layers: int = 4
    rev_epochs: int = 2000
    rev_batch: int = 256

    # Editor
    edit_d_model: int = 384
    edit_n_heads: int = 6
    edit_n_layers: int = 6
    edit_epochs: int = 2000
    edit_batch: int = 256

    # Data
    n_train_seqs: int = 5000
    max_seq_len: int = 18

    # HRM
    n_candidates: int = 20
    n_edit_rounds: int = 3
    temperature: float = 0.7
    reviewer_threshold: float = 0.5


class HRMPipeline:
    """Full HRM pipeline: Generator + Reviewer + Editor.

    Usage:
        pipeline = HRMPipeline()
        pipeline.train_all()
        results = pipeline.generate(n=100)
    """

    def __init__(self, config: Optional[HRMConfig] = None):
        self.config = config or HRMConfig()
        self.tok2id, self.id2tok, self.all_words = build_vocab()
        self.vocab_size = len(self.all_words)
        self.generator: Optional[MDLMTransformer] = None
        self.reviewer: Optional[ReviewerModel] = None
        self.editor: Optional[MDLMTransformer] = None
        self.cfg_gen = CFGGenerator(seed=42)

    # ── Training ──────────────────────────────────────────────────────

    def _build_model(self, d_model, n_heads, n_layers):
        return MDLMTransformer(
            vocab_size=self.vocab_size,
            d_model=d_model, n_heads=n_heads, n_layers=n_layers,
            max_seq_len=self.config.max_seq_len + 2, dropout=0.1,
        ).to(DEVICE)

    def _build_reviewer(self, d_model, n_heads, n_layers):
        return ReviewerModel(
            vocab_size=self.vocab_size,
            d_model=d_model, n_heads=n_heads, n_layers=n_layers,
            max_seq_len=self.config.max_seq_len + 2, dropout=0.1,
        ).to(DEVICE)

    def train_generator(self, train_data, test_data, verbose=True):
        """Train the generator with MDLM loss."""
        cfg = self.config
        model = self._build_model(cfg.gen_d_model, cfg.gen_n_heads, cfg.gen_n_layers)
        n_params = sum(p.numel() for p in model.parameters())
        if verbose:
            print(f"\n{'='*60}")
            print(f"TRAINING GENERATOR ({n_params:,} params, {cfg.gen_epochs} epochs)")
            print(f"{'='*60}")

        opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
        n_train = len(train_data)
        best_test = float('inf')
        best_state = None
        t0 = time.time()

        for epoch in range(cfg.gen_epochs):
            idx = torch.randint(0, n_train, (cfg.gen_batch,))
            batch = train_data[idx]
            opt.zero_grad()
            from mdlm import mdlm_loss
            loss = mdlm_loss(model, batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            if epoch % 300 == 0 or epoch == cfg.gen_epochs - 1:
                model.eval()
                with torch.no_grad():
                    test_loss = mdlm_loss(model, test_data)
                model.train()
                if test_loss.item() < best_test:
                    best_test = test_loss.item()
                    best_state = {k: v.clone() for k, v in model.state_dict().items()}
                if verbose:
                    print(f"  ep {epoch:4d}: train={loss.item():.4f} test={test_loss.item():.4f} "
                          f"({time.time()-t0:.0f}s)")

        if best_state:
            model.load_state_dict(best_state)
        self.generator = model
        return model

    def train_reviewer(self, positive_seqs, negative_seqs, verbose=True):
        """Train the reviewer to classify grammatical vs ungrammatical."""
        cfg = self.config

        # Encode positive and negative
        pos_encoded = encode_sequences(positive_seqs, self.tok2id, cfg.max_seq_len + 2).to(DEVICE)
        neg_encoded = encode_sequences(negative_seqs, self.tok2id, cfg.max_seq_len + 2).to(DEVICE)

        # Split train/test
        n_pos = len(pos_encoded)
        n_neg = len(neg_encoded)
        n_pos_train = int(0.9 * n_pos)
        n_neg_train = int(0.9 * n_neg)

        train_x = torch.cat([pos_encoded[:n_pos_train], neg_encoded[:n_neg_train]])
        train_y = torch.cat([torch.ones(n_pos_train), torch.zeros(n_neg_train)]).to(DEVICE)
        test_x = torch.cat([pos_encoded[n_pos_train:], neg_encoded[n_neg_train:]])
        test_y = torch.cat([torch.ones(n_pos - n_pos_train), torch.zeros(n_neg - n_neg_train)]).to(DEVICE)

        # Shuffle
        perm = torch.randperm(len(train_x))
        train_x, train_y = train_x[perm], train_y[perm]

        model = self._build_reviewer(cfg.rev_d_model, cfg.rev_n_heads, cfg.rev_n_layers)
        n_params = sum(p.numel() for p in model.parameters())
        if verbose:
            print(f"\n{'='*60}")
            print(f"TRAINING REVIEWER ({n_params:,} params, {cfg.rev_epochs} epochs)")
            print(f"{'='*60}")

        opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
        t0 = time.time()
        n_train = len(train_x)

        for epoch in range(cfg.rev_epochs):
            idx = torch.randint(0, n_train, (min(cfg.rev_batch, n_train),))
            batch_x = train_x[idx]
            batch_y = train_y[idx]
            opt.zero_grad()
            logits = model(batch_x)
            loss = F.binary_cross_entropy_with_logits(logits, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            if epoch % 300 == 0 or epoch == cfg.rev_epochs - 1:
                model.eval()
                with torch.no_grad():
                    test_logits = model(test_x)
                    test_loss = F.binary_cross_entropy_with_logits(test_logits, test_y)
                    preds = (test_logits > 0).float()
                    acc = (preds == test_y).float().mean().item()
                model.train()
                if verbose:
                    print(f"  ep {epoch:4d}: train={loss.item():.4f} test={test_loss.item():.4f} "
                          f"acc={acc:.1%} ({time.time()-t0:.0f}s)")

        self.reviewer = model
        return model

    def train_editor(self, train_data, test_data, verbose=True):
        """Train the editor with partial-mask loss."""
        cfg = self.config
        model = self._build_model(cfg.edit_d_model, cfg.edit_n_heads, cfg.edit_n_layers)
        n_params = sum(p.numel() for p in model.parameters())
        if verbose:
            print(f"\n{'='*60}")
            print(f"TRAINING EDITOR ({n_params:,} params, {cfg.edit_epochs} epochs)")
            print(f"{'='*60}")

        opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
        n_train = len(train_data)
        best_test = float('inf')
        best_state = None
        t0 = time.time()

        for epoch in range(cfg.edit_epochs):
            idx = torch.randint(0, n_train, (cfg.edit_batch,))
            batch = train_data[idx]
            opt.zero_grad()
            loss = editor_loss(model, batch, self.tok2id, corruption_rate=0.3)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            if epoch % 300 == 0 or epoch == cfg.edit_epochs - 1:
                model.eval()
                with torch.no_grad():
                    test_loss = editor_loss(model, test_data, self.tok2id, corruption_rate=0.3)
                model.train()
                if test_loss.item() < best_test:
                    best_test = test_loss.item()
                    best_state = {k: v.clone() for k, v in model.state_dict().items()}
                if verbose:
                    print(f"  ep {epoch:4d}: train={loss.item():.4f} test={test_loss.item():.4f} "
                          f"({time.time()-t0:.0f}s)")

        if best_state:
            model.load_state_dict(best_state)
        self.editor = model
        return model

    def train_all(self, verbose=True):
        """Train all three models sequentially."""
        cfg = self.config

        # Generate training data
        if verbose:
            print("Generating training data...")
        positive_seqs = self.cfg_gen.generate_dataset(n=cfg.n_train_seqs, seed=42)
        # Also generate a separate set for editor (different seed for diversity)
        editor_seqs = self.cfg_gen.generate_dataset(n=cfg.n_train_seqs, seed=99)

        # Encode positive sequences
        train_encoded = encode_sequences(positive_seqs, self.tok2id, cfg.max_seq_len + 2).to(DEVICE)
        editor_encoded = encode_sequences(editor_seqs, self.tok2id, cfg.max_seq_len + 2).to(DEVICE)

        # Train/test split
        n_train = int(0.9 * len(train_encoded))
        train_data = train_encoded[:n_train]
        test_data = train_encoded[n_train:]

        n_edit_train = int(0.9 * len(editor_encoded))
        edit_train = editor_encoded[:n_edit_train]
        edit_test = editor_encoded[n_edit_train:]

        # Generate negative examples for reviewer
        rng = random.Random(123)
        negative_seqs = generate_negative_examples(positive_seqs[:2000], rng)

        if verbose:
            print(f"  Positive sequences: {len(positive_seqs)}")
            print(f"  Negative sequences: {len(negative_seqs)}")
            print(f"  Vocab size: {self.vocab_size}")

        # Train Generator
        self.train_generator(train_data, test_data, verbose)

        # Train Reviewer
        self.train_reviewer(positive_seqs, negative_seqs, verbose)

        # Train Editor
        self.train_editor(edit_train, edit_test, verbose)

        return self

    # ── Generation Pipeline ──────────────────────────────────────────

    @torch.no_grad()
    def generate(self, n: int = 100, verbose=True) -> Dict:
        """Full HRM generation pipeline.

        Steps:
          1. Generator produces n candidates
          2. Reviewer scores each
          3. Editor refines low-scoring candidates
          4. Return statistics + best results
        """
        cfg = self.config
        self.generator.eval()
        self.reviewer.eval()
        self.editor.eval()

        if verbose:
            print(f"\n{'='*60}")
            print(f"HRM GENERATION PIPELINE")
            print(f"{'='*60}")

        # Step 1: Generate candidates
        from mdlm import sample_mdLM
        seq_len = cfg.max_seq_len + 2
        samples = sample_mdLM(
            self.generator, seq_len=seq_len, tok2id=self.tok2id,
            n_samples=n, n_steps=40, temperature=cfg.temperature,
        )
        decoded = decode_tokens(samples, self.id2tok)

        if verbose:
            print(f"\n--- Step 1: Generator produced {n} candidates ---")
            for i in range(min(10, n)):
                print(f"  {i:2d}: {decoded[i]}")

        # Step 2: Reviewer scores
        scores_gen = torch.sigmoid(self.reviewer(samples))  # [n]
        if verbose:
            n_pass_gen = (scores_gen > cfg.reviewer_threshold).sum().item()
            print(f"\n--- Step 2: Reviewer scoring ---")
            print(f"  Passed threshold (>={cfg.reviewer_threshold}): {n_pass_gen}/{n}")
            print(f"  Mean score: {scores_gen.mean().item():.3f}")
            print(f"  Score range: [{scores_gen.min().item():.3f}, {scores_gen.max().item():.3f}]")

        # Step 3: Editor refines low-scoring candidates
        if cfg.n_edit_rounds > 0 and self.editor is not None:
            if verbose:
                print(f"\n--- Step 3: Editor refinement ({cfg.n_edit_rounds} rounds) ---")
            refined, scores_refined = editor_refine(
                self.editor, samples.clone(), self.reviewer,
                self.tok2id, self.id2tok,
                n_steps=cfg.n_edit_rounds * 5, temperature=0.5,
            )
            decoded_refined = decode_tokens(refined, self.id2tok)

            # Show improvements
            improvements = (scores_refined > scores_gen).sum().item()
            if verbose:
                print(f"  Improved by editor: {improvements}/{n}")
                print(f"  Mean score after edit: {scores_refined.mean().item():.3f}")
                print(f"  Passed threshold: {(scores_refined > cfg.reviewer_threshold).sum().item()}/{n}")

            # Show some before/after examples
            if verbose:
                print(f"\n  Before → After (editor improvements):")
                shown = 0
                for b in range(n):
                    if scores_refined[b] > scores_gen[b] + 0.05 and shown < 10:
                        print(f"    [{scores_gen[b].item():.2f}→{scores_refined[b].item():.2f}] "
                              f"{decoded[b]}")
                        print(f"         → {decoded_refined[b]}")
                        shown += 1

            samples = refined
            decoded = decoded_refined
            scores = scores_refined
        else:
            scores = scores_gen

        # Final analysis
        if verbose:
            print(f"\n{'='*60}")
            print(f"FINAL RESULTS")
            print(f"{'='*60}")

        # Grammar check with CFG checker
        unique_seqs = set()
        grammatical = 0
        for seq_str in decoded:
            words = seq_str.split()
            words = [w for w in words if w != "[M]"]
            if words:
                unique_seqs.add(" ".join(words))
                ok, _ = check_grammar(words)
                if ok:
                    grammatical += 1

        # Novel check (not in training data)
        train_set = set()
        for seq in self.cfg_gen.generate_dataset(n=500, seed=42):
            train_set.add(" ".join(seq))

        novel = unique_seqs - train_set

        result = {
            "experiment": "hrm_pipeline",
            "timestamp": datetime.now().isoformat(),
            "vocab_size": self.vocab_size,
            "n_generated": n,
            "n_unique": len(unique_seqs),
            "n_grammatical": grammatical,
            "n_novel": len(novel),
            "mean_reviewer_score": scores.mean().item(),
            "pass_rate": (scores > cfg.reviewer_threshold).float().mean().item(),
            "sample_sequences": sorted(unique_seqs)[:50],
            "config": {
                "gen_params": sum(p.numel() for p in self.generator.parameters()),
                "rev_params": sum(p.numel() for p in self.reviewer.parameters()),
                "edit_params": sum(p.numel() for p in self.editor.parameters()) if self.editor else 0,
            },
        }

        if verbose:
            print(f"  Total generated: {n}")
            print(f"  Unique sequences: {len(unique_seqs)}")
            print(f"  Grammatical (CFG check): {grammatical}/{n} ({grammatical/n:.1%})")
            print(f"  Novel (not in training): {len(novel)}/{len(unique_seqs)}")
            print(f"  Mean reviewer score: {scores.mean().item():.3f}")
            print(f"  Pass rate (≥{cfg.reviewer_threshold}): {(scores > cfg.reviewer_threshold).float().mean().item():.1%}")
            print(f"\n  Top 20 highest-scored unique sequences:")
            scored = list(zip(decoded, scores.tolist()))
            scored.sort(key=lambda x: -x[1])
            shown = set()
            count = 0
            for seq, score in scored:
                if seq not in shown and count < 20:
                    shown.add(seq)
                    words = [w for w in seq.split() if w != "[M]"]
                    ok, _ = check_grammar(words)
                    mark = "✓" if ok else "✗"
                    print(f"    {mark} [{score:.2f}] {seq}")
                    count += 1

        return result


# ═══════════════════════════════════════════════════════ NP-Subject check

def run_hrm_experiment():
    """Run the full HRM pipeline experiment."""
    config = HRMConfig(
        gen_d_model=384, gen_n_heads=6, gen_n_layers=6, gen_epochs=2000,
        rev_d_model=256, rev_n_heads=4, rev_n_layers=4, rev_epochs=2000,
        edit_d_model=384, edit_n_heads=6, edit_n_layers=6, edit_epochs=2000,
        n_train_seqs=5000, max_seq_len=18,
        n_candidates=100, n_edit_rounds=3, temperature=0.7,
    )

    pipeline = HRMPipeline(config)
    pipeline.train_all(verbose=True)

    # Save models
    torch.save({
        "generator": pipeline.generator.state_dict(),
        "reviewer": pipeline.reviewer.state_dict(),
        "editor": pipeline.editor.state_dict() if pipeline.editor else None,
        "tok2id": pipeline.tok2id,
        "id2tok": pipeline.id2tok,
    }, RESULTS_DIR / "hrm_models.pt")

    # Generate
    result = pipeline.generate(n=100, verbose=True)

    out = RESULTS_DIR / "hrm_results.json"
    with open(out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nResults saved to {out}")


if __name__ == "__main__":
    run_hrm_experiment()
