"""
Masked Diffusion Language Model (MDLM) for SplatsDB.

WHY THIS EXISTS:
  - FF failed (energy collapse → adversarial gradients)
  - Score matching in 1024D works for embeddings but not sequences
  - MDLM works natively on DISCRETE tokens with well-defined cross-entropy loss

ARCHITECTURE:
  Forward process:  [t1, t2, t3, t4] → [M, t2, M, t4] → [M, M, M, M]
                    (progressive masking, like BERT)
  Reverse process:  [M, M, M, M] → predict each masked token conditioned on others
                    (transformer with attention)

  Model: transformer encoder
    Input:  token embeddings + position + mask embedding + timestep
    Output: logits over vocabulary for each position

  Training: L = E_t [ Σ_i cross_entropy(pred_i, true_i) for masked positions ]
  Sampling: start from all-MASK, iteratively unmask lowest-confidence first

INTEGRATION WITH LATENT DIFFUSION:
  - Latent diffusion (PCA+SVGD) generates a "semantic topic" embedding
  - That embedding conditions the transformer (cross-attention)
  - MDLM generates tokens consistent with that topic
  - Two-level: global semantics (latent) + local syntax (masked diffusion)

Math (Sahoo et al. 2024, arXiv:2406.03709):
  Forward: q(x_t | x_0) = Π_i Cat(x_t_i | (1-β_t) e_{x_0_i} + β_t e_MASK)
  Reverse: p_θ(x_0 | x_t) = softmax(transformer(x_t, t))
  Loss:    L = E_t E_{x_t~q} [ Σ_i CE(pred_θ_i, x_0_i) ]  (only masked positions)
"""
import math
import json
import sys
import time
import random
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RESULTS_DIR = REPO / "results"


# ═══════════════════════════════════════════════════════════════════════════
# Vocabulary & Training Data
# ═══════════════════════════════════════════════════════════════════════════

# Special tokens
PAD, MASK, BOS, EOS, UNK = 0, 1, 2, 3, 4
SPECIAL_TOKENS = ["<pad>", "<mask>", "<bos>", "<eos>", "<unk>"]

# Vocabulary: 5 semantic categories, 20 words each
VOCAB_WORDS = {
    "animals": ["cat", "dog", "bird", "fish", "horse", "cow", "sheep", "chicken",
                "rabbit", "mouse", "bear", "lion", "tiger", "wolf", "deer", "elephant",
                "monkey", "snake", "turtle", "duck"],
    "colors": ["red", "blue", "green", "yellow", "white", "black", "orange", "purple",
               "pink", "brown", "gray", "silver", "gold", "cyan", "magenta", "crimson",
               "scarlet", "azure", "emerald", "amber"],
    "food": ["bread", "milk", "cheese", "apple", "banana", "rice", "pasta",
             "meat", "egg", "sugar", "salt", "honey", "coffee", "tea",
             "wine", "beer", "soup", "salad", "tomato", "potato"],
    "emotion": ["happy", "sad", "angry", "calm", "excited", "afraid", "brave", "tired",
                "bored", "amazed", "proud", "grateful", "lonely", "hopeful",
                "worried", "relaxed", "confused", "surprised", "joyful", "peaceful"],
    "nature": ["sun", "moon", "star", "fire", "earth", "sky", "cloud", "rain", "snow",
               "wind", "storm", "mountain", "river", "lake", "ocean", "forest",
               "desert", "valley", "island", "cave"],
}

# Grammar words (function words for coherent sentences)
FUNCTION_WORDS = {
    "determiners": ["the", "a", "an", "this", "that", "these", "those"],
    "prepositions": ["in", "on", "at", "under", "over", "near", "by", "with", "from", "to"],
    "verbs": ["is", "are", "was", "were", "runs", "jumps", "eats", "drinks", "sees",
              "likes", "wants", "goes", "comes", "plays", "sleeps", "sings"],
    "adjectives_extra": ["big", "small", "old", "new", "fast", "slow", "hot", "cold",
                         "bright", "dark", "beautiful", "wild", "gentle", "fierce"],
}


def build_vocab():
    """Build token→id and id→token mappings."""
    words = list(SPECIAL_TOKENS)
    for cat_words in VOCAB_WORDS.values():
        words.extend(cat_words)
    for func_words in FUNCTION_WORDS.values():
        words.extend(func_words)
    words = list(dict.fromkeys(words))  # dedup
    tok2id = {w: i for i, w in enumerate(words)}
    id2tok = {i: w for w, i in tok2id.items()}
    return tok2id, id2tok, words


def generate_training_sequences(n=2000, seed=42):
    """Generate grammatically correct short sequences.

    Templates ensure syntactic coherence:
      [det] [adj] [noun] [verb] [prep] [det] [noun]
      e.g., "the happy cat drinks cold milk"
    """
    random.seed(seed)

    all_nouns = []
    for cat in ["animals", "food", "nature"]:
        all_nouns.extend(VOCAB_WORDS[cat])

    all_adj = VOCAB_WORDS["colors"] + VOCAB_WORDS["emotion"] + FUNCTION_WORDS["adjectives_extra"]
    all_verbs = FUNCTION_WORDS["verbs"]
    all_dets = FUNCTION_WORDS["determiners"]
    all_preps = FUNCTION_WORDS["prepositions"]

    sequences = []
    templates = [
        # [det] [adj] [noun] [verb] [det] [noun]
        lambda: [random.choice(all_dets), random.choice(all_adj), random.choice(all_nouns),
                 random.choice(all_verbs), random.choice(all_dets), random.choice(all_nouns)],
        # [det] [noun] [verb] [prep] [det] [noun]
        lambda: [random.choice(all_dets), random.choice(all_nouns),
                 random.choice(all_verbs), random.choice(all_preps),
                 random.choice(all_dets), random.choice(all_noun := all_nouns)],
        # [adj] [noun] [verb] [adj] [noun]
        lambda: [random.choice(all_adj), random.choice(all_nouns),
                 random.choice(all_verbs),
                 random.choice(all_adj), random.choice(all_nouns)],
        # [det] [noun] [verb] [det] [adj] [noun]
        lambda: [random.choice(all_dets), random.choice(all_nouns),
                 random.choice(all_verbs), random.choice(all_dets),
                 random.choice(all_adj), random.choice(all_nouns)],
    ]

    for _ in range(n):
        seq = random.choice(templates)()
        sequences.append(seq)

    return sequences


# ═══════════════════════════════════════════════════════════════════════════
# Transformer Model
# ═══════════════════════════════════════════════════════════════════════════

class MDLMTransformer(nn.Module):
    """Transformer for masked diffusion language modeling.

    Architecture:
      - Token embeddings + positional embeddings + timestep embedding
      - N transformer encoder layers (self-attention)
      - Output projection to vocabulary

    The model sees a partially masked sequence and predicts the original
    tokens at masked positions. This is exactly BERT's masked language
    modeling, but with a continuous noise schedule.
    """

    def __init__(self, vocab_size, d_model=256, n_heads=4, n_layers=4,
                 max_seq_len=32, dropout=0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model

        # Embeddings
        self.token_emb = nn.Embedding(vocab_size, d_model, padding_idx=PAD)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.time_emb = nn.Sequential(
            nn.Linear(d_model, d_model), nn.SiLU(), nn.Linear(d_model, d_model),
        )
        self.time_mlp = nn.Sequential(
            nn.Linear(d_model, d_model), nn.SiLU(), nn.Linear(d_model, d_model),
        )

        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4,
            dropout=dropout, activation="gelu", batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, n_layers)

        # Output head
        self.ln_f = nn.LayerNorm(d_model)
        self.output = nn.Linear(d_model, vocab_size)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def _timestep_embedding(self, t, max_period=10000):
        """Sinusoidal timestep embedding."""
        half = self.d_model // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(half, device=t.device) / half
        )
        args = t[:, None] * freqs[None, :]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        return self.time_mlp(emb)

    def forward(self, tokens, t):
        """
        Args:
            tokens: [batch, seq_len] — partially masked token IDs
            t: [batch] — diffusion timestep (0 = clean, 1 = fully masked)

        Returns:
            logits: [batch, seq_len, vocab_size]
        """
        batch, seq_len = tokens.shape

        # Position indices
        pos = torch.arange(seq_len, device=tokens.device).unsqueeze(0).expand(batch, -1)

        # Embed
        x = self.token_emb(tokens) + self.pos_emb(pos)

        # Timestep conditioning (add to every position)
        t_emb = self._timestep_embedding(t.float())  # [batch, d_model]
        x = x + t_emb.unsqueeze(1)  # broadcast over seq_len

        # Transformer
        # Create padding mask
        pad_mask = (tokens == PAD)
        x = self.transformer(x, src_key_padding_mask=pad_mask)

        # Output
        x = self.ln_f(x)
        logits = self.output(x)
        return logits


# ═══════════════════════════════════════════════════════════════════════════
# Diffusion Process (Forward & Reverse)
# ═══════════════════════════════════════════════════════════════════════════

def forward_mask(tokens, t, mask_id=MASK):
    """Apply forward masking process.

    Each token independently masked with probability t.
    t=0: no masking. t=1: all masked.

    Args:
        tokens: [batch, seq_len] clean tokens
        t: [batch] masking probability per sequence

    Returns:
        masked_tokens: [batch, seq_len]
        mask_positions: [batch, seq_len] boolean (True = was masked)
    """
    batch, seq_len = tokens.shape
    # Per-token mask probability (all tokens in a sequence share the same t)
    prob_mask = t[:, None].expand(batch, seq_len)
    rand = torch.rand_like(tokens.float())
    mask_positions = rand < prob_mask
    masked_tokens = tokens.clone()
    masked_tokens[mask_positions] = mask_id
    return masked_tokens, mask_positions


def mdlm_loss(model, tokens, n_timesteps=100):
    """MDLM training loss.

    Sample t uniformly, mask tokens with prob t, predict original.

    L = E_t [ CE at masked positions ]
    """
    batch = tokens.shape[0]

    # Sample timestep
    t = torch.rand(batch, device=tokens.device)

    # Forward mask
    masked_tokens, mask_positions = forward_mask(tokens, t)

    # Predict
    logits = model(masked_tokens, t)  # [batch, seq, vocab]

    # Cross-entropy only at masked positions
    # Flatten masked positions
    mask_flat = mask_positions.reshape(-1)
    logits_flat = logits.reshape(-1, logits.shape[-1])
    tokens_flat = tokens.reshape(-1)

    if mask_flat.sum() == 0:
        return torch.tensor(0.0, device=tokens.device)

    loss = F.cross_entropy(
        logits_flat[mask_flat],
        tokens_flat[mask_flat],
    )
    return loss


# ═══════════════════════════════════════════════════════════════════════════
# Sampling (Reverse Process)
# ═══════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def sample_mdLM(model, seq_len, tok2id, n_samples=20, n_steps=50, device=DEVICE,
                temperature=0.8):
    """Generate sequences via iterative unmasking with temperature.

    Start from all-MASK. At each step:
      1. Predict logits for all masked positions
      2. Sample tokens from tempered distribution
      3. Keep the most confident predictions (lowest entropy)
      4. Repeat until all positions are filled
    """
    model.eval()
    batch = n_samples

    # Start: all MASK
    tokens = torch.full((batch, seq_len), MASK, device=device)

    # Number of tokens to unmask per step
    total_positions = seq_len
    tokens_per_step = max(1, total_positions // n_steps)

    for step in range(n_steps):
        t = torch.full((batch,), 1.0 - step / n_steps, device=device)
        t = t.clamp(min=0.01, max=0.99)

        logits = model(tokens, t)  # [batch, seq, vocab]

        # For MASK positions: sample from tempered distribution
        mask_positions = (tokens == MASK)

        for b in range(batch):
            masked_idx = mask_positions[b].nonzero(as_tuple=True)[0]
            if len(masked_idx) == 0:
                continue

            # Get logits at masked positions, apply temperature
            pos_logits = logits[b, masked_idx] / temperature  # [n_masked, vocab]

            # Sample from tempered distribution
            probs = F.softmax(pos_logits, dim=-1)
            sampled = torch.multinomial(probs, num_samples=1).squeeze(-1)

            # Confidence = max probability
            confidence = probs.max(dim=-1)[0]

            # Keep top-k most confident
            n_keep = min(tokens_per_step, len(masked_idx))
            top_confident = confidence.topk(n_keep)[1]

            for idx in top_confident:
                pos = masked_idx[idx]
                tokens[b, pos] = sampled[idx]

    return tokens


@torch.no_grad()
def sample_mdLM_confident(model, seq_len, tok2id, n_samples=20, n_steps=100,
                          confidence_threshold=0.3, device=DEVICE):
    """Generate with confidence-based unmasking.

    Only commit predictions where model confidence > threshold.
    This produces more coherent sequences.
    """
    model.eval()
    batch = n_samples
    tokens = torch.full((batch, seq_len), MASK, device=device)
    committed = torch.zeros(batch, seq_len, dtype=torch.bool, device=device)

    for step in range(n_steps):
        n_masked = (tokens == MASK).sum().item()
        if n_masked == 0:
            break

        t_val = max(0.01, 1.0 - step / n_steps)
        t = torch.full((batch,), t_val, device=device)

        logits = model(tokens, t)
        probs = F.softmax(logits, dim=-1)  # [batch, seq, vocab]
        confidence, predicted = probs.max(dim=-1)  # [batch, seq]

        # Unmask positions where confidence is high enough
        mask_positions = (tokens == MASK)
        high_conf = mask_positions & (confidence > confidence_threshold)

        # Also unmask lowest-confidence positions near the end (forced commit)
        progress = step / n_steps
        if progress > 0.7:
            # Force-commit remaining masked positions gradually
            n_remaining = mask_positions.sum().item()
            if n_remaining > 0:
                # Commit positions in order of confidence
                n_force = max(1, n_remaining // (n_steps - step + 1))
                for b in range(batch):
                    masked_idx = mask_positions[b].nonzero(as_tuple=True)[0]
                    if len(masked_idx) == 0:
                        continue
                    conf_at_mask = confidence[b, masked_idx]
                    _, sorted_idx = conf_at_mask.sort(descending=True)
                    for i, idx in enumerate(sorted_idx[:n_force]):
                        pos = masked_idx[idx]
                        tokens[b, pos] = predicted[b, pos]
                        mask_positions[b, pos] = False
        else:
            tokens[high_conf] = predicted[high_conf]

    # Fill any remaining MASK with most likely token
    remaining = (tokens == MASK)
    if remaining.any():
        t = torch.full((batch,), 0.01, device=device)
        logits = model(tokens, t)
        tokens[remaining] = logits[remaining].argmax(dim=-1)

    return tokens


# ═══════════════════════════════════════════════════════════════════════════
# Training & Evaluation
# ═══════════════════════════════════════════════════════════════════════════

def encode_sequences(sequences, tok2id, max_len=8):
    """Encode word sequences to token IDs with padding."""
    encoded = []
    for seq in sequences:
        ids = [BOS] + [tok2id.get(w, UNK) for w in seq[:max_len-2]] + [EOS]
        while len(ids) < max_len:
            ids.append(PAD)
        encoded.append(ids)
    return torch.tensor(encoded, dtype=torch.long)


def decode_tokens(tokens, id2tok):
    """Decode token IDs to word strings."""
    seqs = []
    for row in tokens:
        words = []
        for tid in row:
            tok = id2tok.get(tid.item(), "?")
            if tok in ("<pad>", "<bos>", "<eos>"):
                continue
            if tok == "<mask>":
                words.append("[M]")
            else:
                words.append(tok)
        seqs.append(" ".join(words))
    return seqs


def run_experiment():
    print("=" * 70)
    print("Masked Diffusion Language Model (MDLM)")
    print("=" * 70)

    # Build vocab and data
    tok2id, id2tok, all_words = build_vocab()
    print(f"Vocabulary: {len(all_words)} tokens")

    sequences = generate_training_sequences(n=2000)
    print(f"Training sequences: {len(sequences)}")

    # Show examples
    print(f"\nSample sequences:")
    for i in range(5):
        print(f"  {i}: {' '.join(sequences[i])}")

    # Encode
    encoded = encode_sequences(sequences, tok2id, max_len=8)
    print(f"\nEncoded shape: {encoded.shape}")

    # Train/test split
    n_train = int(0.9 * len(encoded))
    train_data = encoded[:n_train].to(DEVICE)
    test_data = encoded[n_train:].to(DEVICE)

    # Build model
    model = MDLMTransformer(
        vocab_size=len(all_words),
        d_model=256, n_heads=4, n_layers=4,
        max_seq_len=8, dropout=0.1,
    ).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel: {n_params:,} parameters ({n_params/1e6:.1f}M)")

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)

    # Training
    n_epochs = 3000
    batch_size = 256
    print(f"\nTraining {n_epochs} epochs, batch={batch_size}...")

    losses = []
    t0 = time.time()
    best_test_loss = float('inf')
    best_state = None

    for epoch in range(n_epochs):
        # Sample batch
        idx = torch.randint(0, n_train, (batch_size,))
        batch = train_data[idx]

        optimizer.zero_grad()
        loss = mdlm_loss(model, batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if epoch % 200 == 0 or epoch == n_epochs - 1:
            model.eval()
            with torch.no_grad():
                test_loss = mdlm_loss(model, test_data)
            model.train()
            elapsed = time.time() - t0
            losses.append({"epoch": epoch, "train_loss": loss.item(),
                          "test_loss": test_loss.item()})
            print(f"  epoch {epoch:4d}: train={loss.item():.4f}  "
                  f"test={test_loss.item():.4f}  ({elapsed:.0f}s)")

            # Track best model by test loss
            if test_loss.item() < best_test_loss:
                best_test_loss = test_loss.item()
                best_state = {k: v.clone() for k, v in model.state_dict().items()}

    print(f"\nTraining complete: {time.time()-t0:.0f}s")

    # Load best model (before overfitting)
    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"Loaded best model (test_loss={best_test_loss:.4f})")
    model.eval()

    # --- Generate sequences ---
    print(f"\n{'='*70}")
    print("GENERATING SEQUENCES (iterative unmasking)")
    print(f"{'='*70}")

    # Method 1: Standard sampling
    print(f"\n--- Method 1: Uniform unmasking ---")
    samples = sample_mdLM(model, seq_len=8, tok2id=tok2id, n_samples=20, n_steps=30)
    decoded = decode_tokens(samples, id2tok)
    for i, seq in enumerate(decoded):
        print(f"  {i:2d}: {seq}")

    # Method 2: Confidence-based
    print(f"\n--- Method 2: Confidence-based unmasking ---")
    samples2 = sample_mdLM_confident(model, seq_len=8, tok2id=tok2id,
                                      n_samples=20, n_steps=50,
                                      confidence_threshold=0.15)
    decoded2 = decode_tokens(samples2, id2tok)
    for i, seq in enumerate(decoded2):
        print(f"  {i:2d}: {seq}")

    # --- Evaluate grammaticality ---
    print(f"\n{'='*70}")
    print("GRAMMATICALITY EVALUATION")
    print(f"{'='*70}")

    def check_grammar(seq_str):
        """Simple heuristic grammar check."""
        words = seq_str.split()
        # Remove [M] tokens
        words = [w for w in words if w != "[M]"]
        if len(words) < 3:
            return False, "too_short"
        # Check: starts with determiner or adjective
        has_det_start = words[0] in FUNCTION_WORDS["determiners"]
        has_adj = any(w in VOCAB_WORDS["colors"] or
                       w in VOCAB_WORDS["emotion"] or
                       w in FUNCTION_WORDS["adjectives_extra"]
                       for w in words)
        has_noun = any(w in VOCAB_WORDS["animals"] or
                       w in VOCAB_WORDS["food"] or
                       w in VOCAB_WORDS["nature"]
                       for w in words)
        has_verb = any(w in FUNCTION_WORDS["verbs"] for w in words)
        score = sum([has_det_start, has_adj, has_noun, has_verb])
        return score >= 3, f"det={has_det_start} adj={has_adj} noun={has_noun} verb={has_verb}"

    n_eval = 100
    all_samples = sample_mdLM(model, seq_len=8, tok2id=tok2id,
                              n_samples=n_eval, n_steps=30, temperature=0.7)
    all_decoded = decode_tokens(all_samples, id2tok)

    grammatical = 0
    has_structure = 0
    unique_seqs = set()

    for seq in all_decoded:
        seq_clean = seq.replace("[M]", "").strip()
        if seq_clean:
            unique_seqs.add(seq_clean)
        ok, details = check_grammar(seq)
        if ok:
            grammatical += 1
            has_structure += 1

    print(f"\n  Total generated: {n_eval}")
    print(f"  Unique sequences: {len(unique_seqs)}")
    print(f"  Has grammatical structure: {has_structure}/{n_eval} ({has_structure/n_eval:.1%})")

    # Show some unique sequences
    print(f"\n  Unique generated sequences:")
    for seq in sorted(unique_seqs)[:30]:
        ok, _ = check_grammar(seq)
        marker = "✓" if ok else " "
        print(f"    {marker} {seq}")

    # Compare with training data
    train_seqs = set()
    for seq in sequences:
        train_seqs.add(" ".join(seq))

    novel = unique_seqs - train_seqs
    print(f"\n  Novel sequences (not in training): {len(novel)}/{len(unique_seqs)}")

    result = {
        "experiment": "mdlm_text_generation",
        "timestamp": datetime.now().isoformat(),
        "vocab_size": len(all_words),
        "n_train_sequences": n_train,
        "model_params": n_params,
        "n_epochs": n_epochs,
        "train_loss": losses[-1]["train_loss"],
        "test_loss": losses[-1]["test_loss"],
        "n_generated": n_eval,
        "n_unique": len(unique_seqs),
        "n_grammatical": has_structure,
        "n_novel": len(novel),
        "sample_sequences": list(unique_seqs)[:50],
        "loss_history": losses,
    }

    out = RESULTS_DIR / "mdlm_results.json"
    with open(out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    run_experiment()
