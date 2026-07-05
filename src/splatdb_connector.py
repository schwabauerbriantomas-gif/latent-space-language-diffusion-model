"""
SplatsDB connector: load real token embeddings from bge-m3, decode latents
back to tokens via nearest-neighbor (mimics SplatsDB HNSW).

This is the bridge between the FF energy model and SplatsDB's latent space.
In a full integration, these calls would go to the SplatsDB Rust binary via
its MCP/REST API. Here we replicate the same embedding space (bge-m3 1024d)
for self-contained experiments.
"""
import json
import os
import sys
from pathlib import Path
from typing import List, Tuple, Optional

import torch
import numpy as np

REPO = Path(__file__).resolve().parent.parent
DATA_DIR = REPO / "data"
DATA_DIR.mkdir(exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "BAAI/bge-m3"  # SplatsDB's configured embedding model


class SplatDBConnector:
    """Replicates SplatsDB's embedding space for standalone experiments.

    SplatsDB config: bge-m3, 1024d, cosine similarity.
    This connector:
      1. Loads bge-m3 (or a cached subset of token embeddings)
      2. Embeds text → 1024d latent (same as SplatsDB ingest)
      3. Decodes latent → nearest token (same as SplatsDB HNSW search)
    """

    def __init__(self, vocab_size: int = 5000, cache_path: Optional[str] = None):
        self.vocab_size = vocab_size
        self.token_embeddings: Optional[torch.Tensor] = None  # [vocab, 1024]
        self.tokens: Optional[List[str]] = None
        self.embedder = None

        if cache_path and os.path.exists(cache_path):
            print(f"Loading cached embeddings from {cache_path}")
            data = torch.load(cache_path, map_location=DEVICE)
            self.token_embeddings = data["embeddings"]
            with open(cache_path.replace(".pt", "_tokens.json")) as f:
                self.tokens = json.load(f)
        else:
            self._build_vocab(vocab_size)

    def _build_vocab(self, vocab_size: int):
        """Build a vocabulary of common English words + embed them.

        For fast iteration without loading the full bge-m3 model.
        """
        cache = DATA_DIR / f"vocab_embeds_{vocab_size}.pt"
        cache_tokens = DATA_DIR / f"vocab_tokens_{vocab_size}.json"

        if cache.exists() and cache_tokens.exists():
            print(f"Loading cached vocab embeddings from {cache}")
            self.token_embeddings = torch.load(cache).to(DEVICE)
            with open(cache_tokens) as f:
                self.tokens = json.load(f)
            return

        print(f"Building {vocab_size}-word vocabulary with bge-m3...")
        try:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer(MODEL_NAME, device=DEVICE)
        except Exception as e:
            print(f"bge-m3 not available ({e}), using fallback synthetic embeddings")
            self._synthetic_vocab(vocab_size)
            return

        # Common English words as vocabulary
        common_words = self._common_english_words(vocab_size)
        embeddings = model.encode(common_words, convert_to_tensor=True,
                                   normalize_embeddings=True)
        self.token_embeddings = embeddings.to(DEVICE)
        self.tokens = common_words

        torch.save(self.token_embeddings.cpu(), cache)
        with open(cache_tokens, "w") as f:
            json.dump(common_words, f)
        print(f"Saved {vocab_size} embeddings to {cache}")

    def _synthetic_vocab(self, vocab_size: int):
        """Fallback: synthetic embeddings for when bge-m3 is unavailable."""
        torch.manual_seed(42)
        # Create structured embeddings (clusters = word categories)
        self.token_embeddings = torch.randn(vocab_size, 1024, device=DEVICE)
        self.token_embeddings = torch.nn.functional.normalize(
            self.token_embeddings, dim=-1
        )
        self.tokens = [f"token_{i}" for i in range(vocab_size)]

    def _common_english_words(self, n: int) -> List[str]:
        """Return n common English words."""
        base = """the be to of and a in that have I it for not on with he as you
        do at this but his by from they we say her she or an will my one all
        would there their what so up out if about who get which go me when make
        can like time no just him know take people into year your good some
        could them see other than then now look only come its over think also
        back after use two how our work first well way even new want because any
        these give day most us man find here thing tell very when great talk need
        water long little hand high big different old few next early young important
        public bad same able house door sea tree light dark sun moon star fire earth
        king queen man woman child family friend love hate happy sad angry calm
        walk run jump swim fly eat drink sleep dream think speak write read learn
        red blue green yellow white black cat dog bird fish horse cow sheep chicken
        bread milk cheese apple orange banana table chair bed book pen paper""".split()
        words = list(dict.fromkeys(base))  # dedup, preserve order
        while len(words) < n:
            words.append(f"word_{len(words)}")
        return words[:n]

    def embed_text(self, text: str) -> torch.Tensor:
        """Embed a text string → [1, 1024] (single vector)."""
        if self.embedder is None:
            from sentence_transformers import SentenceTransformer
            self.embedder = SentenceTransformer(MODEL_NAME, device=DEVICE)
        emb = self.embedder.encode([text], convert_to_tensor=True,
                                    normalize_embeddings=True)
        return emb.to(DEVICE)

    def decode_latent(self, latent: torch.Tensor) -> Tuple[List[str], torch.Tensor]:
        """Decode a latent vector → nearest token (SplatsDB HNSW equivalent).

        latent: [batch, 1024] or [1024]
        Returns: (tokens, distances)
        """
        if latent.dim() == 1:
            latent = latent.unsqueeze(0)

        if self.token_embeddings is None:
            raise RuntimeError("Vocab not loaded")

        # Cosine similarity (SplatsDB uses cosine for bge-m3)
        latent_norm = torch.nn.functional.normalize(latent.cpu(), dim=-1)
        token_norm = torch.nn.functional.normalize(self.token_embeddings.cpu(), dim=-1)
        sims = latent_norm @ token_norm.T  # [batch, vocab]

        best_idx = sims.argmax(dim=-1)  # [batch]
        tokens = [self.tokens[i] for i in best_idx.cpu().tolist()]
        distances = 1.0 - sims.max(dim=-1)[0]  # cosine distance
        return tokens, distances

    def decode_topk(self, latent: torch.Tensor, k: int = 5) -> List[List[Tuple[str, float]]]:
        """Decode latent → top-k candidate tokens with distances (SplatsDB find_neighbors)."""
        if latent.dim() == 1:
            latent = latent.unsqueeze(0)

        latent_norm = torch.nn.functional.normalize(latent, dim=-1)
        token_norm = torch.nn.functional.normalize(self.token_embeddings, dim=-1)
        sims = latent_norm @ token_norm.T  # [batch, vocab]

        topk_sims, topk_idx = sims.topk(k, dim=-1)
        results = []
        for b in range(latent.shape[0]):
            candidates = [
                (self.tokens[idx], 1.0 - sim)
                for idx, sim in zip(topk_idx[b].cpu().tolist(),
                                     topk_sims[b].cpu().tolist())
            ]
            results.append(candidates)
        return results
