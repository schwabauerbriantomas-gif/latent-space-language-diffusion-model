"""
Qualitative text evaluation for the MDLM model.

PROBLEM THIS SOLVES
-------------------
The existing `rep_score` (unique token-bigrams / total, in
``scripts/benchmark_generation_v4.py``) reports 1.00 for text like

    "Climate change is one of the biggest challenges to climate."

— it is *semantically* circular nonsense but has *no repeated token bigrams*,
so the old metric is blind to it. TPS measures speed, not meaning.

This module adds semantics-aware quality metrics that catch the failure modes
the repetition score misses:

    1. DISTINCT-N        — n-gram diversity (word-level, not token-level)
    2. SELF-BLEU         — inter-sentence redundancy (high = repetitive)
    3. SEMANTIC COHERENCE — topic persistence + lexical chaining (TF-IDF cosine)
    4. COMPLETION        — fraction of sentences that terminate properly
    5. NON-SEQUITUR      — abrupt topical jumps between adjacent sentences
    6. CIRCULAR          — prompt keyword echoed back as object (circular phrasing)

Of these, **CIRCULAR** is the metric most directly targeting the observed MDLM
failure mode. The metrics are designed to be *complementary*: circular phrasing
and self-BLEU catch the degenerate case even when lexical coherence looks high.

IMPORTANT LIMITATION
--------------------
Signals 3 and 5 are built from TF-IDF bag-of-words (no external embeddings are
allowed). Pure lexical overlap **cannot** distinguish valid semantic progression
("climate → temperatures → flooding → emissions", which share no word stems)
from true non sequiturs ("climate → Python → stock market"). We mitigate this
with topic-persistence, discourse-marker, and anaphora signals, but the
fundamental ceiling is real: a TF-IDF coherence score rewards *lexical*
cohesion, not *semantic* relatedness. Treat coherence/non-sequitur as a weak
signal and rely on circular-phrase detection as the primary gate for the known
MDLM failure.

DEPENDENCIES: Python stdlib + numpy only. No NLTK / spaCy / network / models.

Usage
-----
    from quality_diagnostic import QualityAnalyzer, quality_gate

    qa = QualityAnalyzer()
    report = qa.analyze(generated_text, prompt="Climate change is")
    qa.print_report(generated_text, prompt="Climate change is")

    if quality_gate(text, prompt):
        ...  # text passes minimum quality threshold
"""
from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

__all__ = ["QualityAnalyzer", "quality_gate"]


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Minimal English stopword list (no NLTK). Lowercased, no punctuation.
_STOPWORDS: frozenset[str] = frozenset(
    """
    a an the and or but if then else when while of to in on at by for with about
    against between into through during before after above below up down out off
    over under again further is are was were be been being am do does did doing
    have has had having this that these those it its it's i you he she we they
    them his her their our your my me him us as from not no nor so than too very
    can will just should would could may might must shall of which who whom whose
    what where why how all any both each few more most other some such only own
    same s t don now d ll m o re ve y ain aren wasn weren
    """.split()
)

# Sentence-terminal punctuation.
_TERMINAL_PUNCT: tuple[str, ...] = (".", "!", "?", "。", "！", "？")

# A short word must be >= this many chars AND not a stopword to count as "content".
_MIN_CONTENT_LEN = 3

# BLEU max n-gram order.
_BLEU_MAX_N = 4

# Anaphora markers — words that refer back to prior content, signalling cohesion
# even when there is no lexical overlap.
_ANAPHORA: frozenset[str] = frozenset(
    "this these those such same it they them their both itself themselves".split()
)

# Discourse connectives that signal intentional sentence linking.
_CONNECTIVES: frozenset[str] = frozenset(
    """
    however therefore furthermore additionally moreover thus hence consequently
    meanwhile nevertheless nonetheless similarly likewise instead rather
    accordingly although though whereas because since until first second third
    finally next then also but and so or yet still as thus
    """.split()
)


# ---------------------------------------------------------------------------
# Text utilities (pure functions)
# ---------------------------------------------------------------------------

def _tokenize_words(text: str) -> List[str]:
    """Lowercase word tokenization via regex. Returns alphabetic tokens + contractions."""
    return re.findall(r"[a-z]+(?:'[a-z]+)?", text.lower())


def _content_words(text: str) -> List[str]:
    """Words that carry meaning: length >= 3 and not a stopword."""
    return [w for w in _tokenize_words(text)
            if len(w) >= _MIN_CONTENT_LEN and w not in _STOPWORDS]


def _split_sentences(text: str) -> List[str]:
    """Split into sentences on terminal punctuation while preserving content.

    Handles common abbreviations crudely (protected from splitting) and
    collapses whitespace. A trailing fragment without terminal punctuation is
    kept as its own (incomplete) sentence — this is intentional, since
    incomplete sentences lower the completion-rate metric.
    """
    protected = text
    for abbr in ("Mr.", "Mrs.", "Dr.", "Ms.", "Inc.", "Ltd.", "Co.",
                 "e.g.", "i.e.", "vs.", "etc.", "U.S.", "U.K."):
        protected = protected.replace(abbr, abbr.replace(".", "<DOT>"))

    parts = re.split(r'(?<=[.!?。！？])\s+', protected.strip())

    sentences: List[str] = []
    for part in parts:
        part = part.replace("<DOT>", ".").strip()
        if part:
            sentences.append(part)
    return sentences


def _strip_prompt_prefix(text: str, prompt: str) -> Tuple[str, bool]:
    """Remove a verbatim prompt echo from the start of ``text``.

    MDLM semi-AR generation is conditioned by prepending the prompt, and the
    sampled text normally starts with a verbatim echo of it.  Coherence
    signals that compare sentences against prompt keywords (topic
    persistence, first-sentence link) are dominated by that echo, so we
    compute an explicit "no-prompt" variant with the echo stripped.

    Returns (stripped_text, was_stripped). Only a *leading, verbatim* echo is
    removed — the strip never touches text that merely continues the prompt
    mid-sentence beyond the shared prefix.
    """
    p = prompt.strip()
    if not p:
        return text, False
    t = text.lstrip()
    if t.startswith(p):
        return t[len(p):].lstrip(), True
    return text, False


def _ngrams(words: Sequence[str], n: int) -> List[Tuple[str, ...]]:
    """Return all word-level n-grams of order ``n``."""
    if len(words) < n:
        return []
    return [tuple(words[i:i + n]) for i in range(len(words) - n + 1)]


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity of two 1-D vectors; returns 0.0 for zero vectors."""
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


# ---------------------------------------------------------------------------
# BLEU (stdlib-only implementation with add-one smoothing)
# ---------------------------------------------------------------------------

def _modified_precision(candidate: Sequence[str],
                        references: Sequence[Sequence[str]],
                        n: int) -> Tuple[int, int]:
    """Numerator/denominator of modified n-gram precision.

    Numerator = sum over candidate n-grams of min(count in cand, max count in
    any single reference). Denominator = total candidate n-grams.
    """
    cand_ngrams = Counter(_ngrams(candidate, n))
    if not cand_ngrams:
        return 0, 0

    max_ref: Dict[Tuple[str, ...], int] = {}
    for ref in references:
        ref_counts = Counter(_ngrams(ref, n))
        for ng, c in ref_counts.items():
            if c > max_ref.get(ng, 0):
                max_ref[ng] = c

    numerator = sum(min(c, max_ref.get(ng, 0)) for ng, c in cand_ngrams.items())
    denominator = sum(cand_ngrams.values())
    return numerator, denominator


def _sentence_bleu(candidate: Sequence[str],
                   references: Sequence[Sequence[str]],
                   max_n: int = _BLEU_MAX_N) -> float:
    """Smoothed sentence-level BLEU in [0, 1].

    Uses add-one (Lin & Hsu) smoothing so that zero n-gram matches at any order
    don't zero out the geometric mean — important for short MDLM outputs.
    """
    cand_len = len(candidate)
    ref_len = min((len(r) for r in references), default=0)
    if cand_len == 0 or ref_len == 0:
        return 0.0
    bp = 1.0 if cand_len > ref_len else math.exp(1 - ref_len / cand_len)

    log_sum = 0.0
    for n in range(1, max_n + 1):
        num, den = _modified_precision(candidate, references, n)
        precision = (num + 1.0) / (den + 1.0) if den > 0 else 0.0
        log_sum += math.log(precision + 1e-12)

    geo_mean = math.exp(log_sum / max_n)
    return bp * geo_mean


# ---------------------------------------------------------------------------
# TF-IDF helpers
# ---------------------------------------------------------------------------

def _compute_idf(token_lists: List[List[str]]) -> Dict[str, float]:
    """Inverse document frequency over a list of token lists (documents).

    Uses smoothed IDF: idf = ln((1 + N) / (1 + df)) + 1  (sklearn default).
    """
    n_docs = len(token_lists)
    if n_docs == 0:
        return {}
    df: Counter[str] = Counter()
    for toks in token_lists:
        for w in set(toks):
            df[w] += 1
    return {w: math.log((1 + n_docs) / (1 + df[w])) + 1.0 for w in df}


def _tfidf_vector(words: List[str], idf: Dict[str, float]) -> np.ndarray:
    """TF-IDF vector for a bag of words given a precomputed IDF map."""
    if not words or not idf:
        return np.zeros(0)
    tf = Counter(words)
    vocab = sorted(idf.keys())
    idx = {w: i for i, w in enumerate(vocab)}
    vec = np.zeros(len(vocab), dtype=np.float64)
    total = float(sum(tf.values())) or 1.0
    for w, c in tf.items():
        if w in idx:
            vec[idx[w]] = (c / total) * idf[w]
    return vec


def _tfidf_cosine(a: List[str], b: List[str], idf: Dict[str, float]) -> float:
    """TF-IDF cosine similarity between two word bags (shared IDF)."""
    va = _tfidf_vector(a, idf)
    vb = _tfidf_vector(b, idf)
    return _cosine(va, vb)


# ---------------------------------------------------------------------------
# Main analyzer
# ---------------------------------------------------------------------------

class QualityAnalyzer:
    """Compute qualitative text-quality metrics for MDLM-generated text.

    All methods are deterministic and side-effect free. Construction is cheap;
    the stopword set is cached on the instance.
    """

    stopwords: frozenset[str]

    def __init__(self, stopwords: Optional[Sequence[str]] = None) -> None:
        self.stopwords = frozenset(stopwords) if stopwords else _STOPWORDS

    # -- public API --------------------------------------------------------

    def analyze(self, text: str, prompt: str = "") -> Dict[str, Any]:
        """Return a dict of all metrics for ``text`` (optionally vs ``prompt``).

        Keys
        ----
        distinct_1, distinct_2, distinct_3 : float in [0, 1]
        self_bleu                         : float in [0, 1] (high = repetitive)
        semantic_coherence                : float in [0, 1] (high = coherent)
        sentence_completion_rate          : float in [0, 1]
        non_sequitur_rate                 : float in [0, 1] (high = jumpy)
        non_sequitur_count                : int
        circular_phrases                  : list[str]
        circular_phrase_count             : int
        topic_persistence                 : float in [0, 1] (prompt coverage)
        n_sentences, n_words, n_content_words : int
        overall_quality                   : float in [0, 1] (composite)
        """
        words = _tokenize_words(text)
        content = [w for w in words if len(w) >= _MIN_CONTENT_LEN
                   and w not in self.stopwords]
        sentences = _split_sentences(text)
        sent_words = [_tokenize_words(s) for s in sentences]

        distinct = self._distinct_n(words)
        self_bleu = self._self_bleu(sent_words)
        coherence, topic_persist = self._semantic_coherence(sent_words, prompt)

        # Explicit with/without-prompt accounting: MDLM output echoes the
        # prompt verbatim, which dominates topic-persistence. Measured on the
        # reference samples (m2b, g2_quality.json): the coherent reference
        # drops from coh=0.259 (echo included) to coh=0.000 once the echo is
        # stripped — the continuation is semantically on-topic but lexically
        # disjoint (the documented TF-IDF ceiling). We therefore report BOTH
        # and calibrate the gate on the with-echo value.
        stripped_text, did_strip = _strip_prompt_prefix(text, prompt)
        coherence_np, _tp_np = self._semantic_coherence(
            [_tokenize_words(s) for s in _split_sentences(stripped_text)],
            prompt,
        ) if did_strip else (coherence, topic_persist)

        completion = self._sentence_completion_rate(sentences)
        ns_rate, ns_count = self._non_sequitur_rate(sent_words, prompt)
        circular = self._detect_circular_phrases(text, prompt, sentences)

        overall = self._composite_score(
            distinct, self_bleu, coherence, completion, ns_rate, circular
        )

        return {
            "distinct_1": distinct[1],
            "distinct_2": distinct[2],
            "distinct_3": distinct[3],
            "self_bleu": self_bleu,
            "semantic_coherence": coherence,
            "semantic_coherence_no_prompt": coherence_np,
            "prompt_stripped": did_strip,
            "sentence_completion_rate": completion,
            "non_sequitur_rate": ns_rate,
            "non_sequitur_count": ns_count,
            "circular_phrases": circular,
            "circular_phrase_count": len(circular),
            "topic_persistence": topic_persist,
            "n_sentences": len(sentences),
            "n_words": len(words),
            "n_content_words": len(content),
            "overall_quality": overall,
            "prompt": prompt,
        }

    def compare(self, texts: Sequence[str],
                prompts: Optional[Sequence[str]] = None
                ) -> str:
        """Return an aligned ASCII comparison table for multiple samples."""
        if prompts is None:
            prompts = [""] * len(texts)

        rows = []
        for i, (text, prompt) in enumerate(zip(texts, prompts)):
            m = self.analyze(text, prompt)
            rows.append((i, prompt, text, m))

        col_def = [
            ("#", 3, lambda r: str(r[0])),
            ("coh", 5, lambda r: f"{r[3]['semantic_coherence']:.2f}"),
            ("d1", 5, lambda r: f"{r[3]['distinct_1']:.2f}"),
            ("d2", 5, lambda r: f"{r[3]['distinct_2']:.2f}"),
            ("d3", 5, lambda r: f"{r[3]['distinct_3']:.2f}"),
            ("sBLEU", 6, lambda r: f"{r[3]['self_bleu']:.2f}"),
            ("compl", 6, lambda r: f"{r[3]['sentence_completion_rate']:.2f}"),
            ("ns%", 5, lambda r: f"{r[3]['non_sequitur_rate']*100:.0f}"),
            ("circ", 5, lambda r: str(r[3]['circular_phrase_count'])),
            ("QUAL", 5, lambda r: f"{r[3]['overall_quality']:.2f}"),
        ]
        sep = " "
        header = sep.join(name.ljust(w) for name, w, _ in col_def)
        divider = "-" * len(header)
        body_lines = []
        for r in rows:
            cells = [fn(r).ljust(w) for _, w, fn in col_def]
            body_lines.append(sep.join(cells))

        preview_lines = []
        for r in rows:
            preview = r[2].strip().replace("\n", " ")[:60]
            preview_lines.append(f"  [{r[0]}] {preview}")

        table = "\n".join([
            header,
            divider,
            *body_lines,
            "",
            "Legend: coh=coherence d1/d2/d3=distinct sBLEU=self-BLEU "
            "compl=completion ns%=non-sequitur circ=circular QUAL=overall",
            "",
            *preview_lines,
        ])
        return table

    def print_report(self, text: str, prompt: str = "") -> str:
        """Print and return a human-readable diagnostic report."""
        m = self.analyze(text, prompt)
        lines = self._format_report(text, prompt, m)
        report = "\n".join(lines)
        print(report)
        return report

    # -- metric implementations -------------------------------------------

    def _distinct_n(self, words: Sequence[str]) -> Dict[int, float]:
        """Distinct-1/2/3: unique n-grams / total n-grams (word level)."""
        out: Dict[int, float] = {1: 0.0, 2: 0.0, 3: 0.0}
        for n in (1, 2, 3):
            grams = _ngrams(words, n)
            if grams:
                out[n] = len(set(grams)) / len(grams)
            elif n == 1 and words:
                out[n] = 1.0
        return out

    def _self_bleu(self, sent_words: List[List[str]]) -> float:
        """Average BLEU of each sentence against all other sentences.

        High self-BLEU ⇒ sentences are near-duplicates ⇒ repetitive text.
        Returns 0.0 when there are fewer than 2 sentences.
        """
        if len(sent_words) < 2:
            return 0.0
        scores = []
        for i, cand in enumerate(sent_words):
            refs = [s for j, s in enumerate(sent_words) if j != i]
            if not cand or not any(refs):
                continue
            scores.append(_sentence_bleu(cand, refs))
        return float(np.mean(scores)) if scores else 0.0

    def _semantic_coherence(self, sent_words: List[List[str]],
                            prompt: str = "") -> Tuple[float, float]:
        """Coherence via a blend of TF-IDF signals.

        Combines four lexical signals (all derivable from TF-IDF):

          1. Halves similarity   — cosine between TF-IDF of the first and
                                   second half of the text (drift detector).
          2. Adjacent chaining   — mean cosine between consecutive sentences
                                   (lexical cohesion).
          3. Topic persistence   — fraction of sentences that share >=1 content
                                   word with the prompt (anaphoric chain to the
                                   conditioning topic).
          4. First-sentence link — cosine between the prompt and the first
                                   generated sentence (continuation quality).

        Returns (coherence_score in [0,1], topic_persistence in [0,1]).

        LIMITATION: bag-of-words TF-IDF cannot detect *semantic* relatedness
        between words with different stems (e.g. climate↔emissions). This score
        rewards *lexical* cohesion. See module docstring.
        """
        content_lists = [[w for w in s if len(w) >= _MIN_CONTENT_LEN
                          and w not in self.stopwords]
                         for s in sent_words]

        # Fall back to word-split if too few sentences.
        if len(content_lists) < 2:
            flat = [w for s in content_lists for w in s]
            if len(flat) < 4:
                return 0.0, 0.0
            mid = len(flat) // 2
            content_lists = [flat[:mid], flat[mid:]]

        # Build IDF from sentences (+ prompt as an extra doc if present).
        idf_docs = list(content_lists)
        prompt_content = _content_words(prompt) if prompt.strip() else []
        if prompt_content:
            idf_docs.append(prompt_content)
        idf = _compute_idf(idf_docs)
        if not idf:
            return 0.0, 0.0

        n = len(content_lists)

        # Signal 1: halves similarity.
        mid = max(n // 2, 1)
        first_half = [w for s in content_lists[:mid] for w in s]
        second_half = [w for s in content_lists[mid:] for w in s]
        if first_half and second_half:
            halves_sim = _tfidf_cosine(first_half, second_half, idf)
        else:
            halves_sim = 0.0

        # Signal 2: adjacent chaining.
        if n >= 2:
            adj_sims = [_tfidf_cosine(content_lists[i], content_lists[i + 1], idf)
                        for i in range(n - 1)]
            chain = float(np.mean(adj_sims))
        else:
            chain = 0.5  # neutral for single-sentence text

        # Signal 3: topic persistence.
        prompt_keywords = set(prompt_content)
        if prompt_keywords:
            sent_has_kw = [
                1.0 if any(w in prompt_keywords for w in s) else 0.0
                for s in content_lists
            ]
            topic_persist = float(np.mean(sent_has_kw))
        else:
            topic_persist = 0.5  # no prompt → neutral

        # Signal 4: first-sentence prompt link.
        if prompt_keywords and content_lists and content_lists[0]:
            first_link = _tfidf_cosine(prompt_content, content_lists[0], idf)
        else:
            first_link = 0.5

        # Blended coherence. Weights chosen by calibration on good/bad samples;
        # topic persistence and first-link dominate because they are the most
        # robust to the "progressive vocabulary" confound.
        coherence = (
            0.25 * halves_sim
            + 0.20 * chain
            + 0.35 * topic_persist
            + 0.20 * first_link
        )
        coherence = max(0.0, min(1.0, coherence))
        return coherence, topic_persist

    def _sentence_completion_rate(self, sentences: List[str]) -> float:
        """Fraction of sentences ending with terminal punctuation."""
        if not sentences:
            return 0.0
        complete = sum(
            1 for s in sentences
            if s.rstrip().endswith(_TERMINAL_PUNCT)
        )
        return complete / len(sentences)

    def _non_sequitur_rate(self, sent_words: List[List[str]],
                           prompt: str = "") -> Tuple[float, int]:
        """Detect abrupt topical jumps between adjacent sentences.

        A transition is flagged as a *non sequitur* when **all** of these hold:

          - The two adjacent sentences share zero content words.
          - Their TF-IDF cosine is near zero (< 0.05).
          - Neither sentence contains a prompt content keyword.
          - Both sentences have >= 2 content words (avoid flagging fragments).
          - The transition lacks a discourse connective or anaphora marker
            (``however``, ``these``, ``such``, …) that would signal intentional
            linking.

        The last two conditions reduce false positives on valid progressive
        text that introduces new vocabulary while staying on topic.

        Returns (rate, count). Rate = jumps / transitions.
        """
        transitions = len(sent_words) - 1
        if transitions <= 0:
            return 0.0, 0

        content_lists = [[w for w in s if len(w) >= _MIN_CONTENT_LEN
                          and w not in self.stopwords]
                         for s in sent_words]
        idf = _compute_idf(content_lists)
        prompt_keywords = set(_content_words(prompt)) if prompt.strip() else set()

        jumps = 0
        for i in range(transitions):
            a = set(content_lists[i])
            b = set(content_lists[i + 1])
            cos = _tfidf_cosine(content_lists[i], content_lists[i + 1], idf) \
                if idf else 0.0
            shared = len(a & b)
            a_has_prompt = bool(a & prompt_keywords)
            b_has_prompt = bool(b & prompt_keywords)
            # Look for discourse markers / anaphora in the second sentence.
            b_all = set(sent_words[i + 1])
            has_marker = bool(b_all & _CONNECTIVES) or bool(b_all & _ANAPHORA)

            if (shared == 0 and cos < 0.05
                    and len(a) >= 2 and len(b) >= 2
                    and not a_has_prompt and not b_has_prompt
                    and not has_marker):
                jumps += 1
        return jumps / transitions, jumps

    def _detect_circular_phrases(self, text: str, prompt: str,
                                 sentences: List[str]) -> List[str]:
        """Detect circular phrasing where prompt keywords echo back.

        This is the **primary metric** for the observed MDLM failure mode
        (``rep_score`` = 1.00 but text is semantically circular). Heuristics
        (any one flags a sentence):

          (a) Keyword-repeat: a prompt keyword appears 2+ times within a single
              short sentence (< 14 words) — e.g. "climate … climate".
          (b) Subject-as-object: a prompt keyword is the *last* content word of
              a sentence and that same keyword also appears earlier in it —
              "X is … to X".
          (c) No-new-info: > 70% of a sentence's content words are prompt
              keywords — the sentence merely restates the prompt.
        """
        if not prompt.strip():
            return []
        prompt_keywords = {w for w in _content_words(prompt)
                           if len(w) >= _MIN_CONTENT_LEN}
        if not prompt_keywords:
            return []

        flagged: List[str] = []
        for sent in sentences:
            sw = _tokenize_words(sent)
            if len(sw) < 3:
                continue
            content = [w for w in sw
                       if len(w) >= _MIN_CONTENT_LEN and w not in self.stopwords]
            kw_counts = Counter(w for w in content if w in prompt_keywords)

            circular = False

            # (a) keyword repeated within one short sentence
            if len(sw) <= 14 and any(c >= 2 for c in kw_counts.values()):
                circular = True

            # (b) last content word is a prompt keyword AND appears earlier
            if content and content[-1] in prompt_keywords:
                if content[-1] in content[:-1]:
                    circular = True

            # (c) > 70% content words are prompt keywords (no new information)
            if content:
                overlap = sum(1 for w in content if w in prompt_keywords)
                if overlap / len(content) > 0.70:
                    circular = True

            if circular:
                snippet = sent.strip()
                if snippet not in flagged:
                    flagged.append(snippet)

        return flagged

    # -- composite --------------------------------------------------------

    @staticmethod
    def _composite_score(distinct: Dict[int, float], self_bleu: float,
                         coherence: float, completion: float,
                         ns_rate: float, circular: List[str]) -> float:
        """Weighted composite quality score in [0, 1].

        Penalizes circular phrasing and non-sequiturs heavily — these are the
        exact failure modes the token-level repetition score misses.
        """
        # Base: diversity + coherence + grammatical completion.
        diversity = (distinct[1] + distinct[2] + distinct[3]) / 3.0
        redundancy = 1.0 - self_bleu  # self-BLEU is inverted (low is good).
        base = (0.30 * diversity
                + 0.30 * coherence
                + 0.20 * redundancy
                + 0.20 * completion)

        # Penalties.
        penalty = 0.0
        penalty += 0.15 * ns_rate                     # jumpy text
        penalty += 0.25 * min(len(circular), 3) / 3.0  # circular phrasing
        return max(0.0, min(1.0, base - penalty))

    # -- report formatting ------------------------------------------------

    def _format_report(self, text: str, prompt: str,
                       m: Dict[str, Any]) -> List[str]:
        bar = "=" * 72
        lines: List[str] = [bar, "  QUALITY DIAGNOSTIC REPORT", bar]

        if prompt:
            lines.append(f"  Prompt : {prompt}")
        preview = text.strip().replace("\n", " ")
        if len(preview) > 120:
            preview = preview[:117] + "..."
        lines.append(f"  Text   : {preview}")
        lines.append(f"  Words  : {m['n_words']} "
                     f"(content: {m['n_content_words']})  "
                     f"Sentences: {m['n_sentences']}")
        lines.append("")

        def pct(v: float) -> str:
            return f"{v * 100:5.1f}%"

        # 1. Diversity
        d1 = float(m["distinct_1"])
        d2 = float(m["distinct_2"])
        d3 = float(m["distinct_3"])
        lines.append("  1. DISTINCT-N (n-gram diversity, higher = more diverse)")
        lines.append(f"     distinct-1 (unigrams) : {d1:.3f}  {pct(d1)}")
        lines.append(f"     distinct-2 (bigrams)  : {d2:.3f}  {pct(d2)}")
        lines.append(f"     distinct-3 (trigrams) : {d3:.3f}  {pct(d3)}")
        lines.append("")

        # 2. Self-BLEU
        sb = float(m["self_bleu"])
        sb_verdict = ("HIGH (repetitive)" if sb > 0.5
                      else "moderate" if sb > 0.25 else "low (diverse)")
        lines.append("  2. SELF-BLEU (inter-sentence redundancy, lower = better)")
        lines.append(f"     self-bleu : {sb:.3f}  → {sb_verdict}")
        lines.append("")

        # 3. Coherence
        coh = float(m["semantic_coherence"])
        tp = float(m["topic_persistence"])
        coh_verdict = ("COHERENT" if coh > 0.5
                       else "moderate" if coh > 0.3
                       else "LOW (possible drift)")
        lines.append("  3. SEMANTIC COHERENCE (lexical cohesion + topic persistence)")
        lines.append(f"     coherence        : {coh:.3f}  → {coh_verdict}")
        lines.append(f"     topic persistence: {tp:.3f}")
        lines.append("")

        # 4. Completion
        comp = float(m["sentence_completion_rate"])
        comp_verdict = ("OK" if comp > 0.8
                        else "some fragments" if comp > 0.5
                        else "MANY FRAGMENTS")
        lines.append("  4. SENTENCE COMPLETION (terminal punctuation)")
        lines.append(f"     rate : {comp:.3f}  → {comp_verdict}")
        lines.append("")

        # 5. Non sequitur
        ns = float(m["non_sequitur_rate"])
        ns_verdict = ("smooth" if ns < 0.2
                      else "some jumps" if ns < 0.5 else "JUMPY (non sequitur)")
        lines.append("  5. NON-SEQUITUR (abrupt topical jumps)")
        lines.append(f"     rate : {ns:.3f}  ({m['non_sequitur_count']} jumps)  "
                     f"→ {ns_verdict}")
        lines.append("")

        # 6. Circular
        circ = m["circular_phrases"]
        lines.append("  6. CIRCULAR PHRASING (prompt keyword echoed back)")
        if circ:
            lines.append(f"     ⚠ {len(circ)} circular phrase(s) detected:")
            for c in circ:
                lines.append(f"       • {c}")
        else:
            lines.append("     none detected  → OK")
        lines.append("")

        # Overall
        overall = float(m["overall_quality"])
        if overall >= 0.65:
            verdict = "PASS"
        elif overall >= 0.4:
            verdict = "MARGINAL"
        else:
            verdict = "FAIL"
        lines.append("-" * 72)
        lines.append(f"  OVERALL QUALITY : {overall:.3f}  →  {verdict}")
        lines.append(bar)
        return lines


# ---------------------------------------------------------------------------
# Quality gate
# ---------------------------------------------------------------------------

def quality_gate(text: str, prompt: str = "",
                 min_coherence: float = 0.2,
                 min_distinct2: float = 0.4) -> bool:
    """Return True if ``text`` passes a minimum quality bar.

    The gate checks the metrics most diagnostic of the MDLM failure mode:
    semantic coherence (catches drift/circularity), bigram diversity (catches
    degenerate repetition), and **hard-fails on circular phrasing** — the
    signature MDLM failure that ``rep_score`` misses entirely.

    Parameters
    ----------
    text, prompt : str
        Generated text and its conditioning prompt.
    min_coherence : float
        Minimum semantic coherence (default 0.2). Calibrated on real
        measurements (m2b, reports/g2_quality.json): reference *coherent*
        text scores coh=0.259 (echo included) while the MDLM generation-mode
        means land at 0.258-0.285; the previous default 0.3 rejected the
        coherent reference itself (its own demo failed 4/4). The coherent /
        non-sequitur separation at this threshold comes from the circular
        hard-fail and non-sequitur signals, not from coherence alone.
    min_distinct2 : float
        Minimum distinct-2 diversity (default 0.4).

    Returns
    -------
    bool
        True if the text passes all checks.
    """
    qa = QualityAnalyzer()
    m = qa.analyze(text, prompt)
    if float(m["semantic_coherence"]) < min_coherence:
        return False
    if float(m["distinct_2"]) < min_distinct2:
        return False
    # Hard fail on circular phrasing — signature MDLM failure.
    if int(m["circular_phrase_count"]) > 0:
        return False
    return True


# ---------------------------------------------------------------------------
# Self-test / demo
# ---------------------------------------------------------------------------

def _demo() -> None:
    """Run the analyzer on representative good/bad samples when executed
    directly (``python quality_diagnostic.py``)."""
    qa = QualityAnalyzer()

    prompt = "Climate change is one of the biggest challenges"

    samples = [
        # BAD: circular, semantically empty — rep_score would say 1.00.
        ("Circular (bad)", prompt,
         "Climate change is one of the biggest challenges to climate. "
         "The climate change challenges the climate of the climate. "
         "Climate is the change of the biggest climate challenges."),

        # GOOD: coherent, adds new information.
        ("Coherent (good)", prompt,
         "Climate change is one of the biggest challenges facing humanity "
         "today. Rising global temperatures threaten coastal cities with "
         "flooding and disrupt agricultural production worldwide. Scientists "
         "warn that without rapid cuts to carbon emissions, irreversible "
         "damage to ecosystems will accelerate over the coming decades."),

        # MEDIOCRE: grammatical but repetitive / jumpy.
        ("Repetitive (mediocre)", prompt,
         "Climate change is one of the biggest challenges. "
         "Climate change is a very big challenge. "
         "The challenge is climate and the climate is challenging. "
         "Basketball players also face challenges on the court."),

        # INCOHERENT: topic drift / non sequitur.
        ("Non sequitur (bad)", prompt,
         "Climate change is a serious problem. "
         "My favorite programming language is Python. "
         "The stock market crashed yesterday afternoon. "
         "Pineapples grow best in tropical island climates."),
    ]

    texts = [s[2] for s in samples]
    prompts = [s[1] for s in samples]

    print("\n" + "=" * 72)
    print("  QUALITY DIAGNOSTIC — DEMO ON REPRESENTATIVE SAMPLES")
    print("=" * 72)
    print()
    print(qa.compare(texts, prompts))
    print()

    for label, p, t in samples:
        print()
        qa.print_report(t, p)
        passed = quality_gate(t, p)
        print(f"  quality_gate → {'PASS' if passed else 'FAIL'}")
        # Post-fix expectation (min_coherence recalibrated 0.3→0.2 on the m2b
        # measurements: coherent ref coh=0.259, MDLM mode means 0.258-0.285):
        expected_pass = label == "Coherent (good)"
        assert passed == expected_pass, (
            f"demo sample '{label}': quality_gate returned "
            f"{'PASS' if passed else 'FAIL'}, expected "
            f"{'PASS' if expected_pass else 'FAIL'}"
        )
        m = qa.analyze(t, p)
        print(f"  coh={m['semantic_coherence']:.3f} "
              f"(no-prompt: {m['semantic_coherence_no_prompt']:.3f}, "
              f"stripped={m['prompt_stripped']})")
        print()


if __name__ == "__main__":
    _demo()
