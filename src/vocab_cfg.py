"""
Expanded Vocabulary & Context-Free Grammar for MDLM.

VOCABULARY: ~800+ content words across 15 categories + 200+ function words
GRAMMAR:    Context-Free Grammar with recursive rules for diverse syntax

CFG generates these structures:
  S  → NP VP | NP VP Conj S | NP VP Adv
  NP → Det? AdjP? N (PP | RelClause)*
  VP → V (NP | PP | Adv | S)*
  PP → Prep NP
  AdjP → (Adj)+
  RelClause → "that" VP | "who" VP

Examples generated:
  "the brave dog sees a bird in the forest"
  "cats and dogs play in the garden"
  "she says that the old man walks slowly"
  "the red car that runs fast stops near the river"
"""
import random
import re
from typing import List, Tuple, Dict, Set

# ═══════════════════════════════════════════════════════════════════════════
# Special Tokens
# ═══════════════════════════════════════════════════════════════════════════

PAD, MASK, BOS, EOS, UNK = 0, 1, 2, 3, 4
SPECIAL_TOKENS = ["<pad>", "<mask>", "<bos>", "<eos>", "<unk>"]

# ═══════════════════════════════════════════════════════════════════════════
# Content Vocabulary (15 categories, ~800+ words)
# ═══════════════════════════════════════════════════════════════════════════

VOCAB = {
    # ── Animals (50) ──
    "animals": [
        "cat", "dog", "bird", "fish", "horse", "cow", "sheep", "chicken",
        "rabbit", "mouse", "bear", "lion", "tiger", "wolf", "deer", "elephant",
        "monkey", "snake", "turtle", "duck", "eagle", "fox", "goat", "goose",
        "hamster", "kangaroo", "leopard", "owl", "panda", "parrot", "penguin",
        "pig", "pigeon", "rabbit", "rat", "rooster", "seal", "shark", "squirrel",
        "swan", "turkey", "whale", "zebra", "camel", "donkey", "frog", "hawk",
        "lizard", "lobster", "octopus",
    ],
    # ── Colors (30) ──
    "colors": [
        "red", "blue", "green", "yellow", "white", "black", "orange", "purple",
        "pink", "brown", "gray", "silver", "gold", "cyan", "magenta", "crimson",
        "scarlet", "azure", "emerald", "amber", "violet", "teal", "ivory",
        "coral", "lavender", "turquoise", "maroon", "beige", "indigo", "bronze",
    ],
    # ── Food (50) ──
    "food": [
        "bread", "milk", "cheese", "apple", "banana", "rice", "pasta", "meat",
        "egg", "sugar", "salt", "honey", "coffee", "tea", "wine", "beer",
        "soup", "salad", "tomato", "potato", "onion", "garlic", "carrot",
        "chicken", "beef", "pork", "lamb", "butter", "cream", "yogurt",
        "lemon", "orange", "grape", "cherry", "peach", "mango", "berry",
        "melon", "pumpkin", "pepper", "ginger", "basil", "mint", "chocolate",
        "cookie", "cake", "pie", "candy", "juice",
    ],
    # ── Emotions (40) ──
    "emotions": [
        "happy", "sad", "angry", "calm", "excited", "afraid", "brave", "tired",
        "bored", "amazed", "proud", "grateful", "lonely", "hopeful", "worried",
        "relaxed", "confused", "surprised", "joyful", "peaceful", "anxious",
        "curious", "determined", "disappointed", "disgusted", "embarrassed",
        "enthusiastic", "furious", "gentle", "gloomy", "guilty", "horrified",
        "humble", "impatient", "jealous", "nostalgic", "optimistic", "panicked",
        "pessimistic", "content",
    ],
    # ── Nature (50) ──
    "nature": [
        "sun", "moon", "star", "fire", "earth", "sky", "cloud", "rain", "snow",
        "wind", "storm", "mountain", "river", "lake", "ocean", "forest",
        "desert", "valley", "island", "cave", "hill", "cliff", "beach",
        "field", "meadow", "glacier", "volcano", "waterfall", "canyon",
        "delta", "peninsula", "plateau", "tundra", "jungle", "swamp",
        "lagoon", "reef", "dune", "boulder", "pebble", "sand", "mud",
        "ice", "steam", "smoke", "ash", "dust", "frost", "mist", "dew",
    ],
    # ── Body Parts (25) ──
    "body": [
        "head", "hand", "foot", "arm", "leg", "eye", "ear", "nose", "mouth",
        "heart", "back", "neck", "shoulder", "finger", "toe", "knee",
        "elbow", "wrist", "ankle", "hip", "chest", "stomach", "brain",
        "tongue", "teeth",
    ],
    # ── Clothing (25) ──
    "clothing": [
        "shirt", "pants", "dress", "skirt", "jacket", "coat", "hat", "shoe",
        "boot", "sock", "glove", "scarf", "belt", "tie", "sweater", "vest",
        "jeans", "shorts", "blouse", "uniform", "robe", "cloak", "gown",
        "helmet", "cape",
    ],
    # ── Tools & Objects (30) ──
    "tools": [
        "hammer", "knife", "rope", "bucket", "ladder", "shovel", "axe",
        "saw", "nail", "screw", "wheel", "chain", "lever", "pulley",
        "book", "pen", "paper", "candle", "lamp", "key", "lock", "mirror",
        "clock", "bell", "flag", "sign", "map", "compass", "anchor", "sail",
    ],
    # ── Vehicles (20) ──
    "vehicles": [
        "car", "truck", "bus", "train", "boat", "ship", "plane", "bike",
        "cart", "wagon", "carriage", "canoe", "raft", "submarine",
        "helicopter", "rocket", "scooter", "tractor", "ambulance", "caravan",
    ],
    # ── Buildings & Places (30) ──
    "places": [
        "house", "tower", "bridge", "castle", "church", "temple", "palace",
        "cabin", "tent", "barn", "stable", "garage", "school", "library",
        "hospital", "market", "harbor", "dock", "mill", "factory", "garden",
        "park", "yard", "street", "road", "path", "tunnel", "wall", "gate",
        "fountain",
    ],
    # ── Professions (30) ──
    "professions": [
        "farmer", "baker", "smith", "hunter", "fisher", "sailor", "soldier",
        "doctor", "teacher", "scholar", "priest", "merchant", "builder",
        "weaver", "potter", "miner", "tailor", "carpenter", "mason",
        "gardener", "cook", "guard", "guide", "healer", "scribe",
        "artist", "musician", "dancer", "actor", "writer",
    ],
    # ── Materials (20) ──
    "materials": [
        "wood", "stone", "metal", "iron", "steel", "copper", "glass",
        "paper", "cloth", "leather", "rubber", "plastic", "ceramic",
        "brick", "concrete", "marble", "granite", "clay", "wax", "thread",
    ],
    # ── Plants (25) ──
    "plants": [
        "tree", "flower", "grass", "bush", "vine", "reed", "moss", "fern",
        "oak", "pine", "birch", "willow", "maple", "palm", "cedar",
        "rose", "lily", "tulip", "daisy", "orchid", "thorn", "root",
        "branch", "leaf", "trunk",
    ],
}

# ═══════════════════════════════════════════════════════════════════════════
# Function Words
# ═══════════════════════════════════════════════════════════════════════════

FUNC = {
    # Determiners (15)
    "determiners": [
        "the", "a", "an", "this", "that", "these", "those",
        "some", "many", "few", "all", "both", "each", "every", "no",
    ],
    # Prepositions (25)
    "prepositions": [
        "in", "on", "at", "under", "over", "near", "by", "with", "from",
        "to", "into", "onto", "upon", "above", "below", "behind",
        "between", "among", "through", "across", "around", "beyond",
        "against", "along", "beside",
    ],
    # Primary Verbs (40)
    "verbs": [
        "is", "are", "was", "were", "runs", "jumps", "eats", "drinks",
        "sees", "likes", "wants", "goes", "comes", "plays", "sleeps",
        "sings", "walks", "flies", "swims", "climbs", "hunts", "builds",
        "makes", "finds", "gives", "takes", "brings", "calls", "tells",
        "shows", "feels", "hears", "smells", "knows", "thinks", "says",
        "reads", "writes", "opens", "closes",
    ],
    # Auxiliary / Modal Verbs (12)
    "auxiliaries": [
        "can", "could", "will", "would", "shall", "should", "may", "might",
        "must", "does", "do", "did",
    ],
    # Adjectives — Size & Shape (25)
    "adj_size": [
        "big", "small", "old", "new", "fast", "slow", "hot", "cold",
        "bright", "dark", "tall", "short", "wide", "narrow", "deep",
        "shallow", "heavy", "light", "thick", "thin", "huge", "tiny",
        "long", "round", "flat",
    ],
    # Adjectives — Quality (25)
    "adj_quality": [
        "beautiful", "wild", "gentle", "fierce", "strong", "weak", "clean",
        "dirty", "sharp", "dull", "smooth", "rough", "soft", "hard",
        "fresh", "stale", "rich", "poor", "loud", "quiet", "sweet",
        "bitter", "warm", "cool", "ancient",
    ],
    # Adverbs (30)
    "adverbs": [
        "quickly", "slowly", "loudly", "quietly", "carefully", "happily",
        "sadly", "bravely", "easily", "gently", "suddenly", "slowly",
        "again", "always", "never", "often", "sometimes", "here", "there",
        "everywhere", "nowhere", "away", "back", "down", "up", "far",
        "near", "together", "alone", "indeed",
    ],
    # Conjunctions (15)
    "conjunctions": [
        "and", "or", "but", "because", "although", "while", "if", "when",
        "where", "since", "unless", "until", "so", "yet", "for",
    ],
    # Pronouns (15)
    "pronouns": [
        "he", "she", "it", "they", "we", "i", "you", "him", "her",
        "them", "us", "me", "his", "her", "their",
    ],
    # Question words (6)
    "question_words": [
        "who", "what", "where", "when", "why", "how",
    ],
    # Query words (for InformationSeeker QueryGenerator)
    "query_words": [
        "tell", "me", "about", "find", "information",
        "search", "for", "learn", "more", "need",
        "show", "get", "retrieve", "fetch", "data",
    ],
    # Quantifiers (10)
    "quantifiers": [
        "one", "two", "three", "many", "several", "plenty", "enough",
        "hundreds", "thousands", "countless",
    ],
}

# ═══════════════════════════════════════════════════════════════════════════
# POS tag sets (for grammar checking and CFG generation)
# ═══════════════════════════════════════════════════════════════════════════

# All nouns (concrete things)
ALL_NOUNS: List[str] = (
    VOCAB["animals"] + VOCAB["food"] + VOCAB["nature"] +
    VOCAB["body"] + VOCAB["clothing"] + VOCAB["tools"] +
    VOCAB["vehicles"] + VOCAB["places"] + VOCAB["materials"] + VOCAB["plants"]
)

# Nouns that can be subjects (animate + professions + pronouns)
ANIMATE_NOUNS: List[str] = VOCAB["animals"] + VOCAB["professions"]
SUBJECT_NOUNS: List[str] = ANIMATE_NOUNS + FUNC["pronouns"]

# All adjectives
ALL_ADJ: List[str] = (
    VOCAB["colors"] + VOCAB["emotions"] +
    FUNC["adj_size"] + FUNC["adj_quality"]
)

# All verbs
ALL_VERBS: List[str] = FUNC["verbs"]

# All content words for vocab building
ALL_CONTENT: List[str] = ALL_NOUNS + ALL_ADJ + VOCAB["emotions"]

# All function words
ALL_FUNC: List[str] = (
    FUNC["determiners"] + FUNC["prepositions"] + FUNC["verbs"] +
    FUNC["auxiliaries"] + FUNC["adverbs"] + FUNC["conjunctions"] +
    FUNC["pronouns"] + FUNC["quantifiers"] + FUNC["question_words"]
)

# ═══════════════════════════════════════════════════════════════════════════
# Vocabulary Builder
# ═══════════════════════════════════════════════════════════════════════════

def build_vocab() -> Tuple[Dict[str, int], Dict[int, str], List[str]]:
    """Build token↔id mappings. Returns (tok2id, id2tok, all_words)."""
    words = list(SPECIAL_TOKENS)
    for cat_words in VOCAB.values():
        words.extend(cat_words)
    for func_words in FUNC.values():
        words.extend(func_words)
    # Dedup preserving order
    words = list(dict.fromkeys(words))
    tok2id = {w: i for i, w in enumerate(words)}
    id2tok = {i: w for w, i in tok2id.items()}
    return tok2id, id2tok, words


def get_vocab_size() -> int:
    """Total vocabulary size including special tokens."""
    _, _, words = build_vocab()
    return len(words)


# ═══════════════════════════════════════════════════════════════════════════
# Context-Free Grammar (CFG) Sentence Generator
# ═══════════════════════════════════════════════════════════════════════════

class CFGGenerator:
    """Context-Free Grammar that generates diverse, grammatically valid sentences.

    Grammar rules (production probabilities in parentheses):

      S     → NP VP (0.5) | NP VP Conj S (0.15) | NP VP Adv (0.15) | NP VP PP (0.2)
      NP    → Det AdjP N (0.25) | Det N (0.25) | Pronoun (0.2) | Det AdjP N PP (0.15) | Det AdjP N RelCl (0.15)
      AdjP  → Adj (0.7) | Adj AdjP (0.3)
      VP    → V (0.15) | V NP (0.3) | V NP PP (0.15) | V PP (0.1) | Aux V (0.1) | V Adv (0.1) | V S (0.1)
      PP    → Prep NP (1.0)
      RelCl → that VP (0.5) | who VP (0.5)

    This generates sentences like:
      "the brave lion sees a small bird in the old forest"
      "she says that the black cat sleeps"
      "the eagle that flies high dives quickly"
      "the farmer builds a house and the baker makes bread"
    """

    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)

    def _choice(self, lst):
        return self.rng.choice(lst)

    def _chance(self, p):
        return self.rng.random() < p

    def generate_S(self) -> List[str]:
        """Generate a full sentence (S production)."""
        r = self.rng.random()
        if r < 0.50:
            # S → NP VP
            return self.generate_NP() + self.generate_VP()
        elif r < 0.65:
            # S → NP VP Conj S  (compound sentence)
            left = self.generate_NP() + self.generate_VP()
            conj = [self._choice(FUNC["conjunctions"][:4])]  # and, or, but, so
            # Make right clause shorter
            if self._chance(0.6):
                right = self.generate_NP() + self.generate_VP()
            else:
                right = self.generate_VP()  # drop subject (shared)
            return left + conj + right
        elif r < 0.80:
            # S → NP VP Adv
            return self.generate_NP() + self.generate_VP() + [self._choice(FUNC["adverbs"])]
        else:
            # S → NP VP PP
            return self.generate_NP() + self.generate_VP() + self.generate_PP()

    def generate_NP(self) -> List[str]:
        """Noun phrase."""
        r = self.rng.random()
        if r < 0.20:
            # NP → Pronoun
            return [self._choice(FUNC["pronouns"])]
        elif r < 0.45:
            # NP → Det N
            return [self._choice(FUNC["determiners"][:7]),
                    self._choice(ALL_NOUNS)]
        elif r < 0.70:
            # NP → Det AdjP N
            return [self._choice(FUNC["determiners"][:7])] + \
                   self.generate_AdjP() + [self._choice(ALL_NOUNS)]
        elif r < 0.85:
            # NP → Det AdjP N PP
            return [self._choice(FUNC["determiners"][:7])] + \
                   self.generate_AdjP() + [self._choice(ALL_NOUNS)] + \
                   self.generate_PP()
        else:
            # NP → Det AdjP N RelCl
            return [self._choice(FUNC["determiners"][:7])] + \
                   self.generate_AdjP() + [self._choice(ANIMATE_NOUNS)] + \
                   self.generate_RelCl()

    def generate_AdjP(self) -> List[str]:
        """Adjective phrase."""
        adj = self._choice(ALL_ADJ)
        if self._chance(0.3):
            # AdjP → Adj AdjP (recursive)
            return [adj] + self.generate_AdjP()
        return [adj]

    def generate_VP(self) -> List[str]:
        """Verb phrase."""
        r = self.rng.random()
        if r < 0.15:
            # VP → V (intransitive)
            return [self._choice(FUNC["verbs"][4:])]  # action verbs only
        elif r < 0.45:
            # VP → V NP
            return [self._choice(FUNC["verbs"][4:])] + self.generate_NP()
        elif r < 0.60:
            # VP → V NP PP
            return [self._choice(FUNC["verbs"][4:])] + \
                   self.generate_NP() + self.generate_PP()
        elif r < 0.70:
            # VP → V PP
            return [self._choice(FUNC["verbs"][4:])] + self.generate_PP()
        elif r < 0.80:
            # VP → Aux V
            return [self._choice(FUNC["auxiliaries"]),
                    self._choice(FUNC["verbs"][4:])]
        elif r < 0.90:
            # VP → V Adv
            return [self._choice(FUNC["verbs"][4:]),
                    self._choice(FUNC["adverbs"])]
        else:
            # VP → V (simple)
            return [self._choice(FUNC["verbs"][4:])]

    def generate_PP(self) -> List[str]:
        """Prepositional phrase."""
        return [self._choice(FUNC["prepositions"][:10])] + self.generate_NP()

    def generate_RelCl(self) -> List[str]:
        """Relative clause."""
        rel = "that" if self._chance(0.5) else "who"
        return [rel] + self.generate_VP()

    def generate_sentence(self) -> List[str]:
        """Generate one complete sentence."""
        return self.generate_S()

    def generate_dataset(self, n: int = 5000, max_len: int = 16,
                         seed: int = 42) -> List[List[str]]:
        """Generate n sentences, filtering by length.

        Args:
            n: number of sentences to generate
            max_len: maximum sequence length (tokens)
            seed: RNG seed

        Returns:
            List of word lists
        """
        self.rng = random.Random(seed)
        sentences = []
        attempts = 0
        while len(sentences) < n and attempts < n * 5:
            attempts += 1
            s = self.generate_sentence()
            if 4 <= len(s) <= max_len:
                sentences.append(s)
        return sentences


# ═══════════════════════════════════════════════════════════════════════════
# POS Tagger (for grammar checking)
# ═══════════════════════════════════════════════════════════════════════════

# Build word→POS lookup
_WORD_POS: Dict[str, str] = {}

def _build_pos_lookup():
    """Build a word → POS tag mapping."""
    global _WORD_POS
    if _WORD_POS:
        return
    for cat, words in VOCAB.items():
        if cat in ("animals", "food", "nature", "body", "clothing",
                    "tools", "vehicles", "places", "materials", "plants"):
            for w in words:
                _WORD_POS[w] = "NOUN"
        elif cat == "colors":
            for w in words:
                _WORD_POS[w] = "ADJ"
        elif cat == "emotions":
            for w in words:
                _WORD_POS[w] = "ADJ"
        elif cat == "professions":
            for w in words:
                _WORD_POS[w] = "NOUN"
    for w in FUNC["determiners"]:
        _WORD_POS[w] = "DET"
    for w in FUNC["prepositions"]:
        _WORD_POS[w] = "PREP"
    for w in FUNC["verbs"]:
        _WORD_POS[w] = "VERB"
    for w in FUNC["auxiliaries"]:
        _WORD_POS[w] = "AUX"
    for w in FUNC["adj_size"] + FUNC["adj_quality"]:
        _WORD_POS[w] = "ADJ"
    for w in FUNC["adverbs"]:
        _WORD_POS[w] = "ADV"
    for w in FUNC["conjunctions"]:
        _WORD_POS[w] = "CONJ"
    for w in FUNC["pronouns"]:
        _WORD_POS[w] = "PRON"
    for w in FUNC["quantifiers"]:
        _WORD_POS[w] = "QUANT"
    for w in FUNC["question_words"]:
        _WORD_POS[w] = "QWORD"
    for w in ["that", "who"]:
        _WORD_POS[w] = "REL"


def get_pos(word: str) -> str:
    """Get POS tag for a word."""
    _build_pos_lookup()
    return _WORD_POS.get(word, "UNK")

# Words that can be multiple POS tags depending on context
_AMBIGUOUS = {
    "that": ["DET", "REL"],
    "who": ["REL", "PRON"],
}


def tag_sequence(words: List[str]) -> List[str]:
    """Tag a sequence of words with POS tags.

    Handles ambiguous words contextually:
      - "that" after a NOUN → REL (relative pronoun)
      - "that" at sentence start → DET (determiner)
      - "who" → always REL (we don't have who-questions in vocab)
    """
    _build_pos_lookup()
    tags = []
    for i, w in enumerate(words):
        if w == "that":
            # Look at previous tag: if NOUN → REL, else DET
            prev_tag = tags[-1] if tags else None
            if prev_tag == "NOUN":
                tags.append("REL")
            else:
                tags.append("DET")
        elif w == "who":
            tags.append("REL")
        else:
            tags.append(_WORD_POS.get(w, "UNK"))
    return tags


# ═══════════════════════════════════════════════════════════════════════════
# Grammar Checker (for HRM Reviewer model)
# ═══════════════════════════════════════════════════════════════════════════

def check_grammar(words: List[str]) -> Tuple[bool, str]:
    """Check if a word sequence is grammatically well-formed.

    Uses POS pattern matching against valid templates.

    Returns:
        (is_valid, reason)
    """
    if len(words) < 3:
        return False, "too_short"

    tags = tag_sequence(words)
    tag_str = " ".join(tags)

    # Valid POS patterns (regex on tag sequences)
    # Core: needs a subject (DET?/PRON) and a verb
    valid_patterns = [
        # [Det?] [Adj*] Noun/ProN Verb ... (standard SVO/SVI)
        r"^(DET\s+)?(ADJ\s+)*(NOUN|PRON)\s+(VERB|AUX)(\s+.*)?$",
        # Adjective-initial: "brave dogs run"
        r"^ADJ\s+(NOUN)\s+(VERB|AUX)(\s+.*)?$",
        # Compound: ... CONJ (Det?) Noun/ProN Verb
        r"^.*CONJ\s+(DET\s+)?(ADJ\s+)*(NOUN|PRON)\s+(VERB|AUX).*$",
        # Quantifier start: "many cats sleep"
        r"^(QUANT\s+)(NOUN|PRON)\s+(VERB|AUX)(\s+.*)?$",
    ]

    for pattern in valid_patterns:
        if re.match(pattern, tag_str):
            has_verb = any(t in ("VERB", "AUX") for t in tags)
            has_subject = any(t in ("NOUN", "PRON") for t in tags)
            if has_verb and has_subject:
                return True, "ok"

    return False, f"pattern_mismatch: {tag_str}"


def grammar_score(words: List[str]) -> float:
    """Return a soft grammar score [0, 1]."""
    valid, _ = check_grammar(words)
    if valid:
        return 1.0
    # Partial credit
    tags = tag_sequence(words)
    has_verb = any(t in ("VERB", "AUX") for t in tags) if tags else False
    has_noun = any(t in ("NOUN", "PRON") for t in tags) if tags else False
    has_det = "DET" in tags if tags else False
    return sum([0.3 * has_verb, 0.3 * has_noun, 0.2 * has_det, 0.2])


# ═══════════════════════════════════════════════════════════════════════════
# Self-test
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    tok2id, id2tok, words = build_vocab()
    print(f"Vocabulary size: {len(words)} tokens")
    print(f"  Content words: {len(ALL_CONTENT)}")
    print(f"  Function words: {len(set(ALL_FUNC))}")
    print(f"  Nouns: {len(ALL_NOUNS)}")
    print(f"  Adjectives: {len(ALL_ADJ)}")
    print(f"  Verbs: {len(ALL_VERBS)}")

    # Generate sample sentences
    gen = CFGGenerator(seed=123)
    print(f"\nSample CFG-generated sentences:")
    for i in range(20):
        s = gen.generate_sentence()
        tags = tag_sequence(s)
        valid, reason = check_grammar(s)
        mark = "✓" if valid else "✗"
        print(f"  {mark} {' '.join(s)}")
        print(f"    [{reason}] {' '.join(tags)}")

    # Generate dataset
    dataset = gen.generate_dataset(n=20, seed=42)
    lengths = [len(s) for s in dataset]
    print(f"\nDataset sample: {len(dataset)} sentences")
    print(f"  Length range: {min(lengths)}–{max(lengths)}")
