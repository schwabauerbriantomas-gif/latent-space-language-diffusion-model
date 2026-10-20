"""Build eval_r10 dataset: 100 prompts (4 categories) + 120 human reference continuations.

Prompts: deterministic, seed 42. Categories:
  - wiki (40): first ~25 tokens of real documents from local corpus
  - creative (25): story/narrative openers
  - structured (20): lists/letters/technical
  - adversarial (15): keyword retention test (rare topic words)

References: real human continuations extracted from the local corpus
(ultra_fineweb_en.jsonl): split each doc mid-way, first half = prompt, second = ref.
Only for wiki-like docs of >= 200 tokens (approx by words * 1.3).

Output: eval/prompts_r10.jsonl, eval/refs_r10.jsonl
"""
import json, random, re, sys, os

random.seed(42)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EVAL_DIR = os.path.join(ROOT, "eval")
os.makedirs(EVAL_DIR, exist_ok=True)

CORPUS = os.path.join(ROOT, "data", "ultra_fineweb_en.jsonl")

def clean(t):
    t = re.sub(r"\s+", " ", t).strip()
    return t

def words(t): return len(t.split())

# ---------- collect docs ----------
docs = []
if os.path.exists(CORPUS):
    with open(CORPUS, encoding="utf-8", errors="ignore") as f:
        for i, line in enumerate(f):
            if i >= 4000: break
            try:
                obj = json.loads(line)
            except Exception:
                continue
            text = obj.get("text") or obj.get("content") or ""
            text = clean(text)
            if words(text) >= 150:
                docs.append(text)
print(f"docs collected: {len(docs)}")

# ---------- wiki prompts: doc openers ----------
wiki_prompts = []
for d in docs:
    w = d.split()
    if len(w) < 40: continue
    p = " ".join(w[:25])
    if len(p) > 30 and p[-1] not in ".!?":
        wiki_prompts.append(p)
    if len(wiki_prompts) >= 40: break

# ---------- human reference continuations (split docs) ----------
refs = []
for d in docs:
    w = d.split()
    if len(w) < 200: continue
    mid = len(w) // 2
    # find a sentence boundary near the middle for a clean split
    prefix = " ".join(w[max(0, mid-30):mid+10])
    m = list(re.finditer(r"[.!?] ", prefix))
    if not m: continue
    cut_local = m[-1].end()
    cut = max(0, mid-30) + cut_local
    prompt_txt = " ".join(w[:cut])
    ref_txt = " ".join(w[cut:cut+220])  # ~220 words continuation
    if len(ref_txt.split()) < 100: continue
    refs.append({"prompt": prompt_txt[-500:], "ref": ref_txt})
    if len(refs) >= 120: break
print(f"refs built: {len(refs)}")

# ---------- creative ----------
creative_seeds = [
    "Once upon a time in a small coastal town,",
    "The old lighthouse keeper had a secret he never told anyone:",
    "It was the kind of morning that made people believe in second chances.",
    "When the letter arrived, Marta recognized the handwriting immediately:",
    "The train pulled out of the station just as the rain started,",
    "In the winter of that year, the river froze for the first time in decades,",
    "Nobody in the village remembered who had planted the apple orchard,",
    "The map had been in the family for three generations,",
    "On the last day of summer, the carnival came to town and",
    "He opened the door slowly, expecting the worst, but",
    "The dog waited at the bus stop every afternoon at four,",
    "There was something strange about the house at the end of the lane:",
    "As the sun set behind the mountains, the travelers realized",
    "The library's oldest book had a note tucked between its pages:",
    "Every night at midnight, the machines in the old factory started by themselves,",
    "She had practiced the speech a hundred times, but when she stood up",
    "The fisherman's daughter knew the sea better than any map,",
    "Legend said the cave held the treasure of the last king, and",
    "The photograph showed two children standing in front of a house that",
    "When the power went out, the neighbors gathered in the street and",
    "Grandfather's workshop smelled of cedar and machine oil, and",
    "The letter began with words she had never expected to read:",
    "Nobody had climbed the north face of the mountain since the accident,",
    "In the drawer of the antique desk, Elena found a key with a tag that read:",
    "The storm knocked out the bridge, and with it the last road out of the valley,",
]
structured_seeds = [
    "Dear friend, I am writing to share some exciting news:",
    "Dear Sir or Madam, I am writing regarding your recent advertisement for",
    "To prepare this recipe you will need the following ingredients:",
    "The assembly instructions for this product are as follows:",
    "To set up your development environment, follow these steps:",
    "The committee reached the following conclusions:",
    "Here is a summary of the main findings of the study:",
    "The maintenance schedule for the equipment includes the following tasks:",
    "Dear team, thank you all for your hard work this quarter.",
    "The hiking trail is divided into three sections:",
    "To fix a leaking faucet, you will need the following tools:",
    "The report covers three main areas of concern:",
    "Welcome to the museum. Before you begin your visit, please note:",
    "The warranty covers the following components:",
    "To apply for the scholarship, candidates must submit:",
    "The experiment was conducted under the following conditions:",
    "For best results with this fertilizer, follow the application guide:",
    "The city council approved the following measures:",
    "A healthy sleep routine includes the following habits:",
    "The training program consists of four modules:",
]
adversarial_pairs = [
    ("chrysanthemum", "The botanical garden's rare collection included the chrysanthemum, and"),
    ("telescope", "At the observatory, the old telescope was still used every clear night, and"),
    ("beehive", "The farmer kept a beehive behind the barn, and every autumn"),
    ("xylophone", "In the music room, forgotten in the corner, stood a xylophone, and"),
    ("glacier", "The expedition reached the glacier after nine days of hiking, and"),
    ("origami", "Her grandmother taught her origami using only newspaper, and"),
    ("volcano", "The island's volcano had been dormant for two centuries, and"),
    ("aquarium", "The city aquarium's newest exhibit featured a jellyfish tank, and"),
    ("labyrinth", "The palace garden was designed as a labyrinth, and visitors who"),
    ("harpsichord", "The concert opened with a piece for harpsichord, and the audience"),
    ("metallurgy", "The region's economy was built on metallurgy, and the museum"),
    ("pancake", "Every Sunday morning, the diner served its famous pancake stack, and"),
    ("compass", "Lost in the forest, he checked his compass for the fifth time, and"),
    ("greenhouse", "Behind the school there was a greenhouse where students grew tomatoes, and"),
    ("submarine", "The museum's retired submarine attracted curious visitors, and"),
]

prompts = []
for p in wiki_prompts: prompts.append({"id": f"wiki-{len(prompts)+1:03d}", "cat": "wiki", "prompt": p})
for p in creative_seeds: prompts.append({"id": f"creative-{len(prompts)-39:03d}", "cat": "creative", "prompt": p})
for p in structured_seeds: prompts.append({"id": f"structured-{len(prompts)-64:03d}", "cat": "structured", "prompt": p})
for kw, p in adversarial_pairs: prompts.append({"id": f"adv-{len(prompts)-84:03d}", "cat": "adversarial", "prompt": p, "keyword": kw})

assert len(prompts) == 100, len(prompts)
with open(os.path.join(EVAL_DIR, "prompts_r10.jsonl"), "w", encoding="utf-8") as f:
    for p in prompts: f.write(json.dumps(p, ensure_ascii=False) + "\n")
with open(os.path.join(EVAL_DIR, "refs_r10.jsonl"), "w", encoding="utf-8") as f:
    for r in refs: f.write(json.dumps(r, ensure_ascii=False) + "\n")
print(f"TOTAL: {len(prompts)} prompts, {len(refs)} refs -> eval/")
