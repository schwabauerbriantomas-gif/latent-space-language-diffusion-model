"""eval_r10 — multi-axis qualitative evaluation protocol for MDLM (swarm report F).

Stages implemented:
  S1 degeneration (CPU): rep-n, longest-repeat-run, distinct-n, self-BLEU4, Zipf, entropy
  S2 gen-PPL dual judge: gpt2-large (primary) + pythia-1.4b (cross-family), ratio vs human refs
  S4 thematic drift: slope of cos(chunk_i, prompt) over 4 chunks (gpt2-large embeddings)
  S6 syntax: CoLA acceptability (textattack/roberta-base-CoLA)

Protocol rules (from report F): gen-PPL never alone (hacking, arXiv:2606.08417);
report ratio vs human refs; dual judge guards judge-familiarity bias.

Usage:
  python scripts/eval_r10.py --sampler v1 --temp 0.7 --steps 64 --tag run_v1
  python scripts/eval_r10.py --sampler v2 --block 16 --temp 0.2 --steps 32 --tag run_v2
"""
import argparse, json, math, os, sys, time
import numpy as np
import torch

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
os.chdir(ROOT)

from mdlm_v4_optimized import MDLMBPEV4, MDLMConfig, BPETokenizer

DEV = "cuda" if torch.cuda.is_available() else "cpu"
PROMPTS_FILE = "eval/prompts_r10.jsonl"
REFS_FILE = "eval/refs_r10.jsonl"


def load_jsonl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8")]


# ---------------------------------------------------------------- generation
def build_sampler(kind, block, steps, temp, anti_eos):
    """Returns fn(model, tok, prompt_ids, n_new) -> ids of full sequence."""
    from mdlm_v4_optimized import sample_full_parallel_v4

    if kind == "v1":
        def fn(model, tok, prompt_ids, n_new, eos_id):
            seq = 128
            outs = sample_full_parallel_v4(
                model, tok, prompt_ids=prompt_ids, seq_len=seq, n_samples=1,
                n_steps=steps, temperature=temp, device=DEV)
            return outs[0], None
        return fn
    # v2: semi-AR block decoding + anti-EOS annealing (context is NEVER corrupted:
    # committed tokens from previous blocks are real token ids, masks only inside
    # the current block). RCR removed: it re-masked stale relative positions.
    def fn(model, tok, prompt_ids, n_new, eos_id):
        model.eval()
        seq = 128
        out_full = list(prompt_ids)
        n_blocks = math.ceil(n_new / block)
        with torch.no_grad():
            for b in range(n_blocks):
                if len(out_full) - len(prompt_ids) >= n_new: break
                remaining_target = n_new - (len(out_full) - len(prompt_ids))
                bsz = min(block, remaining_target)
                window = torch.full((1, seq), tok.mask_id, dtype=torch.long, device=DEV)
                ctx_ids = out_full[-(seq - bsz):]
                window[0, :len(ctx_ids)] = torch.tensor(ctx_ids, device=DEV)
                total_masked = bsz
                for step in range(steps):
                    remaining = int((window[0] == tok.mask_id).sum())
                    if remaining == 0: break
                    t = torch.tensor([max(remaining / max(total_masked, 1), 1e-3)], device=DEV)
                    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                        logits = model(window, t)
                    logits = logits.float()
                    # anti-EOS annealing: suppress EOS only early in the whole generation
                    frac_done = (len(out_full) - len(prompt_ids)) / max(n_new, 1)
                    if anti_eos and eos_id is not None and frac_done < 0.6:
                        logits[..., eos_id] -= 3.0 * (1.0 - frac_done)
                    probs = torch.softmax(logits, dim=-1)
                    if not torch.isfinite(probs).all():
                        probs = torch.nan_to_num(probs, nan=1.0 / probs.shape[-1], posinf=1.0, neginf=0.0)
                    mask_idx = (window[0] == tok.mask_id).nonzero(as_tuple=True)[0]
                    if mask_idx.numel() == 0: break
                    probs_m = probs[0, mask_idx]  # [K, V]
                    flat = probs_m / temp if temp > 0 else probs_m.clamp(min=1e-9)
                    flat = flat / flat.sum(-1, keepdim=True)
                    sampled_m = torch.multinomial(flat, 1).squeeze(-1)  # [K]
                    conf_m = probs_m.max(dim=-1)[0]
                    cum_prev = 0.5 * (1 - math.cos(math.pi * step / steps))
                    cum_now = 0.5 * (1 - math.cos(math.pi * (step + 1) / steps))
                    target = int(round(cum_now * total_masked))
                    k = max(1, min(target - int(round(cum_prev * total_masked)), mask_idx.numel()))
                    order = conf_m.argsort(descending=True)
                    commit_local = order[:k]
                    commit = mask_idx[commit_local]
                    window[0, commit] = sampled_m[commit_local]
                out_full.extend(window[0, len(ctx_ids):].tolist())
                if eos_id in window[0, len(ctx_ids):].tolist():
                    idx = out_full.index(eos_id, len(prompt_ids))
                    out_full = out_full[:idx + 1]
                    break
        return tok.decode(out_full), None
    return fn


# ---------------------------------------------------------------- S1 degeneration (pure CPU)
def ngram_repetition(tokens, n):
    if len(tokens) < n: return 0.0
    seen, rep = {}, 0
    for i in range(len(tokens) - n + 1):
        g = tuple(tokens[i:i+n])
        if g in seen: rep += 1
        seen[g] = True
    return rep / max(len(tokens) - n + 1, 1)


def longest_repeat_run(tokens):
    best = 0
    for n in range(2, min(20, len(tokens) // 3 + 1)):
        for start in range(len(tokens) - 2 * n + 1):
            seg = tuple(tokens[start:start+n])
            run = 1
            pos = start + n
            while pos + n <= len(tokens) and tuple(tokens[pos:pos+n]) == seg:
                run += 1; pos += n
            best = max(best, run)
    return best  # repetitions of the same n-gram


def distinct_n(tokens, n):
    if len(tokens) < n: return 0.0
    gs = set(tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1))
    return len(gs) / max(len(tokens)-n+1, 1)


def self_bleu4(texts):
    """Approx self-BLEU-4: mean max 4-gram precision of each text vs others."""
    import collections
    def ngrams(t, n=4):
        toks = t.lower().split()
        return [tuple(toks[i:i+n]) for i in range(len(toks)-n+1)]
    all_ng = [set(ngrams(t)) for t in texts]
    scores = []
    for i, t in enumerate(texts):
        ng = ngrams(t)
        if not ng: continue
        best = 0.0
        mine = set(ng)
        for j, other in enumerate(all_ng):
            if i == j or not other: continue
            ov = len(mine & other) / max(len(mine), 1)
            best = max(best, ov)
        scores.append(best)
    return float(np.mean(scores)) if scores else 0.0


def zipf_coef(tokens):
    import collections
    cnt = collections.Counter(tokens)
    freqs = sorted(cnt.values(), reverse=True)
    if len(freqs) < 10: return 0.0
    x = np.log10(np.arange(1, len(freqs)+1))
    y = np.log10(freqs)
    return float(-np.polyfit(x, y, 1)[0])


def unigram_entropy(tokens):
    import collections
    cnt = collections.Counter(tokens)
    N = sum(cnt.values())
    return -sum((c/N) * math.log2(c/N) for c in cnt.values())


def stage1_degeneration(texts):
    per = {"rep2": [], "rep3": [], "rep4": [], "longest_run": [],
           "distinct2": [], "distinct3": [], "entropy": [], "zipf": []}
    for t in texts:
        toks = t.lower().split()
        if len(toks) < 20: continue
        per["rep2"].append(ngram_repetition(toks, 2))
        per["rep3"].append(ngram_repetition(toks, 3))
        per["rep4"].append(ngram_repetition(toks, 4))
        per["longest_run"].append(longest_repeat_run(toks))
        per["distinct2"].append(distinct_n(toks, 2))
        per["distinct3"].append(distinct_n(toks, 3))
        per["entropy"].append(unigram_entropy(toks))
        per["zipf"].append(zipf_coef(toks))
    out = {k: float(np.mean(v)) for k, v in per.items() if v}
    out["self_bleu4"] = self_bleu4(texts)
    return out


# ---------------------------------------------------------------- S2 gen-PPL dual judge
_JUDGES = {}
def get_judge(name):
    if name in _JUDGES: return _JUDGES[name]
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.float16).to(DEV).eval()
    _JUDGES[name] = (tok, model)
    return _JUDGES[name]


@torch.no_grad()
def text_ppl(judge_name, texts, max_len=512):
    tok, model = get_judge(judge_name)
    ppls = []
    eos = tok.eos_token_id
    for t in texts:
        ids = tok(t, return_tensors="pt", truncation=True, max_length=max_len).input_ids.to(DEV)
        if ids.shape[1] < 8: continue
        out = model(ids, labels=ids)
        ppls.append(float(torch.exp(out.loss)))
    return float(np.mean(ppls)) if ppls else float("nan")


def stage2_genppl(samples, refs_text):
    out = {}
    for judge in ["gpt2-large", "EleutherAI/pythia-1.4b"]:
        s = text_ppl(judge, samples)
        r = text_ppl(judge, refs_text)
        out[judge] = {"samples_ppl": round(s, 2), "refs_ppl": round(r, 2),
                      "ratio_samples_over_refs": round(s / r, 3)}
    return out


# ---------------------------------------------------------------- S4 drift (embeddings from gpt2-large)
@torch.no_grad()
def embed_texts(judge_name, texts, max_len=64):
    tok, model = get_judge(judge_name)
    embs = []
    for t in texts:
        ids = tok(t, return_tensors="pt", truncation=True, max_length=max_len).input_ids.to(DEV)
        outp = model(ids, output_hidden_states=True)
        h = outp.hidden_states[-1][0]          # [L, d]
        embs.append(h.mean(0).float().cpu().numpy())
    return np.stack(embs)


def cos(a, b): return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def stage4_drift(samples, prompts, judge="gpt2-large"):
    slopes, final_sims = [], []
    for p, t in zip(prompts, samples):
        words = t.split()
        if len(words) < 80: continue
        chunks = [" ".join(words[i:i+20]) for i in range(0, 80, 20)]
        embs = embed_texts(judge, [p] + chunks)
        pe = embs[0]
        sims = [cos(pe, e) for e in embs[1:]]
        x = np.arange(len(sims))
        slope = float(np.polyfit(x, sims, 1)[0])
        slopes.append(slope); final_sims.append(sims[-1])
    return {"drift_slope_mean": round(float(np.mean(slopes)), 4) if slopes else None,
            "final_sim_mean": round(float(np.mean(final_sims)), 4) if final_sims else None,
            "n": len(slopes)}


# ---------------------------------------------------------------- S6 syntax (CoLA)
_COLA = {}
def cola_scores(texts, max_len=128):
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    if "cola" not in _COLA:
        tk = AutoTokenizer.from_pretrained("textattack/roberta-base-CoLA")
        md = AutoModelForSequenceClassification.from_pretrained(
            "textattack/roberta-base-CoLA").to(DEV).eval()
        _COLA["cola"] = (tk, md)
    tk, md = _COLA["cola"]
    scores = []
    with torch.no_grad():
        for t in texts:
            sents = [s.strip() for s in t.replace("!", ".").replace("?", ".").split(".") if 5 < len(s.split()) < 80]
            sents = sents[:8]
            if not sents:
                continue
            batch = tk(sents, return_tensors="pt", padding=True, truncation=True, max_length=max_len).to(DEV)
            logits = md(**batch).logits[:, 1]  # acceptable
            scores.extend(torch.sigmoid(logits).tolist())
    return float(np.mean(scores)) if scores else float("nan")


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sampler", choices=["v1", "v2"], default="v1")
    ap.add_argument("--steps", type=int, default=64)
    ap.add_argument("--temp", type=float, default=0.7)
    ap.add_argument("--block", type=int, default=16)
    ap.add_argument("--anti-eos", action="store_true", default=True)
    ap.add_argument("--no-anti-eos", dest="anti_eos", action="store_false")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--ckpt", default="checkpoints/mdlm_v4_best.pt")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--samples-from", default=None, help="path to samples.json: skip generation")
    ap.add_argument("--skip-judges", action="store_true", help="S1 only (fast)")
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    prompts = load_jsonl(PROMPTS_FILE)
    refs = load_jsonl(REFS_FILE)
    outdir = f"eval/reports/{args.tag}"
    os.makedirs(outdir, exist_ok=True)

    print(f"loading model {args.ckpt} ...")
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = MDLMConfig(**{k: v for k, v in ckpt["config"].items()})
    model = MDLMBPEV4(cfg).to(DEV).eval()
    model.load_state_dict(ckpt["model_state"], strict=True)
    tok = BPETokenizer()
    gen_fn = build_sampler(args.sampler, args.block, args.steps, args.temp, args.anti_eos)

    t0 = time.time()
    if args.samples_from:
        texts = json.load(open(args.samples_from, encoding="utf-8"))
        print(f"loaded {len(texts)} samples from {args.samples_from}")
    else:
        texts = []
        for i, p in enumerate(prompts):
            ids = tok.encode(p["prompt"], add_special=True)
            text, _ = gen_fn(model, tok, ids, n_new=128, eos_id=tok.eos_id)
            text = " ".join(text.split())
            texts.append(text)
            if (i+1) % 20 == 0:
                print(f"  gen {i+1}/100 ({time.time()-t0:.0f}s)", flush=True)
    json.dump(texts, open(f"{outdir}/samples.json", "w", encoding="utf-8"), ensure_ascii=False)

    report = {"tag": args.tag, "sampler": args.sampler, "temp": args.temp,
              "steps": args.steps, "block": args.block, "seed": args.seed,
              "gen_seconds": round(time.time()-t0, 1)}

    print("S1 degeneration ...")
    report["S1_degeneration"] = stage1_degeneration(texts)
    report["S1_refs_baseline"] = stage1_degeneration([r["ref"] for r in refs])

    if not args.skip_judges:
        print("S2 gen-PPL dual judge ...")
        report["S2_genppl"] = stage2_genppl(texts, [r["ref"] for r in refs])
        print("S4 thematic drift ...")
        report["S4_drift"] = stage4_drift(texts, [p["prompt"] for p in prompts])
        print("S6 syntax CoLA ...")
        report["S6_cola"] = round(cola_scores(texts), 4)
        report["S6_cola_refs"] = round(cola_scores([r["ref"] for r in refs]), 4)

    json.dump(report, open(f"{outdir}/metrics.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print(json.dumps(report, indent=1)[:2000])
    print(f"\nreport -> {outdir}/metrics.json")


if __name__ == "__main__":
    main()
