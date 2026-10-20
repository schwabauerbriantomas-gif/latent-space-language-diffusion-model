"""best_of_n — FASE 2 verifier loop: generate N samples per prompt, pick best by Skywork RM.

Skywork-Reward-V2-Qwen3-0.6B: sequence classifier over chat-formatted (user, assistant).
Score = logit of 'chosen'. We select argmax score among N generations.

Usage:
  python scripts/best_of_n.py --sampler v2 --block 16 --steps 32 --temp 0.9 --n 8 \
      --tag bon_v2 --eval-dir eval/reports/bon_v2
"""
import argparse, json, math, os, sys, time
import torch

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
os.chdir(ROOT)

from mdlm_v4_optimized import MDLMBPEV4, MDLMConfig, BPETokenizer
from eval_r10 import build_sampler, load_jsonl

RM_PATH = "D:/huggingface/rm_skywork_flat"


@torch.no_grad()
def rm_scores(rm_tok, rm, texts):
    out = []
    for t in texts:
        conv = [{"role": "user", "content": "Write a coherent text continuation."},
                {"role": "assistant", "content": t[:1200]}]
        enc = rm_tok.apply_chat_template(conv, tokenize=True, return_tensors="pt")
        ids = enc.input_ids if hasattr(enc, "input_ids") else enc
        ids = ids.to(rm.device)
        logit = rm(input_ids=ids).logits[0][0].item()
        out.append(logit)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sampler", choices=["v1", "v2"], default="v2")
    ap.add_argument("--block", type=int, default=16)
    ap.add_argument("--steps", type=int, default=32)
    ap.add_argument("--temp", type=float, default=0.9)  # high temp for diversity across N
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--ckpt", default="checkpoints/mdlm_v4_best.pt")
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    prompts = load_jsonl("eval/prompts_r10.jsonl")
    outdir = os.path.join(ROOT, "eval", "reports", args.tag)
    os.makedirs(outdir, exist_ok=True)

    print("loading MDLM ...")
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = MDLMConfig(**{k: v for k, v in ckpt["config"].items()})
    model = MDLMBPEV4(cfg).cuda().eval()
    model.load_state_dict(ckpt["model_state"], strict=True)
    tok = BPETokenizer()

    print("loading Skywork RM ...")
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    rm_tok = AutoTokenizer.from_pretrained(RM_PATH)
    rm = AutoModelForSequenceClassification.from_pretrained(
        RM_PATH, torch_dtype=torch.bfloat16).cuda().eval()

    gen_fn = build_sampler(args.sampler, args.block, args.steps, args.temp, True)
    DEV = "cuda"
    t0 = time.time()
    best_texts, best_scores, all_means = [], [], []
    for i, p in enumerate(prompts):
        ids = tok.encode(p["prompt"], add_special=True)
        cands = []
        for _ in range(args.n):
            text, _ = gen_fn(model, tok, ids, n_new=128, eos_id=tok.eos_id)
            text = " ".join(text.split())
            cands.append(text)
        scores = rm_scores(rm_tok, rm, cands)
        bi = int(max(range(len(cands)), key=lambda j: scores[j]))
        best_texts.append(cands[bi])
        best_scores.append(round(scores[bi], 2))
        all_means.append(sum(scores) / len(scores))
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/100 ({time.time()-t0:.0f}s)  best={scores[bi]:.1f} mean={all_means[-1]:.1f}", flush=True)

    json.dump(best_texts, open(f"{outdir}/samples.json", "w", encoding="utf-8"), ensure_ascii=False)
    json.dump({"tag": args.tag, "n": args.n, "temp": args.temp,
               "best_score_mean": round(sum(best_scores) / len(best_scores), 2),
               "cand_score_mean": round(sum(all_means) / len(all_means), 2)},
              open(f"{outdir}/rm_summary.json", "w", encoding="utf-8"), indent=1)
    print(f"done -> {outdir}/samples.json (best mean {sum(best_scores)/len(best_scores):.1f} vs cand mean {sum(all_means)/len(all_means):.1f})")


if __name__ == "__main__":
    main()
