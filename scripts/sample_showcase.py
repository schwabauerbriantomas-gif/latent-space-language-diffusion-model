"""Showcase of samples from the retrained v4.1 checkpoint (ppl ~62).

Generates samples for several prompts at two temperatures, saves to
results/samples_v41.md for the release card.
"""
import sys, os, torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from mdlm_v4_optimized import MDLMBPEV4, MDLMConfig, BPETokenizer, sample_full_parallel_v4

CKPT = os.path.join(os.path.dirname(__file__), "..", "checkpoints", "mdlm_v4_best.pt")
DEV = "cuda" if torch.cuda.is_available() else "cpu"

def main():
    ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
    cfg = MDLMConfig(**{k: v for k, v in ckpt["config"].items()})
    model = MDLMBPEV4(cfg).to(DEV).eval()
    model.load_state_dict(ckpt["model_state"], strict=True)
    tok = BPETokenizer()
    print(f"loaded step={ckpt.get('step')} ppl={ckpt.get('ppl'):.1f} | device={DEV}")

    prompts = ["The", "Once upon a time", "Scientists have discovered", "Dear friend,"]
    lines = [f"# MDLM v4.1 samples — step {ckpt.get('step')}, eval PPL {ckpt.get('ppl'):.1f}\n"]
    for temp in (0.7, 0.3):
        for p in prompts:
            ids = tok.encode(p, add_special=True)
            outs = sample_full_parallel_v4(model, tok, prompt_ids=ids,
                                           seq_len=128, n_samples=2,
                                           n_steps=64, temperature=temp, device=DEV)
            for i, o in enumerate(outs):
                o = " ".join(o.split())  # collapse whitespace for readability
                lines.append(f"\n## prompt=`{p}` temp={temp} sample={i}\n> {o}")
                print(f"\n[temp={temp} | {p} | #{i}]\n{o}\n")
    out = os.path.join(os.path.dirname(__file__), "..", "results", "samples_v41.md")
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\nsaved -> {out}")

if __name__ == "__main__":
    main()
