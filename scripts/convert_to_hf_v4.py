"""
Convert MDLM v4 checkpoint to safetensors format for HuggingFace.
Updated for v4 architecture (RMSNorm, SwiGLU, simplified AdaLN).
"""
import sys, json, torch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from mdlm_v4_optimized import MDLMConfig as MDLMConfigV4, MDLMBPEV4

REPO = Path(__file__).resolve().parent.parent
ckpt_path = REPO / "checkpoints" / "mdlm_v4_best.pt"

if not ckpt_path.exists():
    print(f"ERROR: {ckpt_path} not found. Train the model first.")
    sys.exit(1)

ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
config = MDLMConfigV4(**ckpt["config"])
model = MDLMBPEV4(config)
model.load_state_dict(ckpt["model_state"])

# Save as safetensors
from safetensors.torch import save_file

output_dir = REPO / "hf_model_v4"
output_dir.mkdir(exist_ok=True)

state_dict = {k: v.contiguous() for k, v in model.state_dict().items()}
save_file(state_dict, str(output_dir / "model.safetensors"))

# Save config
config_dict = {
    "architectures": ["MDLMBPEV4"],
    "model_type": "mdlm",
    "version": "v4",
    "d_model": config.d_model,
    "n_heads": config.n_heads,
    "n_layers": config.n_layers,
    "vocab_size": config.vocab_size,
    "max_seq_len": config.max_seq_len,
    "d_ff": config.d_ff,
    "dropout": config.dropout,
    "use_checkpointing": config.use_checkpointing,
    "torch_dtype": "float32",
    "optimizations": [
        "RMSNorm (no bias, no mean-centering)",
        "SwiGLU FFN (better than GELU, fewer params)",
        "Simplified AdaLN (4×d_model instead of 6×d_model)",
        "Logit-normal timestep sampling",
        "ELBO-weighted loss",
        "Weight tying (no output bias)",
        "torch.compile compatible",
    ],
    "training": {
        "params": sum(p.numel() for p in model.parameters()),
        "ppl": ckpt.get("ppl", "N/A"),
        "step": ckpt.get("step", "N/A"),
        "loss": ckpt.get("eval_loss", "N/A"),
    },
}

with open(output_dir / "config.json", "w") as f:
    json.dump(config_dict, f, indent=2)

params = sum(p.numel() for p in model.parameters())
print(f"Saved model.safetensors ({params/1e6:.1f}M params)")
print(f"Saved config.json")
print(f"  PPL: {ckpt.get('ppl', 'N/A')}")
print(f"  Step: {ckpt.get('step', 'N/A')}")
print(f"  Eval loss: {ckpt.get('eval_loss', 'N/A')}")
