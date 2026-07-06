"""Convert MDLM checkpoint to safetensors format for HuggingFace."""
import sys, torch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from mdlm_bpe_v3 import MDLMConfig, MDLMBPEV3

REPO = Path(__file__).resolve().parent.parent
ckpt = torch.load(REPO / "checkpoints" / "mdlm_bpe_v3_best.pt",
                   map_location="cpu", weights_only=False)
config = MDLMConfig(**ckpt["config"])
model = MDLMBPEV3(config)
model.load_state_dict(ckpt["model_state"])

# Save as safetensors
from safetensors.torch import save_file

output_dir = REPO / "hf_model"
output_dir.mkdir(exist_ok=True)

state_dict = {k: v.contiguous() for k, v in model.state_dict().items()}
save_file(state_dict, str(output_dir / "model.safetensors"))

# Save config
import json
config_dict = {
    "architectures": ["MDLMBPEV3"],
    "model_type": "mdlm",
    "d_model": config.d_model,
    "n_heads": config.n_heads,
    "n_layers": config.n_layers,
    "vocab_size": config.vocab_size,
    "max_seq_len": config.max_seq_len,
    "d_ff": config.d_ff,
    "dropout": config.dropout,
    "torch_dtype": "float32",
    "training": {
        "params": sum(p.numel() for p in model.parameters()),
        "ppl": ckpt.get("ppl", 102.6),
        "step": ckpt.get("step", 44000),
        "loss": ckpt.get("loss", 4.63),
    },
    "training_details": {
        "data": "Ultra-FineWeb (1M docs, 272M tokens)",
        "epochs": 3,
        "batch_size": 32,
        "seq_len": 128,
        "hardware": "NVIDIA RTX 3090",
        "training_time": "~7 hours",
    }
}

with open(output_dir / "config.json", "w") as f:
    json.dump(config_dict, f, indent=2)

params = sum(p.numel() for p in model.parameters())
print(f"Saved model.safetensors ({params/1e6:.1f}M params)")
print(f"Saved config.json")
