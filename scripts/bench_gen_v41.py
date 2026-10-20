"""Benchmark de generacion del checkpoint v4.1 (post-retrain plain-CE).
Correr desde la raiz del repo con python de Windows (GPU).
"""
import sys, os, json, time, math
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))
from mdlm_v4_optimized import MDLMBPEV4, MDLMConfig, BPETokenizer, sample_full_parallel_v4
from safetensors.torch import load_file

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REL = os.path.join(ROOT, 'hf_model_v4')
DEV = 'cuda'

CFG = json.load(open(os.path.join(REL, 'config.json')))
model = MDLMBPEV4(MDLMConfig(vocab_size=CFG['vocab_size'], d_model=CFG['d_model'],
                             n_heads=CFG['n_heads'], n_layers=CFG['n_layers'],
                             max_seq_len=CFG['max_seq_len'], d_ff=CFG['d_ff'], dropout=0.1))
sd = load_file(os.path.join(REL, 'model.safetensors'))
model.load_state_dict(sd, strict=True)
model.eval().to(DEV)
tok = BPETokenizer(os.path.join(REL, 'tokenizer.json'))
print(f'params {sum(p.numel() for p in model.parameters()):,} cargados (strict)', flush=True)

prompts = [
    'The future of artificial intelligence',
    'Climate change is one of the biggest challenges',
    'To build a reliable system, you need',
    'The history of science shows that',
    'In a small town at the edge of the mountains',
    'Economists have long debated whether',
]
results = []
for p in prompts:
    ids = tok.encode(p, add_special=False)
    torch.cuda.synchronize()
    t0 = time.time()
    out = sample_full_parallel_v4(model, tok, prompt_ids=ids, seq_len=128,
                                  n_samples=1, n_steps=32, temperature=0.7, device=DEV)
    torch.cuda.synchronize()
    dt = time.time() - t0
    n_tok = len(tok.encode(out[0], add_special=False))
    results.append({'prompt': p, 'tps': round(n_tok / dt, 1), 'time_s': round(dt, 2),
                    'text': out[0][:220]})
    print(f"[{n_tok} tok, {n_tok/dt:.1f} tok/s] {p} -> {out[0][:120]!r}", flush=True)

tps = [r['tps'] for r in results]
summary = {'checkpoint': 'v4.1 plain-CE retrain', 'gpu': torch.cuda.get_device_name(0),
           'n_steps': 32, 'seq_len': 128, 'mean_tps': round(sum(tps)/len(tps), 1),
           'results': results}
json.dump(summary, open(os.path.join(ROOT, 'results', 'gen_bench_v41.json'), 'w'), indent=1)
print(f"\nMEAN: {summary['mean_tps']} tok/s  -> results/gen_bench_v41.json")
