# Swarm: salto cualitativo para MDLM v4.1 (2026-10-20)

5 agentes (A: frontera dLLM, B: decoding, C: verifiers, F: métricas, G: vocab/datos).
73 hallazgos, 90+ fuentes. JSONs por agente en este dir.

## Consenso triangulado (fases)
- FASE 0: protocolo r10 de métricas (F) — gates G1-G7, anti-hacking gen-PPL (2606.08417)
- FASE 1: sampler v2 (A+B) — semi-AR bloques 16 + low-conf remasking + RCR + 128 steps + T 0.2 + anti-EOS (solo inferencia, 1-2 días)
- FASE 2: verifier loop (C) — best-of-N Skywork-RM-0.6B + FK steering k=4 + T2M remask-don't-replace (2-4 días)
- FASE 3: datos + KD (G+A) — FineWeb-Edu≥1K tok 70% / PG-19 15% / wiki 10% / peS2o 5%, ventanas intra-doc, complementary masking, KD Qwen3-0.6B; 1-2B tokens = 33-67h GPU
- FASE 4 (opcional): conversión BD3-LM (repo kuleshov-group/bd3lms) — gen-PPL −25-30% documentado

## Hallazgos que refutan intuiciones
1. Gen-PPL es hackeable (samplers naive SOTA-genPPL con texto incoherente) — nunca sola, dual-juez + ratio vs refs humanos.
2. Vocab 10K→32K NO ayuda ahora: en undertraining el vocab óptimo BAJA (Tao et al. NeurIPS 2024); re-evaluar 16-24K solo tras escalar datos.
3. Fix anti-deriva más barato: ventanas de entrenamiento que no crucen bordes de documento (check de 10 min).
4. CaRE: la temperatura domina MAUVE → comparar configs siempre a NFE/T igualados.
