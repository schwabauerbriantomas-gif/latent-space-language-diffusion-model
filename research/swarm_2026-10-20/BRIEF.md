# BRIEF — Swarm: metodologías para salto cualitativo en MDLM 183M

## Contexto del proyecto (verificado, NO re-derivar)
- Modelo: MDLM (masked diffusion LM) v4.1, 182,795,264 params, vocab BPE 10K, seq 128, plain CE sobre posiciones enmascaradas, t logit-normal.
- Entrenado en UNA RTX 3090 24GB: ~136M tokens (1.06M seqs), ~3.5h, ppl held-out 62.4.
- Muestras crudas: sintaxis local correcta (3-10 tokens), formato aprendido (listas, cartas), PERO deriva temática, sin discurso sostenido, colapso a repetición a temperatura baja.
- Pipeline propio ya implementado: guidance logit (repetición/frecuencia/n-gram/top-p), AR-oracle (Qwen3-0.6B teacher-forcing) para regenerar segmentos malos, memoria vectorial SplatsDB.
- Restricción DURA: una sola RTX 3090 24GB, sin cluster; entrenamientos deben caber en horas-días. Código en PyTorch. Fecha: octubre 2026 — papers de 2024-2026 son objetivos primos.
- Objetivo del usuario: salto CUALITATIVO grande y verificable en la calidad del texto generado (coherencia global, discurso sostenido), no mejoras de +2% en ppl.

## Reglas de rigor (obligatorias)
1. CERO fabricación: cada técnica con paper real (título + autores/institución + arXiv id o URL + fecha). Si no encontrás evidencia cuantitativa, escribí "sin números verificados".
2. Prioridad: papers/blogs de código abierto CON experimentos reproducibles en single-GPU, sobre anuncios sin evidencia.
3. Cada finding debe declarar fit a las restricciones (3090, 183M, horas-días).
4. Separar HECHOS (con cita) de TU OPINIÓN (marcada como tal).
5. Techo práctico: ~40 tool calls. Profundidad > amplitud. Cortá y escribí el JSON aunque falte material.
6. Read-only: NO tocas el repo (/mnt/d/...), NO entrenas, NO usás GPU, NO commiteás. Tu única escritura: tu JSON en reports/.
7. Independencia: NO leas los reportes de otros agentes (reports/*.json de hermanos).
8. Escribí tu reporte ANTES de responder, aunque sea parcial: /tmp/mdlm_swarm/reports/<TU_ID>.json

## Esquema JSON de salida
{
  "agent": "<ID>",
  "domain": "<tu dominio>",
  "findings": [{"technique": "...", "source": "paper/url", "date": "YYYY-MM", "evidence": "números con cita", "fit_3090_183m": "high|medium|low + por qué", "expected_impact": "...", "implementation_cost": "...", "risk": "..."}],
  "top_recommendations": [{"rank": 1, "technique": "...", "why": "...", "how_concrete": "plan de integración con el MDLM existente, concreto", "effort": "días/GPU-horas estimadas honestas", "expected_gain": "...", "first_experiment": "el experimento barato que falsifica o valida en <1 día"}],
  "sources": [{"title": "...", "url": "...", "reliability": "alta|media|baja", "date": "..."}],
  "gaps": "qué NO pudiste verificar"
}
