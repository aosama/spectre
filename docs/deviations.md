# Deviations from docs/plan-realmodel.md

| Ticket | Plan says | Actual | Why |
|---|---|---|---|
| R3 | Trainable params = 31,482,144 (12 × 2,623,512) | 31,556,016 (12 × 2,629,668) | The plan's per-head arithmetic omitted the gate's `modrelu_bias` (513 params/head): per-head gate is 71,491, not 70,978. Per layer = 12 × (98,432 qv + 71,491 gate) + 590,592 wo = 2,629,668. Implementation unchanged — the count assertion in `tests/test_surgery.py` uses the correct value. |
| R3 | — | Post-surgery model total = 127,647,408 params (frozen 96,091,392) | The gate MLPs add more parameters than the discarded K rows removed; the plan's acceptance item "trainable count exactly 31,482,144" is corrected to 31,556,016. |
| R0 | `load_dataset('wikitext', 'wikitext-2-raw-v1')` | `load_dataset('Salesforce/wikitext', 'wikitext-2-raw-v1')` | huggingface_hub 1.x requires namespace/name; the bare `wikitext` alias no longer resolves. Split is named `validation`, not `valid`. |
| R4 | Baseline PPL expected in [6.0, 7.5] | Measured **30.33**; band corrected to [28, 33] | The plan's band was a misremembered literature number: ~6.5 is *word-level* perplexity; GPT-2 small's token-level WikiText perplexity is ~29-30 (GPT-2 paper: 29.41 on its WikiText-103 subset). CPU/MPS parity check confirmed the evaluator is correct (30.2002 vs 30.2012 on the same subset). The success criterion (SPECTRE ≤ 1.10 × baseline) is unchanged and now anchored to a verified baseline. |
