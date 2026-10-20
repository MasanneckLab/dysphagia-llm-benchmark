# Dysphagia LLM benchmark

Code and derived data for our comparison of GPT-5.2, OpenEvidence, Sonar and
Sonar-Reasoning-Pro on guideline-concordant dysphagia management.

The data contain 90 questions and 360 rated answers. Questions form 45 topic
pairs, each with one knowledge and one case question. The frozen analysis data
were checked against the current Supplementary Material 3 on 7 October 2026.

## Reproduce

Use Python 3.12:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements-lock.txt
.venv/bin/python reproduce.py
.venv/bin/python test_revision.py
```

No API calls or keys are needed. The script regenerates all tables, figures and
model fits from `data/response_level.csv`. It uses fixed seeds. Original source
workbooks are not bundled, so their hash test is skipped. The optional
scikit-learn comparison also skips if that package is absent.

## Files

- `analysis.py` and `ordinal_mixed.py`: statistical analysis and figures.
- `data/`: audited response-level data, guideline mapping and exact prompt.
- `results/` and `figures/`: numerical results, fit diagnostics and PDF figures.
- `source_audit/`: collection provenance and explicit data corrections.
- `test_revision.py`: numerical and data-consistency checks.

Bootstrap intervals preserve topic pairs within each domain. The full nested
ordinal model does not give a reliable finite fit. The reduced guideline-adjusted
fit uses 352 responses; descriptive analyses and the no-guideline sensitivity
retain all 360. Pairwise significance depends on the model. See the diagnostic
JSON files and sensitivity tables in `results/`.

## Status

Private working repository for review. Public release and the license remain
subject to author approval. Credentials, raw API logs, manuscripts and reviewer
comments are excluded. The bundled R cross-check has not been run locally.
