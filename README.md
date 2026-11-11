# A-MEM reproduction: k=10 versus k=5

FTEC5660 - Agentic AI for Business and FinTech

I used A-MEM to answer questions about one LoCoMo conversation with DeepSeek. I compared retrieval k=10 with k=5 and ran each setting three times on the same 199 questions. The memory was built once and reused for the later QA runs.

Original project: https://github.com/WujiangXu/A-mem

Paper: https://arxiv.org/html/2502.12110v11

This is a partial reproduction with a different model and a small data subset. The exact original upstream commit was not recorded. The authors' license is retained.

## Files

- `test_advanced_robust.py`: evaluation entry point.
- `memory_layer.py`, `memory_layer_robust.py`: memory storage, updates, and retrieval.
- `llm_text_parsers.py`: plain-text prompt templates and response parsing.
- `load_dataset.py`: dataset loading and reference selection.
- `utils.py`: scoring functions.
- `data/locomo10.json`: conversation and QA data.
- `requirements.txt`: exported Python dependency versions.
- `results/`: six original result JSONs, environment record, and dependency export.
- `CHANGES.md`: the setup adaptation and retrieval-parameter comparison.

The experiment runs through command-line arguments. Separate configuration JSONs or helper-script directories are not required. The report and Zoom recording are submitted separately.

## Environment and settings

- MacBook Air (M2, 2022), Apple M2, 16 GB RAM.
- Recorded environment: macOS 27.0.1, arm64, Python 3.13.5.
- Model: `deepseek-flash`; provider: DeepSeek.
- Base URL: `https://api.deepseek.com`; backend: `openai`.
- Answer temperature: 0.7 for categories 1-4; 0.5 for category 5.
- Most other calls default to temperature 0.7; focused keyword retry uses 0.3.
- `max_tokens=1000`; thinking disabled; `top_p` not explicitly set.
- First conversation only: `--ratio 0.1`; `--seed 42`; `--legacy_c5_reference`.
- Conversation-local memory; index refresh disabled.

Seed 42 fixes the order of category-5 answer options. It does not fix API generation. Both retrieval settings use the same prompt code. The dependency files record the working environment; a clean installation was not repeated during submission preparation.

## Install

From the downloaded repository folder:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m nltk.downloader punkt punkt_tab wordnet
```

Prepare the local models while Hugging Face is reachable:

```bash
HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 python -c "from sentence_transformers import SentenceTransformer; m=SentenceTransformer('all-MiniLM-L6-v2'); print(m.encode(['test']).shape)"
HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 python -c "from bert_score import score; p,r,f=score(['This is a test.'], ['This is a test.'], lang='en'); print(f.item())"
```

These checks do not call DeepSeek. The first should print `(1, 384)` and the second a score close to 1. Downloads require working network access; later model loading can use the local cache.

## Run a new experiment

New experiments make paid API calls. Checking the existing results in the final section does not.

On Mac/zsh, run the first line alone and enter the key locally, then export the variables:

```zsh
read -s "OPENAI_API_KEY?DeepSeek API key: "
export OPENAI_API_KEY
export OPENAI_BASE_URL="https://api.deepseek.com"
```

Keep the real API key out of source files and commits. Save new outputs separately so the original results remain unchanged:

```bash
mkdir -p reruns
```

Baseline:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -u test_advanced_robust.py \
  --backend openai --model deepseek-flash \
  --dataset data/locomo10.json --ratio 0.1 \
  --retrieve_k 10 --seed 42 --legacy_c5_reference \
  --output reruns/baseline_k10_run1.json \
  2>&1 | tee reruns/baseline_k10_run1.log
```

Modification:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -u test_advanced_robust.py \
  --backend openai --model deepseek-flash \
  --dataset data/locomo10.json --ratio 0.1 \
  --retrieve_k 5 --seed 42 --legacy_c5_reference \
  --output reruns/modified_k5_run1.json \
  2>&1 | tee reruns/modified_k5_run1.log
```

Repeat with run2 and run3 output/log names. Wait for each run to finish before starting the next, and reuse the same successfully built cache. A new memory build may produce different scores from the saved experiment. The original cache and run logs are not included in this submission.

## Results

F1 and BLEU-1 scores are multiplied by 100. The table reports the mean and sample standard deviation across three run means.

| Metric | k=10 | k=5 | Change (5 minus 10) |
| --- | ---: | ---: | ---: |
| Overall F1, 199 questions | 35.27 +/- 1.60 | 29.91 +/- 1.12 | -5.37 |
| Overall BLEU-1 | 30.19 +/- 1.29 | 25.21 +/- 1.28 | -4.98 |
| Categories 1-4 F1, 152 questions | 36.79 +/- 0.72 | 31.35 +/- 0.29 | -5.44 |
| Categories 1-4 BLEU-1 | 30.26 +/- 0.25 | 25.36 +/- 0.19 | -4.89 |

Sample variances of run means, on the squared 0-100 scale:

| Metric | k=10 | k=5 |
| --- | ---: | ---: |
| Overall F1 | 2.5675 | 1.2643 |
| Overall BLEU-1 | 1.6724 | 1.6370 |
| Categories 1-4 F1 | 0.5121 | 0.0868 |
| Categories 1-4 BLEU-1 | 0.0624 | 0.0356 |

The saved category-5 refusal-correctness metric averages 70.92% for k=10 and 76.60% for k=5. Correct refusals may receive low F1 against the legacy reference, so categories 1-4 are also shown separately. The refusal metric is the supplied code's phrase/trap-answer heuristic, not a human assessment.

F1 uses unique-word sets after lowercasing and replacing `. , ! ?` with spaces. BLEU-1 uses NLTK word tokens and sentence-level BLEU with method1 smoothing. The reported values average per-question scores.

## Check the saved averages without an API

Run this from the repository root. It uses only the Python standard library and the six JSON files already in `results/`:

```bash
python3 - <<'PY'
import json
import statistics as s
from pathlib import Path

for k, label in [(10, 'baseline'), (5, 'modified')]:
    runs = [json.loads(Path(f'results/{label}_k{k}_run{i}.json').read_text()) for i in (1, 2, 3)]
    assert all(len(d['individual_results']) == 199 for d in runs)
    for name, cats in [('overall', None), ('categories 1-4', {1, 2, 3, 4})]:
        for metric in ['f1', 'bleu1']:
            values = [100 * s.mean(r['metrics'][metric] for r in d['individual_results']
                      if cats is None or r['category'] in cats) for d in runs]
            print(f'k={k}, {name}, {metric}: mean={s.mean(values):.2f}, '
                  f'SD={s.stdev(values):.2f}, variance={s.variance(values):.4f}')
PY
```

The six result JSONs do not store `retrieve_k`; the condition names follow the saved filenames and recorded execution commands. This check recalculates averages from saved scores, not new model answers.

## Conclusion and limits

On this conversation, k=10 had higher average F1 and BLEU-1 in all three run-number comparisons, while the refusal metric favored k=5. This does not show that k=10 is optimal on the full dataset. The model, data scope, and retrieval settings differ from the paper.

The code adds linked memories after the initial top-k search, so k is not an exact count of final prompt notes. The experiment did not measure comparable speed, tokens, or cost. The repeats used one fixed memory and therefore do not measure the variability of rebuilding memory from scratch.
