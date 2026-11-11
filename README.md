# A-MEM reproduction: retrieval k=10 versus k=5

Course: FTEC5660, Agentic AI for Business and FinTech.

I ran A-MEM on the first conversation in LoCoMo using DeepSeek. I compared retrieval k=10 with k=5, with three QA runs for each setting. Both settings used the same saved memory and 199 questions.

This is a partial reproduction with a different model and a small data subset. It is not a reproduction of every number in the paper. The original authors' source and MIT license are retained. The six supplied experimental Python files have not been edited while preparing this submission.

## Start here: check the saved experiment

No API key, model download, or third-party Python packages are needed for these commands. From this folder, use Python 3:

```bash
python3 scripts/summarize_results.py
```

The script checks question/reference alignment, matches all six runs to the bundled first conversation, checks aggregate scores and failure fields, and recalculates the saved F1 values using the source code's unique-word rule. It prints the results and writes three CSV tables under `analysis/`.

To show an individual example:

```bash
python3 scripts/summarize_results.py --case painting
python3 scripts/summarize_results.py --case counterexample
python3 scripts/summarize_results.py --case adversarial
```

For checks only:

```bash
python3 scripts/summarize_results.py --check-only
```

These commands read saved answers. They do not generate new answers or call DeepSeek. BLEU-1 aggregates are recomputed from the saved question scores; this standard-library script does not rerun the NLTK BLEU implementation.

## Results

All F1 and BLEU-1 scores below are multiplied by 100. `+/-` is the sample standard deviation of three run means, not variation across questions.

| Metric | k=10 | k=5 | Change (5 minus 10) |
| --- | ---: | ---: | ---: |
| Overall F1, 199 questions | 35.27 +/- 1.60 | 29.91 +/- 1.12 | -5.37 |
| Overall BLEU-1 | 30.19 +/- 1.29 | 25.21 +/- 1.28 | -4.98 |
| Categories 1-4 F1, 152 questions | 36.79 +/- 0.72 | 31.35 +/- 0.29 | -5.44 |
| Categories 1-4 BLEU-1 | 30.26 +/- 0.25 | 25.36 +/- 0.19 | -4.89 |

The saved adversarial-correctness metric averages 70.92% for k=10 and 76.60% for k=5. Under `--legacy_c5_reference`, correct refusals can have low lexical F1. Both views are reported rather than changing the scoring rule after the experiment.

I observed lower mean F1 and BLEU-1 with k=5 in all three run-number comparisons. This does not mean that k=10 is optimal or that it helps every question. No cost or speed improvement was measured.

## Files

| Path | Contents |
| --- | --- |
| `test_advanced_robust.py` | Supplied evaluation entry point |
| `memory_layer_robust.py`, `memory_layer.py` | Supplied memory and retrieval code |
| `llm_text_parsers.py`, `load_dataset.py`, `utils.py` | Supplied templates, data loader, and scoring code |
| `data/locomo10.json` | Dataset file supplied from the working project |
| `results/` | Six original JSON outputs, dependency export, and environment record |
| `configs/` | Human-readable records for k=10 and k=5; not inputs read by the original CLI |
| `scripts/summarize_results.py` | Added offline analysis and demo script |
| `analysis/` | CSV tables regenerated from the saved outputs |
| `docs/A-MEM_Reproducibility_Report.pdf` | Seven-page report; identity and submission links still need completing |
| `docs/DEMO_GUIDE_zh.md` | Recording plan and speaking notes |
| `docs/GITHUB_GUIDE_zh.md` | Steps for creating and uploading the submission repository |
| `CHANGES.md` | What changed in the experiment and in the later packaging work |
| `PROVENANCE.md`, `manifest.json` | Source origin, missing evidence, and SHA-256 hashes |

## Recorded environment

- MacBook Air (M2, 2022), Apple M2, 16 GB RAM.
- macOS 27.0.1, arm64, Python 3.13.5, as recorded on the working computer.
- Key exported versions: sentence-transformers 5.1.2, transformers 4.57.1, torch 2.15.0, openai 3.46.0, bert-score 0.3.13.
- Full versions are in `requirements.txt`, copied without changes from `results/requirements_final.txt`.

This is the environment exported after the experiment. A clean installation of the full pinned environment was not tested during packaging.

## Install for a new API experiment

The following section is for running new paid experiments, not for checking the saved results.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m nltk.downloader punkt punkt_tab wordnet
```

Download and check the local models while Hugging Face access is available:

```bash
HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 python -c "from sentence_transformers import SentenceTransformer; m=SentenceTransformer('all-MiniLM-L6-v2'); print(m.encode(['test']).shape)"
HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 python -c "from bert_score import score; p,r,f=score(['This is a test.'], ['This is a test.'], lang='en'); print(f.item())"
```

The embedding check should print `(1, 384)`. The scoring check compares identical text and should return a value close to 1. These checks download/use local models; they do not call DeepSeek. If the network cannot reach Hugging Face, the files need to be downloaded on a working connection before offline evaluation.

## Configure DeepSeek on the recorded Mac/zsh setup

Run the first line by itself, enter the key locally, and then run the exports:

```zsh
read -s "OPENAI_API_KEY?DeepSeek API key: "
export OPENAI_API_KEY
export OPENAI_BASE_URL="https://api.deepseek.com"
```

Do not put the actual key into a Python file or a commit. The controller reads `OPENAI_API_KEY`; the OpenAI-compatible SDK uses `OPENAI_BASE_URL`.

Code-checked generation settings are:

- Answer temperature: 0.7 for categories 1-4; 0.5 for category 5.
- Most other LLM calls default to temperature 0.7; focused keyword retry uses 0.3.
- `max_tokens=1000`; `extra_body={"thinking": {"type": "disabled"}}`.
- `top_p` is omitted, so its provider default was not pinned.
- `--seed 42` fixes category-5 option order; it is not passed as a generation seed to the API.
- Both conditions use the supplied robust plain-text prompts, including the templates in `llm_text_parsers.py`.

## Run baseline and modification

A new run makes paid API calls. The first run also builds memory if no cache exists. The original six outputs stay in `results/`; the examples below save new outputs separately under `reruns/`.

```bash
mkdir -p reruns
```

Baseline example:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -u test_advanced_robust.py \
  --backend openai --model deepseek-flash \
  --dataset data/locomo10.json --ratio 0.1 \
  --retrieve_k 10 --seed 42 --legacy_c5_reference \
  --output reruns/baseline_k10_run1.json \
  2>&1 | tee reruns/baseline_k10_run1.log
```

Modified example:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -u test_advanced_robust.py \
  --backend openai --model deepseek-flash \
  --dataset data/locomo10.json --ratio 0.1 \
  --retrieve_k 5 --seed 42 --legacy_c5_reference \
  --output reruns/modified_k5_run1.json \
  2>&1 | tee reruns/modified_k5_run1.log
```

After each run finishes successfully, repeat with `run2` and `run3` filenames. Do not start several runs against the cache at once. Keep the validated memory cache for the comparisons. A first successful baseline run creates it; later runs should record cache reuse. Once all six new files exist:

```bash
python3 scripts/summarize_results.py --results-dir reruns --output-dir reruns/analysis
```

The runtime cache is not included in this package. Rebuilding it may give different notes and scores. The original six repeats measured QA variation with one fixed memory, not full-pipeline variation with six independent memories.

## How to read k and the metrics

`retrieve_k` controls initial search matches. The memory function also appends linked notes, with a limit that depends on k. It is not an exact count of final prompt notes, and changing 10 to 5 does not demonstrate a 50% reduction in context tokens.

F1 uses sets of unique words after lowercasing and replacing `. , ! ?` with spaces. BLEU-1 uses NLTK word tokens, sentence-level weights `(1, 0, 0, 0)`, method1 smoothing, and the brevity penalty. The report averages these per-question values. Category-5 correctness is a phrase/trap-answer heuristic from the supplied code, not a separate human evaluation.

The result JSON files omit `retrieve_k`. File names and the recorded commands identify the conditions. The later config records and manifest make the submitted snapshot easier to inspect; they cannot retrospectively prove which command produced an old file. The original logs and cache were not supplied for this package.

## Attribution and limits

Upstream project: [WujiangXu/A-mem](https://github.com/WujiangXu/A-mem).

Paper: Xu et al., [A-Mem: Agentic Memory for LLM Agents](https://arxiv.org/html/2502.12110v11), 2025.

The original upstream commit used for the downloaded project was not recorded. The included working source snapshot is identified by hashes in `manifest.json`; no original commit hash has been invented. The supplied `LICENSE` is retained. The dataset is the file supplied from the working project; this README does not assign it a new license.

The six API experiments were performed on the recorded Mac. Packaging checks here covered syntax, source/result consistency, the dataset, and the offline analysis/demo commands. They did not rerun the paid experiment or reinstall the full environment.

Before final submission, fill in the report name/ID and repository/recording links, add any available original run logs, and confirm that the teacher can access the repository and Zoom recording.
