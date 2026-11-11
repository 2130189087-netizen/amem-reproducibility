# A-MEM Reproduction: k=10 vs k=5

FTEC5660 - Agentic AI for Business and FinTech

## 1. Project

I used the [A-MEM authors' code](https://github.com/WujiangXu/A-mem) to test memory-based question answering on LoCoMo. My target was the F1 and BLEU-1 evaluation reported in [Table 1 of the paper](https://arxiv.org/html/2502.12110v11).

I used DeepSeek and the first conversation, which has 199 questions. I compared retrieval k=10 with k=5 and ran each setting three times. The same memory was reused. This is a partial reproduction; my scores are not directly comparable to the paper because the model and experiment settings differ.

The code is based on the authors' project, and the original license is retained. The exact upstream commit used when downloading the project was not recorded.

## 2. Environment and setup

- Computer: MacBook Air (M2, 2022), Apple M2, 16 GB RAM.
- Recorded software: macOS 27.0.1, arm64, Python 3.13.5. Package versions are in `requirements.txt`.
- API: DeepSeek, model `deepseek-flash`, base URL `https://api.deepseek.com`, OpenAI-compatible backend.
- Answer temperature: 0.7 for categories 1-4 and 0.5 for category 5. Other calls normally use 0.7; the focused keyword retry uses 0.3.
- Generation: `max_tokens=1000`, thinking disabled, `top_p` left at the provider default.
- Data: `data/locomo10.json`, first conversation only (`--ratio 0.1`). Both settings use the same robust prompt code and legacy category-5 scoring.

Run the following from the repository folder:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m nltk.downloader punkt punkt_tab wordnet
```

Prepare the embedding and scoring models while Hugging Face is reachable:

```bash
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"
python -c "from bert_score import score; print(score(['test'], ['test'], lang='en')[2].item())"
```

These commands may download model files but do not call DeepSeek. The dependency versions were exported from the working Mac; a fresh installation was not tested again during submission preparation.

In Mac/zsh, run the first line below on its own, enter the API key locally, and then run the two exports:

```zsh
read -s "OPENAI_API_KEY?DeepSeek API key: "
export OPENAI_API_KEY
export OPENAI_BASE_URL="https://api.deepseek.com"
```

Do not include the real key in the repository.

## 3. Run the experiments

New runs use paid API calls. Save them in `reruns/` to keep the original files in `results/` unchanged:

```bash
mkdir -p reruns
```

Baseline, k=10:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -u test_advanced_robust.py \
  --backend openai --model deepseek-flash \
  --dataset data/locomo10.json --ratio 0.1 \
  --retrieve_k 10 --seed 42 --legacy_c5_reference \
  --output reruns/baseline_k10_run1.json \
  2>&1 | tee reruns/baseline_k10_run1.log
```

Modification, k=5:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -u test_advanced_robust.py \
  --backend openai --model deepseek-flash \
  --dataset data/locomo10.json --ratio 0.1 \
  --retrieve_k 5 --seed 42 --legacy_c5_reference \
  --output reruns/modified_k5_run1.json \
  2>&1 | tee reruns/modified_k5_run1.log
```

Repeat each setting with `run2` and `run3` output/log names. Finish one run before starting the next and reuse the same successfully built memory cache. Seed 42 controls category-5 option order, not API generation. The model, data, prompts, and scoring stay the same; the experiment variable is `retrieve_k`.

## 4. Results

The six original outputs are in `results/`. F1 uses unique-word overlap, and BLEU-1 uses sentence-level unigram BLEU. Each run averages the question scores. All values below are on a 0-100 scale; SD and variance are calculated across the three run means, using the sample formula.

| Metric | k=10, mean +/- SD | k=5, mean +/- SD | k=10 variance | k=5 variance |
| --- | ---: | ---: | ---: | ---: |
| Overall F1 (199 questions) | 35.27 +/- 1.60 | 29.91 +/- 1.12 | 2.5675 | 1.2643 |
| Overall BLEU-1 | 30.19 +/- 1.29 | 25.21 +/- 1.28 | 1.6724 | 1.6370 |
| Categories 1-4 F1 (152 questions) | 36.79 +/- 0.72 | 31.35 +/- 0.29 | 0.5121 | 0.0868 |
| Categories 1-4 BLEU-1 | 30.26 +/- 0.25 | 25.36 +/- 0.19 | 0.0624 | 0.0356 |

Reducing k lowered the categories 1-4 mean F1 by 5.44 points and BLEU-1 by 4.89 points. The saved category-5 refusal metric instead favored k=5: 76.60% versus 70.92%. With legacy references, a correct refusal can get low F1, so I report categories 1-4 separately as well.

## 5. Conclusion and limits

In these runs, k=10 gave better average F1 and BLEU-1, but it did not improve every answer. Only one conversation and one fixed memory were tested, so this does not establish the best k for the full dataset. The code also adds linked memories after the initial search; k is not an exact count of final prompt notes. I did not measure comparable speed or cost.

The original memory cache and run logs are not included. A fresh memory build may give different results. The result JSONs omit `retrieve_k`, so the condition labels rely on the saved filenames and recorded commands. The report contains the paper comparison and debugging details; the report and Zoom recording are submitted separately.
