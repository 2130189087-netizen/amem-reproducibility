# Changes

## Experimental setup

I used the supplied A-MEM robust evaluation entry point with DeepSeek's OpenAI-compatible API. The controller disables thinking mode and keeps a maximum of 1000 output tokens. Both conditions share this adaptation.

## One experiment variable

The baseline command uses `--retrieve_k 10`. The modified command uses `--retrieve_k 5`. The model, first conversation, seed, prompt code, scoring options, and saved memory were kept the same in the procedure. Each setting was run three times.

The k parameter affects initial matches and the linked-memory expansion limit. It does not guarantee an exact number of prompt notes.

`configs/baseline_k10.json` and `configs/modified_k5.json` record these settings after the experiments. The original evaluator does not read those files. They must not be described as original runtime logs.

## Submission preparation

The six uploaded Python source files were copied byte-for-byte. No experimental algorithm or scoring code was changed while packaging. I added README/run instructions, an offline result-summary script, CSV summaries, provenance information, and demo/upload guides.

Any commits created now organize the supplied snapshot and submission materials. They are not a reconstruction of the dates or commits used during the original experiment. The original upstream commit and a complete historical diff were not recorded.
