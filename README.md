# groundgate

A grounded-answer harness around open-weight LLMs. It answers only from a small knowledge pack, requires the model to cite its sources, checks the answer with deterministic code (no extra model call), escalates once to a larger model if the check fails, and otherwise refuses safely. It works with English, Hindi and Marathi (Devanagari).

Built for the Hacktoberfest Hack Day Nashik (MLH), challenge "Best Open-Source AI Project". This is a hackathon prototype, not a production system.

- Repository: https://github.com/aaditya755/groundgate
- Hosted demo: TODO: add the URL, or write "not deployed"
- Demo video: TODO: add the link, or write "none"
- License: Apache-2.0

## The problem

Language models often give confident answers that are not backed by any source. The grounding checkers I looked at were built and evaluated mainly for English, and the language lists I saw for one of them did not include Hindi or Marathi. This is an observation from a short search, not a proven gap, and other projects may already cover it.

## How it works

```
question
  -> detect language (Latin vs Devanagari)
  -> retrieve passages (BM25 over words + character 3-grams, same language only)
  -> best match too weak?  refuse early, no model call
  -> tier-1 model answers in strict JSON: {"answer", "sources", "refuse"}
  -> GATE (pure code):
        pass -> return the answer with its sources
        fail -> one escalation to the tier-2 model, with the failure reasons
                  pass -> return     fail -> safe refusal in the user's language
  -> every step is recorded in a trace
```

### Gate checks

1. The output is valid JSON.
2. A refusal from the model is accepted as a legitimate outcome.
3. Every cited source ID is a passage that was actually retrieved.
4. Every number in the answer (Devanagari digits are converted to ASCII first) appears in the cited passages.
5. Token overlap between the answer and the cited passages meets a configurable threshold.
6. The answer's script matches the question's language.

## Quickstart

Verify these commands against the files in the repository before relying on them.

```bash
git clone https://github.com/aaditya755/groundgate.git
cd groundgate
python -m venv .venv
# Windows: .venv\Scripts\activate      Mac/Linux: source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # Windows: copy .env.example .env  (then add your own keys)
pytest
uvicorn app.server:app --reload
python eval/run_eval.py --setups raw,gate --workers 2
```

`python eval/run_eval.py --dry-run` prints the estimated number of model calls without making any.

## Configuration

Set these in `.env`. Never commit that file. See `.env.example` for the full list.

| Variable | Purpose |
|---|---|
| `NVIDIA_API_KEY` | key for the NVIDIA hosted API |
| `OPENROUTER_API_KEY` | key for OpenRouter (fallback provider) |
| `TIER1_MODEL` | small/fast model ID |
| `TIER2_MODEL` | larger model used for the single escalation |
| `PROVIDER_ORDER` | provider fallback order, for example `nvidia,openrouter` |
| `REPLAY_ONLY` | when true, use only cached responses and make no network calls |

TODO: list any other variables from `.env.example` (for example the rate-limit and threshold settings).

## The knowledge pack

`data/knowledge_pack.jsonl` holds one passage per line with the fields `id`, `topic_id`, `title`, `text`, `lang`, `source_url`. The same fact appears once per language (en, hi, mr) under a shared `topic_id`. Retrieval cannot match across languages, so every question needs passages written in its own language.

Status of the current pack: TODO: write exactly one of "placeholder content, not verified facts" or "human-verified content from: <sources>".

To use your own data, replace the file with your passages, keep the same fields, and have a speaker of each language check the text.

## What is original

Only these three things:

1. Deterministic grounding checks that understand Devanagari (digit normalization, script matching, language-filtered retrieval).
2. A verifier-triggered single escalation: the larger model is called only when the code checks fail.
3. A small English/Hindi/Marathi benchmark with a runner that compares setups.

## Related work

The core ideas already exist. This project applies them to Hindi and Marathi and does not claim to have invented them.

- Grounded-answer gates and citation verifiers: verifiable-rag, RagWarden, RAGGround.
- RAG evaluation frameworks: Ragas, DeepEval, TruLens.
- Guardrails: NVIDIA NeMo Guardrails.
- Small-to-large model cascades and routing: FrugalGPT, AutoMix, RouteLLM, NVIDIA Switchyard.

## Results

30 questions (20 answerable, 10 trap questions that the pack cannot answer), 27 passages. Two setups were run: `raw` (tier-1 model with the JSON contract, no gate) and `gate` (the full harness).

| Setup | Total | Errors | Valid citations | Ungrounded | Trap answered wrongly | Trap refused | Answerable refused | Avg calls | Avg extra calls |
|---|---|---|---|---|---|---|---|---|---|
| raw | 30 | 0 | 100.0% | 10.0% | 0.0% | 100.0% | 0.0% | 1.0 | 0.0 |
| gate | 30 | 0 | 100.0% | 0.0% | 0.0% | 100.0% | 5.0% | 1.03 | 0.03 |

Per-language detail is in `eval/results/results.md`. How to read these numbers:

1. Both setups refused every trap question, so the trap questions did not separate them.
2. "Ungrounded" is measured with the gate's own checks, so it is circular for the gate: 0% for the gate is true by construction. For raw, 10% only means 3 of 30 answers failed those checks. Whether they were real invented facts or false alarms: TODO: write "reviewed manually: <findings>" or "not yet reviewed".
3. The gate escalated once in 30 questions and refused one answerable question (5% over-abstention).
4. Latency figures mix cached and live calls, so they are not reported here as a performance comparison.
5. With 30 questions, the results are anecdotal and not statistically significant.
6. A comparison against an LLM-as-judge setup is implemented in the runner but was not run.
7. During one benchmark run, progress stalled for over 20 minutes before resuming. The cause was not investigated.

## Limitations

- The gate catches invented numbers, names and sources. It does not catch subtle paraphrase errors where the wording is wrong but the numbers and sources are right.
- The benchmark is small, and the pack status is described above.
- Free hosted model endpoints can be slow, can stall, are rate-limited, and may be logged by the provider. Do not enter personal information.
- Retrieval is lexical (BM25), so it depends on questions sharing words with the passages.

## Models and licenses

Runtime calls go to OpenAI-compatible endpoints serving open-weight NVIDIA Nemotron models.

- Tier-1 model: TODO: paste the exact `TIER1_MODEL` ID
- Tier-2 model: TODO: paste the exact `TIER2_MODEL` ID
- Licenses: verify on each model's model card. Not stated here.

## Deployment

A `Dockerfile` is included. Put API keys in the host's secret settings, never in the repository or the image. Check the host's current free-tier limits yourself. The app has a per-visitor rate limit, a daily cap and an input length limit, and shows cached example traces when live model calls fail.

## License

Apache-2.0. See `LICENSE`.
