# groundgate Multilingual Evaluation Benchmark Results

## Important Methodological Caveats
1. **Circularity Note:** The ungrounded-answer rate for Setup (c) (`gate`) is computed using
   the same deterministic Python checks that define the gate itself. Therefore, it is a circular
   consistency verification and must NOT be presented as an independent external accuracy metric.
2. **Sample Size:** With a 30-question evaluation dataset, observed percentage variations between
   setups are illustrative and anecdotal; they do not represent statistical significance.
3. **Placeholder Knowledge Pack:** The knowledge pack consists of synthetic, fictional agricultural
   schemes and clearly fake amounts to prevent real-world misguidance during hackathon development.

## Setup Performance Comparison

| Setup | Questions | Errors | Citation Valid | Ungrounded Answers | Trap Wrong Answer | Trap Refusal | Answerable Refusal | Avg Calls | Avg Extra Calls | Avg Latency | Escalation Rate |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| raw | 30 | 0 | 100.0% | 10.0% | 0.0% | 100.0% | 0.0% | 1.0 | 0.0 | 3.09s | 0.0% |
| gate | 30 | 0 | 100.0% | 0.0% | 0.0% | 100.0% | 5.0% | 1.03 | 0.03 | 7.97s | 3.3% |

## Performance by Language

### Setup: raw
| Language | Questions | Errors | Citation Valid | Ungrounded Answers | Trap Refusal | Answerable Refusal | Avg Calls | Avg Latency |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| EN | 10 | 0 | 100.0% | 14.3% | 100.0% | 0.0% | 1.0 | 0.01s |
| HI | 10 | 0 | 100.0% | 0.0% | 100.0% | 0.0% | 1.0 | 0.03s |
| MR | 10 | 0 | 100.0% | 16.7% | 100.0% | 0.0% | 1.0 | 9.22s |

### Setup: gate
| Language | Questions | Errors | Citation Valid | Ungrounded Answers | Trap Refusal | Answerable Refusal | Avg Calls | Avg Latency |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| EN | 10 | 0 | 100.0% | 0.0% | 100.0% | 14.3% | 1.1 | 13.62s |
| HI | 10 | 0 | 100.0% | 0.0% | 100.0% | 0.0% | 1.0 | 10.26s |
| MR | 10 | 0 | 100.0% | 0.0% | 100.0% | 0.0% | 1.0 | 0.01s |

