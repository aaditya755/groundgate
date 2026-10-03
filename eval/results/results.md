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

| Setup | Questions | Citation Valid | Ungrounded Answers | Trap Wrong Answer | Trap Refusal | Answerable Refusal | Avg Calls | Avg Extra Calls | Avg Latency | Escalation Rate |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| gate | 10 | 100.0% | 0.0% | 0.0% | 100.0% | 0.0% | 1.0 | 0.0 | 4.29s | 0.0% |

## Performance by Language

### Setup: gate
| Language | Questions | Citation Valid | Ungrounded Answers | Trap Refusal | Answerable Refusal | Avg Calls | Avg Latency |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| EN | 0 | 0.0% | 0.0% | 0.0% | 0.0% | 0.0 | 0.0s |
| HI | 0 | 0.0% | 0.0% | 0.0% | 0.0% | 0.0 | 0.0s |
| MR | 10 | 100.0% | 0.0% | 100.0% | 0.0% | 1.0 | 4.29s |

