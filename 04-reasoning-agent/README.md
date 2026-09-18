# Self-checking reasoning agent

The agent runs a planner, executor and independent verifier. A failed check is fed into a fresh attempt, with two retries by default. The response keeps the requested JSON schema and exposes only a short explanation, abbreviated plan and verifier checks.

## Run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m unittest discover -s tests -v
OPENAI_API_KEY=... python -m reasoning_agent "A train leaves at 14:30 and arrives at 18:05. How long is the journey?"
```

The application uses OpenAI through the provider boundary in `providers.py`. Prompts live in `reasoning_agent/prompts.py`.

The evaluation script contains eight easy and four tricky cases. Its default fixture mode exercises the agent loop without network calls and produces the checked-in log. Pass `--live` to evaluate the configured model:

```bash
python -m examples.run_examples
OPENAI_API_KEY=... python -m examples.run_examples --live --output examples/live-run-log.jsonl
```

Unit tests use small provider doubles to cover valid output, malformed output, retries and retry exhaustion. Twelve fixture-backed runs are stored in `examples/run-log.jsonl`; the provider is identified in every record.

I initially considered a single prompt that solved and approved its own answer. It made malformed output and false confidence harder to isolate, so the three stages use separate JSON contracts. Given more time, I would add provider-native schema enforcement and domain tools for dates, units and constraint solving.
