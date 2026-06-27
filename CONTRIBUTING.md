# Contributing

MetricDrive is currently a personal research portfolio project. Contributions
are welcome when they improve reproducibility, clarity, metrics, or experiment
quality.

## Development

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -e ".[dev]"
python3 -m unittest discover
```

## Project Principles

- Keep the default demo small enough to run on a laptop.
- Keep raw datasets out of git.
- Prefer clear metrics and ablations over oversized models.
- Document limitations and failure cases.
- Use only public data, public papers, and independent implementation details.
