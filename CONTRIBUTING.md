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
metricdrive export-demo --output docs/demo
```

## Project Principles

- Keep the default demo small enough to run on a laptop.
- Keep raw datasets out of git.
- Prefer clear metrics and ablations over oversized models.
- Document limitations and failure cases.
- Use only public data, public papers, and independent implementation details.

## Conduct

See [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md). In particular, keep discussions
public-safe and avoid implying access to private company data, internal systems,
or non-public implementation details.
