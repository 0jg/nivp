
# Neural IVP

Codebase for Griffiths, Wrathmall, and Gardiner, *Solving physics-based initial value problems with unsupervised machine learning*. Phys. Rev. E _111_, 055302. 2025.

## Quickstart

Ensure `uv` is installed on your system. Installation instructions: https://docs.astral.sh/uv/getting-started/installation/.

```bash
# Create a virtual env and install deps
uv venv

# Run the notebook
uv run marimo run examples/harmonic_oscillator.py # or henon_heiles.py
```

Artifacts are written under `./examples/outputs/figures|paths|models/<system_name>/<run_name>/`.
