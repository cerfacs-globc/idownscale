# idownscale

<p float="left">
    <img src="/doc/gcm_20700101.png" width="400"/>
    <img src="/doc/unet_20700101.png" width="400"/>
</p>

## Project Context

The **IRISCC** project provides high-quality, fine-resolution (10 km) climate projection data downscaled from GCM simulations. This repository contains the tools for preprocessing, bias correction, training, and evaluation of DL downscaling models.

---

## 📖 Documentation

Full documentation is available in the `docs/` directory.

- **[Release Notes](RELEASE_NOTES.md)**: Latest stabilization and scientific fixes.
- **[Getting Started](docs/getting_started.rst)**: Installation and directory structure.
- **[Workflow Management](docs/management.rst)**: Running the `run_exp5_full.sh` pipeline with automated integrity checks.
- **[Evaluation](docs/evaluation.rst)**: Detailed metrics and visualization protocols.

---

## ⚡ Quick Start

### Installation & Workspace Setup

```bash
conda activate idownscale_env
./bin/utils/setup_workspace.sh # Configures absolute paths for your environment
```

### Running the Experiment 5 Pipeline

The main entry point for the full automated workflow is `run_exp5_full.sh`.

```bash
# Run the full pipeline with structured logging and automated validation
./run_exp5_full.sh
```

All process logs are stored in `logs/exp5/<TIMESTAMP>/`.
Final scientific validation artifacts (plots, CSVs, PDF reports) are consolidated in `output/exp5/validation/`.

---

## 🏗 Project Structure

- `bin/`: CLI scripts and automated validation utilities.
- `docs/`: Sphinx documentation (RST).
- `iriscc/`: Core library (100% compliant with Ruff and scientific standards).
- `logs/`: (Ignored) Structured execution logs for debugging.
- `output/`: Results and centralized validation artifacts.
- `tests/`: Automated unit tests (integrated with Pytest).