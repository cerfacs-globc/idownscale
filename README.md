# idownscale

<p float="left">
    <img src="/doc/gcm_20700101.png" width="400"/>
    <img src="/doc/unet_20700101.png" width="400"/>
</p>

## Project Context

The **IRISCC** project provides high-quality, fine-resolution (10 km) climate projection data downscaled from GCM simulations. This repository contains the tools for preprocessing, bias correction, training, and evaluation of DL downscaling models.

---

## 📖 Documentation

The full documentation is now available via **Sphinx** and can be hosted on **ReadTheDocs**.

- **[Getting Started](docs/getting_started.rst)**: Installation and directory structure.
- **[Workflow Management](docs/management.rst)**: How to run the `run_exp5_full.sh` pipeline, use `FORCE`/`REGENERATE`, and monitor logs.
- **[Data Preprocessing](docs/preprocessing.rst)**: Building datasets and computing statistics.
- **[Model Training](docs/training.rst)**: Architecture details and inference loops.
- **[Evaluation](docs/evaluation.rst)**: Metrics and visualization.

---

## ⚡ Quick Start

### Installation

```bash
conda create -n idownscale_env python=3.11
conda activate idownscale_env
pip install -r requirements.txt
```

### Running the Experiment 5 Pipeline

The main entry point for the full automated workflow is `run_exp5_full.sh`.

```bash
# Run the full pipeline (Phases 1-6)
./run_exp5_full.sh

# Resume a specifically failed phase (e.g., Phase 3 Bias Correction)
START_PHASE=3 STOP_PHASE=3 FORCE=1 ./run_exp5_full.sh
```

---

## 🛠 Workflow Controls

| Variable | Effect |
| :--- | :--- |
| `FORCE=1` | Bypasses `.markers/` check. Resumes scripts from existing data. |
| `REGENERATE=1` | Overwrites existing data and starts from scratch. |
| `START_PHASE` | Starting point (1-6). |
| `STOP_PHASE` | Ending point (1-6). |

---

## 🏗 Project Structure

- `bin/`: CLI scripts for each phase.
- `docs/`: Sphinx documentation (RST).
- `iriscc/`: Core library (models, dataloaders, transforms).
- `scripts/`: Misc utility scripts.
- `tests/`: Unit tests.