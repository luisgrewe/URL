<div align="center">

# 🌌 QBCA

**Quantization-Based Clustering Algorithm**

[![Python 3.13+](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/)
[![Dependency Management: uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![Code Style: Flake8](https://img.shields.io/badge/code%20style-flake8-yellow.svg)](https://flake8.pycqa.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-F37626.svg?logo=Jupyter&logoColor=white)](https://jupyter.org/)

An efficient, hierarchical bin-pruning clustering algorithm tailored for high-performance image segmentation and clustering benchmarks.

[Getting Started](#-getting-started) • [Notebooks](#-notebooks--experiments) • [Development](#-development--linting)

</div>

---

## ✨ Features & Highlights

The **QBCA** approach leverages intelligent hierarchical bin-pruning to massively reduce search space during clustering.

- 🚀 **Efficient Quantization:** Aggregates data points into bins to iteratively reduce search overhead.
- 🎨 **Image Segmentation:** Applies Gaussian smoothing and clusters image pixels natively in the `Lab` color space.
- 📊 **Robust Benchmarking:** Direct comparisons against `KMeans` capturing structural validity (Dunn Index, Rand Index) and exact execution times.

---

## 📂 Architecture

```text
📦 URL
 ┣ 📂 config/           # Configuration files for image segmentation runs
 ┣ 📂 data/             # Structured datasets (e.g., wine.arff)
 ┣ 📂 images/           # Generated output figures and benchmarks
 ┣ 📜 qbca.py           # Core Object-Oriented QBCA Implementation
 ┣ 📓 *.ipynb           # Self-contained Jupyter Notebooks for experiments
 ┗ 📜 pyproject.toml    # High-performance dependencies managed via uv
```

---

## 🚀 Getting Started

This repository uses [uv](https://docs.astral.sh/uv/) for incredibly fast dependency tracking and virtual environment management.

### 1️⃣ Install `uv`

Select the command for your operating system:

| Platform | Command |
| :--- | :--- |
| **macOS / Linux** | `curl -LsSf https://astral.sh/uv/install.sh \| sh` |
| **Windows** | `powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 \| iex"` |

### 2️⃣ Clone & Sync

```bash
git clone https://github.com/luisgrewe/URL.git
cd URL
uv sync
```

### 3️⃣ Activate Environment

> [!TIP]
> **Jupyter Users:** Select the project's `.venv` as your Jupyter Kernel to inherit the exact dependency tree as the CLI.

| OS | Command |
| :--- | :--- |
| **macOS / Linux** | `source .venv/bin/activate` |
| **Windows (PowerShell)** | `.venv\Scripts\activate` |

---

## 🧪 Notebooks & Experiments

The experiments act as self-contained end-to-end pipelines. Output plots are stored seamlessly within the `images/` directory.

| Notebook | Description | Output Directory |
| :------- | :---------- | :------ |
| 📓 **`main_analysis.ipynb`** | Core validation. Benchmarks synthetic 2D G5-style Gaussians (k=5) with convergence plots. Evaluates on **Iris** and **Wine** datasets with Rand Index metrics. | 📁 `images/main_analysis/` |
| 🎨 **`image_segmentation.ipynb`** | Color image segmentation driven by `config_segmentation.json`. Applies Gaussian smoothing, Lab transformations, and QBCA clustering. | 📁 `images/image_segmentation/` |
| ⚔️ **`image_segmentation_vs_baseline.ipynb`** | Head-to-head QBCA vs KMeans comparison. Shows execution times, visual results, and Dunn Index metrics side-by-side. | 📁 `images/image_segmentation_vs_baseline/` |

---

## 🛠 Development & Linting

Code quality is enforced through **Flake8** and **nbqa**. All linting tools are managed within your `uv` environment.

<details>
<summary><b>📋 Python & Notebook Linting Commands</b></summary>

### Python Code Analysis (`.py`)
```bash
# Lint the entire repository
uv run flake8 --config .flake8

# Lint a specific file
uv run flake8 qbca.py --config .flake8
```

### Notebook Analysis (`.ipynb`)
```bash
# Check notebook code cells
uv run nbqa flake8 main_analysis.ipynb --config .flake8

# Auto-format issues inside a notebook
uv run nbqa autopep8 main_analysis.ipynb --in-place
```

> **VS Code Integration:** Command Palette → `Python: Select Interpreter` → choose `.venv`. Reload window to see Flake8 findings in real-time.

</details>

</div>
