<div align="center">

# 🌌 QBCA

**Quantization-Based Clustering Algorithm**

[![Python 3.13+](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/)
[![Dependency Management: uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![Code Style: Flake8](https://img.shields.io/badge/code%20style-flake8-yellow.svg)](https://flake8.pycqa.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-F37626.svg?logo=Jupyter&logoColor=white)](https://jupyter.org/)

An efficient, hierarchical bin-pruning clustering algorithm tailored for high-performance image segmentation and clustering benchmarks.

[Getting Started](#-getting-started) •
[Notebooks](#-notebooks--experiments) •
[Development](#-development--linting)

</div>

---

## ✨ Features & Highlights

The **QBCA** approach leverages intelligent hierarchical bin-pruning to massively reduce search space during clustering.
T- 🚀 **Efficient Quantization:** Aggregates data points inT- 🚀 **Efficient Quantization:** Aggregead.
- 🎨 **Image Segmentation:** Applies Gaussian smoothing and clusters image pixels natively - 🎨 **Image Segmentation:** Applies Gaussian smoothing and clusters image pixels natively - 🎨 **Image Segmentation:** Applies Gaussian smoothing and clusters image pies- 🎨 **Image Segmentation:** Applies Gaussian smoothing and clusters image pixels natively - 🎨 **Image Seonfig/           # Configuration files for image segmentation runs
 ┣ 📂 data/             # Structured datasets ┣ 📂 data/             # Structured datasets ┣ 📂 data/             # Structured datasets ┣ 📂 data/             # Structent ┣ 📂 data/     on
 ┣ 📓 *.ipynb  ┣ 📓 *.ipynb  ┣ 📓 *.ipynb  �boo ┣ 📓 *.ipynb  ┣ 📓 *.ipynbect.toml    # Hig ┣ 📓 *.ipynb  ┣ �s man ┣ 📓 *.ipynb  ┣ 📓 *.ipynb  ┣ 📓 *.ipynb  �boo ┣ 📓 *.ipynb  ┣ 📓 *.ipynbect.toml    # Hig ┣ �t dependency tracking and virtual environment management.

### 1. Install `uv`
Select the command for your opeSelect the command for your opeSelect the command for your opeSelect the command for your opeSelect the command for your opeSelect the command for your opeSelect the command for your opeSelect the command for your opeSelect the command for your opeSelect the command for your opeSelect the command for your opeSelect the command for your opeSelect the command for your opeSelect the command for your opeSelect the command for your opeSelect t
### 3. Activate Environment
> [!TIP]
> **Jupyter Users:** Make sure to select the project's `.venv` as your Jupyter Kernel so your notebooks inherit the exact dependency tree as the CLI.

| OS | Activation Command |
| :--- | :--- |
| **macOS / Linux** | `source .venv/bin/activate` |
| **Windows** | `.venv\Scripts\activate` |

---

## 🧪 Notebooks & Experiments

The experiments act as self-contained end-to-end pipelines. Output plots are stored seamlessly within the `images/` directory.

| Notebook | Description | Outputs || Notebook | Descr------- | :------ |
| 📓 **`main_analysis.ipynb`** | **Core validation.** Benchmarks synthetic 2D G5-style Gaussians (k=5) and evaluates convergence on **Iris** (sklearn) and **Wine** (`data/wine.arff`). Measures timing vs. iterations and Rand Index. | 📁 `images/main_analysis/` |
| 🎨 **`image_segmentation.ipynb`** | **Color Image Segmentation.** Driven by `config_segmentation.json`. Smooths elements with Gaussians, applies Lab transformations, clusters using QBCA.| 🎨 **`image_segmentatientation/` | 🎨 **`image_segmentation.ipynb`** | **Color Image Segmentation.** Driven by `config_segmentation.json`. Smooths elements with Gaussians, ale execut| 🎨 **`image_segmentat, a| 🎨 **ndex metrics. | 📁 `images/image_segmentation_vs_baseline| 🎨 **`image_segmentation.it & Linting

Code standard practices are enforced thCode standard practices are enforced thCode standard practices are enforced thCode standard practices are enfs>
<summary><b>Clic<summary><b>Clic<summary><b>Clic<summary><b>Clic<summary><b>Clic<summarypy<summary><b>Clic<summary><b>Clic<summary><b>Clic<summary><b>Clic<summary><b>Clic<summarypy<summary><b>Clic<summary><b>Clic<summary><b>Clic<summary><b>Clic<summary><b>Clic<summarypy<summary><b>Clic<summary><b>Clic<summary><b>Clic<summary><b>Clic<summary><b>Clic<sumal<summary><b>Clic<summary><b>Clic<summary><b>Clic<summary><b>Clic<summary><b>Clic<summarypy<summary><b>Clic<summary><b>Clic<summary><b>Clic<summary><b>Clic<summary><b>Clic<summarypy<summary><b>Clic<summary><b>Clic<summary><b>Clic<summary><b>Clic<summary><b>Clic<summarypy<summary><b>Clic<summary><b>Clic<summary><b>Clic<summary><b>Clic<summary><b>Clic<sumal<summary><b>Clic<summary><b>Clic<summary><b>Clic<summary><b>Clic<summary><b>Clic<summarypy<summary><b>Clic<summary><b>Clic<summary><b>Clic<summary><b>Clic<sumlt with ❤️ for High-Performance Clustering.</sub>
</div>
