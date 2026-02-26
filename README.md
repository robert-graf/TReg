# TReg
<h3 align="center">
<img src="https://github.com/robert-graf/TReg/blob/main/figures/logo.png" width="200">
</h3>

## Installation Guide

### System Requirements

* **Python:** 3.10 or newer
* **Operating Systems:** Tested on **Ubuntu** and **Windows**
* **Hardware (one of the following):**

  * **NVIDIA GPU** with CUDA support and sufficient VRAM (recommended)

    * device: `cuda`
  * **Apple Silicon (M2/M3)** using Metal Performance Shaders

    * device: `mps` (not extensively tested)
  * **Strong CPU** (slowest option)

    * device: `cpu`

> ⚠️ Required GPU memory depends on your image size.

---

## Installation

### 1. Open a Terminal

* **Windows:** Search for `cmd` or `Anaconda Prompt`
* **macOS / Linux:** Search for `Terminal`

---

### 2. Create a Python Environment (Recommended)

Using **Anaconda**:

```bash
conda create -n VIBESegmentator python=3.12.0
conda activate VIBESegmentator
```

---

### 3. Install PyTorch

Install a PyTorch version compatible with your system and GPU.
Follow the official instructions here:

👉 [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/)

Example (may differ depending on your setup):

```bash
pip install torch torchvision torchaudio
```

> 💡 Older GPUs may require older PyTorch versions.

---

### 4. Install Required Python Packages

```bash
pip install TPTBox ruamel.yaml configargparse
pip install nnunetv2
```

**Tested versions:**

* `TPTBox==1.6`
* `ruamel.yaml==0.18.6`
* `configargparse==1.7`
* `nnunetv2==2.4.2`

If `nnunetv2` causes issues, reinstall the tested version:

pip uninstall nnunetv2

pip install nnunetv2==2.4.2


---

### 5. Download TReg

```bash
git clone https://github.com/robert-graf/TReg.git
cd TReg
```

⏱️ Installation typically takes **< 30 minutes**, excluding Anaconda/Python installation.
The longest step is usually installing PyTorch.

---

## Running the Example Notebook

We recommend **VS Code** for the smoothest experience.

### Steps

1. Open `treg_example.ipynb`
2. Select the **VIBESegmentator** Python environment
3. Download the example data (**TODO**) or create your own
4. Use your own segmentation or use on of our provided Segmentation tools (can also be called via TPTBox)
   1. [VIBESeg](https://github.com/robert-graf/VIBESegmentator)
   2. [SPINEPS](https://github.com/Hendrik-code/spineps)
5. Update all file paths in the notebook
6. Run the cells sequentially

---

## Working with Landmark (`.mrk.json`) Files

* Landmark files can be created and opened in **3D Slicer**
* If saved in **Local Coordinates**, landmark positions correspond to **pixel indices**

  * Indexing starts at **0**, so values may appear offset by one

