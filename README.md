# TReg
<h3 align="center">
<img src="https://github.com/robert-graf/TReg/blob/main/figures/logo.png" width="200">
</h3>

## Installation Guide

## Installation Guide

### System Requirements
- Python 3.9 or higher.
- Tested on Ubuntu and Windows.
- One of the following:
  - Nvidia-GPU with 8 GB of RAM or more.
  - A mps device (--ddevice mps)
  - A strong CPU; This is usually very slow. (--ddevice cpu)
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
pip install hf-deepali
pip install nnunetv2
```

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

### Steps Leg

1. Open `treg_leg.ipynb`
2. Select the **VIBESegmentator** Python environment
3. Update all file paths in the notebook
4. Run the cells sequentially

---

### Working with Landmark (`.mrk.json`) Files


* Landmark files can be created and opened in **3D Slicer**
### `poi.json`
* If saved in **Local Coordinates**, landmark positions correspond to **pixel indices**
* Local Indexing starts at **0**, so values may offset by one in soft ware like ITKSnap

