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

## Full-Body POI Inference (`example_inference.py`)

Runs the whole pipeline with automatic segmentation over every CT in a BIDS-style dataset:

1. **[VIBESeg-12](https://github.com/robert-graf/VibeSegmentator)** → 12-label body segmentation (`seg-VIBESeg-12`); invoked via [TPTBox](https://github.com/Hendrik-code/TPTBox)'s `run_vibeseg`.
2. **[SPINEPS](https://github.com/Hendrik-code/spineps)** → vertebra + spine segmentation; ribs are merged in via [TPTBox](https://github.com/Hendrik-code/TPTBox)'s `add_ribs_to_vert_spine` (`seg-vert-rib`, `seg-spine-rib`).
3. **`treg_fullbody.full_body_poi.run_all`** → per-region atlas registration (shoulder, hip, arm, leg, ribs, feet …) and writes landmark POI files to `derivatives-treg/` / `derivatives-final-points/`.

All stages are idempotent — existing output files are skipped on reruns.

### Dataset layout

The script expects a BIDS tree with a `rawdata/` folder containing CTs:

```
<root>/rawdata/sub-XYZ/sub-XYZ_ses-YYYY_sequ-N_ct.nii.gz
```

Derivatives land next to it under `derivatives/`, `derivatives-treg/`, and `derivatives-final-points/`.

### Usage

```bash
python example_inference.py <dataset_root> [options]
```

Options:

| Flag | Default | Description |
|---|---|---|
| `root` (positional) | `/DATA/NAS/datasets_processed/CT_fullbody/dataset-bonescreen-test2` | BIDS dataset root. |
| `--gpu N` | `0` | GPU index. |
| `--limit N` | *all* | Process only the first N CTs. |
| `--no-sort` | *sort on* | Iterate subjects in BIDS traversal order instead of sorted. |
| `--skip-spineps` | off | Skip the SPINEPS stage (expects `seg-vert-rib` + `seg-spine-rib` to already exist). |

Examples:

```bash
# Run on one subject to smoke-test the full pipeline
python example_inference.py /path/to/dataset --limit 1

# Already have vertebra/spine outputs, only refresh the POI stage
python example_inference.py /path/to/dataset --skip-spineps

# Pick a different GPU
python example_inference.py /path/to/dataset --gpu 1
```

### Outputs

For each `sub-XYZ_ses-YYYY_sequ-N_ct.nii.gz`:

- `derivatives/.../seg-VIBESeg-12_msk.nii.gz` — 12-label body segmentation.
- `derivatives/.../seg-vert-rib_msk.nii.gz`, `seg-spine-rib_msk.nii.gz` — vertebra & spine with ribs.
- `derivatives-treg/.../seg-treg-<region>_poi.json` + `.mrk.json` — per-region POI files (viewable in 3D Slicer).
- `derivatives-treg/.../seg-fov-<region>-<subreg>_msk.nii.gz` — registered subregion masks.
- `derivatives-final-points/.../seg-torso_poi.*`, `seg-leg_poi.*`, `seg-treg_msk.nii.gz` — merged final landmark set.

---

### Working with Landmark (`.mrk.json`) Files


* Landmark files can be created and opened in **3D Slicer**
### `poi.json`
* If saved in **Local Coordinates**, landmark positions correspond to **pixel indices**
* Local Indexing starts at **0**, so values may offset by one in soft ware like ITKSnap

