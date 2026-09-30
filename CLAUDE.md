# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Binary crack detection (crack / no crack) on concrete images from the SDNET2018 dataset, using a PyTorch ResNet18 fine-tuned from ImageNet weights. Flat collection of Python scripts, run in order, from the repo root. There is no build system, test suite, linter config, or package layout. `requirements.txt` is referenced in the README but does not exist yet (deps: torch, torchvision, pandas, numpy, scikit-learn, matplotlib, tqdm, Pillow).

## Pipeline (run from repo root; all paths are relative to cwd)

```bash
python 1_zip_file_extraction.py   # ./SDNET2018.zip -> ./SDNET2018/ (zip must be placed manually)
# 2_data_viewing.ipynb            # empty placeholder notebook
python 3_data_cleaning.py         # verify images, write metadata/sdnet2018_full.csv
python 4_data_optimization.py     # stratified 70/15/15 split -> metadata/{train,val,test}.csv
python model_training.py          # trains, saves best_model.pth, evaluates on test
```

The numeric prefixes are the execution order. `data_preprocessing.py` is not run directly; it is imported by `model_training.py`.

## Architecture

Data flows between stages through CSVs in `metadata/`, each with columns `image_path, label, concrete`:

- `3_data_cleaning.py` walks `SDNET2018/`, derives `label` from the class folder name (`C*` -> 1 cracked, otherwise 0) and `concrete` from the parent folder (`D`/`P`/`W`).
- `4_data_optimization.py` does the stratified split (seed 42).
- `data_preprocessing.py` runs at import time: it loads the split CSVs and builds transforms, `CrackDataset`, the three `DataLoader`s (batch 32, `num_workers=4`), inverse-frequency class weights, and the weighted `criterion`. `model_training.py` imports these as module-level globals, so importing it has side effects (reads CSVs, prints).
- `model_training.py` uses `resnet18(pretrained=True)` with `fc` replaced by a 2-class head, Adam at lr 1e-4, 5 epochs, and keeps the checkpoint with the best validation F1 in `best_model.pth`, then reloads it for test evaluation.

## Gotchas

- `SDNET2018/`, datasets, and `__pycache__/` are gitignored; `metadata/sdnet2018_full.csv` is committed and contains absolute-style paths from the original author's machine, so regenerate it with `3_data_cleaning.py` after extracting the data locally. The other split CSVs are not committed.
- The `model_training.py` docstring says `num_workers=0` for Windows, but the loaders are created in `data_preprocessing.py` with `num_workers=4`; change it there if multiprocessing fails.
- `pretrained=True` is deprecated in recent torchvision (use `weights=...`), and `torch.load` in newer torch may need `weights_only=True`.
- The README's inference snippet does `from model_training import model`, but `model` is local to `main()` and does not exist at module level; rebuild the ResNet18 and load `best_model.pth` instead.
- The README's project-structure section and the instruction to install from `requirements.txt` are aspirational in places (no such file).
