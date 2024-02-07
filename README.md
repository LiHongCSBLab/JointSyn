# JointSyn

This repository contains the code and data for "Dual-view jointly learning improves personalized drug synergy prediction"

## Requirements

- python 3.9.12
- numpy >= 1.23.5
- pandas >= 1.5.2
- pytorch >= 1.13.1
- torchvision >= 0.14.1
- dgl >= 0.9.0
- scikit-learn >= 1.3.2
- scipy >= 1.10.1
- rdkit >= 2022.03.2
- networkx >= 2.8.6

## Usage
```bash
# for regression task
cd Model/JointSyn_reg
python main.py
# for classification task
cd Model/JointSyn_cls
python main_reg.py
```
