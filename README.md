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

## Data Preprocessing
The O'Neil and NCI-ALMANAC drug synergy datasets were downloaded from the [DrugComb database](https://drugcomb.fimm.fi/) and removed outliers. The detailed processes of data preprocessing are as follows taking O'Neil for regression task as an example:
1. Go to `./Data/O'Neil` folder.
2. Run 01-String_to_Int.ipynb, convert string to integers in the data set, which means encoding drugs and cell lines.
3. Run 02-Joint_SuperEdge_reg.ipynb, construct joint graph using Super Edge method.
4. The files in ./Data/O'Neil/Preprocessed/reg are the input file of the JointSyn for regression task.

## Running the model
```bash
# for regression task
cd Model/JointSyn_reg
python main.py
# for classification task
cd Model/JointSyn_cls
python main.py
```
### Train
Train the model with the dataset in /rawData. Set split_flag=1, train_flag=1 and test_flag=1 in main.py. Put the preprocessed data into ./Model/JointSyn_reg/rawData folder.

### Test
Use the saved weights predict the novel drug-drug-cell line. Set split_flag=0, train_flag=0 and test_flag=1 in main.py. Put the saved weights into ./Model/JointSyn_reg/save folder.




