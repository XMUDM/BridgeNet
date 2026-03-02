Here is the English version of the README tailored for your BridgeNet project.

---

# BridgeNet

This repository is the official implementation of the paper **"BridgeNet: A Dual-Curvature Product Manifold Approach for Molecular Property Prediction"**.

## Introduction

BridgeNet is a novel architecture based on Graph Neural Networks (GNNs) designed to enhance the performance of molecular dynamics simulations and molecular property prediction. By introducing a dual-curvature product manifold, the model can more accurately capture the complex underlying geometric and topological structures within molecular graph data, achieving exceptional representation capabilities across multiple downstream tasks.

## Environment

It is recommended to use Anaconda or Miniconda to manage your virtual environment. The main dependencies are as follows:

* python >= 3.9.0
* pytorch >= 2.0.0
* torch_geometric >= 2.3.0
* rdkit >= 2022.09.1
* numpy == 1.23.5
* pandas == 1.5.3
* scipy == 1.11.4

## Datasets

This project is evaluated on several standard molecular datasets, comprising a total of **9 main tasks** (3 regression tasks and 6 classification tasks).

* The metrics for the regression tasks have been normalized and aligned during the data processing stage to facilitate a unified evaluation process.
* All molecular graph data and preprocessed features should be placed under the `./data/` directory.

## Quick Start

1. Clone this repository and install the environmental dependencies listed above.
2. Run the data preprocessing script to download the data to `./data/` and generate the manifold embeddings required by the algorithm:
```bash
python src/data/preprocess.py --dataset_dir ./data/

```


3. Configure your desired task type (classification or regression) in `./src/models/config.yaml`.
4. Run the following command to train and evaluate the model:

```bash
# run the main code for training
python -u main.py --model=BridgeNet --dataset=tox21 --task_type=classification --gpu_id=0 --epochs=100

```

## Baselines

To facilitate comparative experiments, evaluation scripts for the following state-of-the-art GNN baseline models are also integrated into the testing framework of this repository:

* **SimSGT**
* **GraphMVP**

You can quickly run the baseline models by modifying the `--model` parameter in the execution command:

```bash
python -u main.py --model=GraphMVP --dataset=freesolv --task_type=regression --gpu_id=0

```

---

**Would you like me to help draft the content for the `config.yaml` template, or expand further on the data preprocessing instructions?**
