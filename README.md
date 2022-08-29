# TOPOAUC
An implement of the ACM MM 22 paper: A Unified Framework against Topology and Class Imbalance.

## Environments
* **Python** 3.7.11
* **Pytorch** 1.11
* **torch-geometric** 2.0.4
* **CUDA** 11.3

## Data
When running `train.py` for the first time, the dataset (CORA, CiteSeer, PubMed) will be automatically downloaded to ./datasets/[dataset] by torch_geometric.

## Training
1. Modify the config file `config.py` (copy the parameters from `./best_params/layers3/[dataset]/search_space_imb_losses_[loss type]_[class imbalance ratio].json`)
2. Run the script:
```shell
CUDA_VISIBLE_DEVICES=0  python train.py --loss ExpGAUC --pair_ner_diff 1 --imb_ratio 10.0
```
Note: ["ExpGAUC","HingeGAUC","SqGAUC"] are three kinds of AUC losses,  `--pair_ner_diff` decides whether to use our TAIL mechanism, `--imb_ratio` controls the class imbalance ratio which could be selected from [10.0,15.0,20.0].

