# PCGNet
Pyramid-Context Guided Feature Fusion for RGB-D Semantic Segmentation

## Setup
**Requisites:** torch, torchvision, tensorboard, numpy, scipy, tqdm, addict. \
Prepare NYUDv2 and SUN-RGBD datasets:
```
# dataset dir: /[root_path]/dataset
# [links & scripts will be updated soon]
```
Clone the project repo:
```
# project dir: /[root_path]/PCGNet
git clone https://github.com/hmdliu/PCGNet
cd PCGNet
```

## Training
1) On a HPC with a singlularity env and slurm:
```
# remember to modify the path in the slurm script
sbatch train.slurm [dataset_name] [exp_id]
```
2) On a computer with GPU:
```
# remember to activate the env
python train.py [dataset_root] [dataset_name] [exp_id] > [log_name].log 2>&1
```

## Model setting
See model config template under *config.py*
```
# ['exp_id's will be updated soon]
```

## Util functions
```
# Check best pred of multiple experiments
python helper.py dump

# Archive training logs
python helper.py log move
```


