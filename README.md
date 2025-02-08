# ExperimentalFold

Retraining OpenFold with structure factor amplitudes.

## Overview

AlphaFold 2 (corresponding open source version OpenFold), is trained on the protein structures that consist of a list of atomic coordinates from the Protein Data Bank. To obtain the atomic coordinates from crystallography experiments, the crystallographer must analyze raw collected data with tools like the Computational Crystallography Toolbox (to get structure factor amplitudes) and Phenix (to phase the structure factor amplitudes and build an atomic model from the resulting electron density). There is a potential for inaccuracies in model building, where the final atomic coordinates may not perfectly represent the true model. This code trains or finetunes OpenFold directly with the structure factor amplitudes and sees if performance can be improved.

We modify and incorporate code from 2 GitHub repos, OpenFold and SFCalculator. The code works by having OpenFold predict the structure of a monomer corresponding to a PDB entry. The monomer is rotated by the Kabsch algorithm to the same orientation of the protein in the PDB entry. Any atoms not in the monomer but in the PDB entry (e.g. ligands, other proteins) are taken directly from the PDB entry. The corresponding MTZ file containing structure factor amplitudes is found, and the unit cell is extracted. From the atom coordinates, the unit cell, and assumed solvent model, the structure factor amplitudes are computed from SFCalculator. We compute the negative log likelihood of these computed structure factor amplitudes given the experimentally derived structure factor amplitude mean and variance. The negative log likelihood is added to the OpenFold loss. 

This code has been tested only on monomers with OpenFold/AlphaFold 2 but can be straightforwardly extended to AlphaFold-multimer or AlphaFold 3.

## Setup & Installation

Directions are specifically given for installing and running on NERSC Perlmutter.

```
export PARENT=parent-directory
cd $PARENT
```

NERSC Account (might be expired) for saved folders
Active NERSC Account for running experiments

### OpenFold

Clone this [fork](https://github.com/vganapati/openfold) of OpenFold into your parent directory.

```
cd $PARENT
git clone repo
```

See installation instructions [here](https://github.com/vganapati/openfold/blob/pl_upgrades/INSTALLATION.md) to install OpenFold on NERSC Perlmutter.

Create the following environment source file: 
```
cd
vi env_openfold

module load conda
cd $PARENT/openfold
conda activate $PARENT/openfold_env
export LIBRARY_PATH=$CONDA_PREFIX/lib:$LIBRARY_PATH
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

export SF_CALC_PATH=$PARENT/SFcalculator_torch
export PYTHONPATH=$SF_CALC_PATH:$PYTHONPATH


export DATA_DIR=$PARENT/openfold_training
export TEMPLATE_MMCIF_DIR=$DATA_DIR/pdb_data/mmcif_files
export TRAIN_DATA_DIR=$TEMPLATE_MMCIF_DIR
export OUTPUT_DIR=$PARENT/openfold_output
```

Activate environment and install matplotlib:
```
source ~/env_openfold
python -m pip install -U matplotlib
```

### SFCalculator

Clone this [fork](https://github.com/vganapati/SFcalculator_torch) of SFcalculator_torch into your parent directory.

```
source ~/env_openfold # if not already in environment
cd $PARENT
python -m pip install gemmi
python -m pip install reciprocalspaceship
python -m pip install tqdm
git clone https://github.com/vganapati/SFcalculator_torch.git
cd $PARENT/SFcalculator_torch
python -m pip install .
```

Test installation:
```
cd $PARENT/openfold
python3 SFcalculator_loss.py
```

### Weights & Biases

Setup an account on [Weights & Biases](https://wandb.ai/site/) for experiment tracking.
```
wandb login
```

## Training

Official documentation on [training OpenFold](https://openfold.readthedocs.io/en/latest/Training_OpenFold.html).

Recommended procedure for training is to start from fewer/shared nodes and scale up to more nodes.

### Login Node

Start by trying to run code on the login node.
```
cd $PARENT/openfold
```

Command to run training from scratch on login node (note that you can use [pdb breakpoints](https://realpython.com/python-debugging-pdb/) for debugging, except in the data loader):
```
python3 train_openfold.py $TRAIN_DATA_DIR $DATA_DIR/alignment_data/alignments $TEMPLATE_MMCIF_DIR $OUTPUT_DIR 2021-10-10 --train_chain_data_cache_path $DATA_DIR/pdb_data/data_caches/chain_data_cache.json --template_release_dates_cache_path $DATA_DIR/pdb_data/data_caches/mmcif_cache.json --config_preset initial_training --seed 42 --obsolete_pdbs_file_path $DATA_DIR/pdb_data/obsolete.dat --num_nodes 1 --gpus 1 --precision bf16-mixed
```

Run this command for finetuning:
```
export CHECKPOINT_PATH=$CFS/m3562/users/vidyagan/openfold/openfold/resources/openfold_params/finetuning_5.pt
python3 train_openfold.py $TRAIN_DATA_DIR $DATA_DIR/alignment_data/alignments $TEMPLATE_MMCIF_DIR $OUTPUT_DIR 2021-10-10 --train_chain_data_cache_path $DATA_DIR/pdb_data/data_caches/chain_data_cache.json --template_release_dates_cache_path $DATA_DIR/pdb_data/data_caches/mmcif_cache.json --seed 42 --obsolete_pdbs_file_path $DATA_DIR/pdb_data/obsolete.dat --num_nodes 1 --gpus 1 --precision bf16-mixed --config_preset finetuning --resume_from_ckpt $CHECKPOINT_PATH --resume_model_weights_only True --use_experimental_loss True
```

Note: Config parameters are located in `openfold/openfold/config.py`.

### Interactive Node

To test the training loop, use an interactive node.

#### Single Shared Interactive Node

Set environment variables:
```
export TOTAL_GPUS=1
export SLURM_NTASKS=$TOTAL_GPUS # needed for deepspeed
```

Can use a regular GPU:
```
salloc --qos shared_interactive --time 03:00:00 --constraint gpu --gpus $TOTAL_GPUS --account=${PROJECT}_g
```

Or can use a high memory GPU:
```
salloc --qos shared_interactive --time 03:00:00 --constraint gpu&hbm80g --gpus $TOTAL_GPUS --account=${PROJECT}_g
```

Run same training/finetuning commands as in login node instructions.

#### Single Interactive Node

Training using deepspeed (can modify to finetuning, see for reference the finetuning command under the login node instructions):
```
export NUM_NODES=1
export TOTAL_GPUS=$((${NUM_NODES}*4))
export SLURM_NTASKS=$TOTAL_GPUS # needed for deepspeed

# for high memory GPUs, use `--constraint gpu&hbm80g`
salloc --nodes $NUM_NODES --qos interactive --time 01:00:00 --constraint gpu --gpus $TOTAL_GPUS --account=${PROJECT}_g

srun --ntasks-per-node=4 --gpus $TOTAL_GPUS --nodes $NUM_NODES python3 train_openfold.py $TRAIN_DATA_DIR $DATA_DIR/alignment_data/alignments $TEMPLATE_MMCIF_DIR $OUTPUT_DIR 2021-10-10 --train_chain_data_cache_path $DATA_DIR/pdb_data/data_caches/chain_data_cache.json --template_release_dates_cache_path $DATA_DIR/pdb_data/data_caches/mmcif_cache.json --config_preset initial_training --seed 42 --obsolete_pdbs_file_path $DATA_DIR/pdb_data/obsolete.dat --num_nodes 1 --gpus 1 --precision bf16-mixed --deepspeed_config deepspeed_config.json
```

#### Multiple Interactive Nodes

Now use multiple interactive nodes to test parallelization of the batch across GPUs and nodes.
```
export NUM_NODES=4
export TOTAL_GPUS=$((${NUM_NODES}*4))

# for high memory GPUs, use `--constraint gpu&hbm80g`
salloc --nodes $NUM_NODES --qos interactive --time 04:00:00 --constraint gpu --gpus $TOTAL_GPUS --account=${PROJECT}_g

srun --ntasks-per-node=4 --gpus $TOTAL_GPUS --nodes $NUM_NODES python3 train_openfold.py $TRAIN_DATA_DIR $DATA_DIR/alignment_data/alignments $TEMPLATE_MMCIF_DIR $OUTPUT_DIR 2021-10-10 --train_chain_data_cache_path $DATA_DIR/pdb_data/data_caches/chain_data_cache.json --template_release_dates_cache_path $DATA_DIR/pdb_data/data_caches/mmcif_cache.json --config_preset initial_training --seed 42 --obsolete_pdbs_file_path $DATA_DIR/pdb_data/obsolete.dat --num_nodes $NUM_NODES --gpus 4 --precision bf16-mixed
```

For finetuning:
```
export CHECKPOINT_PATH=$PARENT/openfold/openfold/resources/openfold_params/finetuning_5.pt
export EXPERIMENT_NAME=test1
mkdir -p ${OUTPUT_DIR}/${EXPERIMENT_NAME}

srun --ntasks-per-node=4 --gpus $TOTAL_GPUS --nodes $NUM_NODES python3 train_openfold.py $TRAIN_DATA_DIR $DATA_DIR/alignment_data/alignments $TEMPLATE_MMCIF_DIR ${OUTPUT_DIR}/${EXPERIMENT_NAME} 2021-10-10 --train_chain_data_cache_path $DATA_DIR/pdb_data/data_caches/chain_data_cache.json --template_release_dates_cache_path $DATA_DIR/pdb_data/data_caches/mmcif_cache.json --config_preset finetuning --seed 42 --obsolete_pdbs_file_path $DATA_DIR/pdb_data/obsolete.dat --num_nodes $NUM_NODES --gpus 4 --precision bf16-mixed --resume_from_ckpt $CHECKPOINT_PATH --resume_model_weights_only True --use_experimental_loss True --log_performance False --wandb --experiment_name ${EXPERIMENT_NAME} --wandb_id None --wandb_project ExperimentalFold --wandb_entity $WANDB_ENTITY --log_every_n_steps 1 --log_lr --checkpoint_every_epoch --max_epochs 10 --train_epoch_len 20
```


Finetuning from saved run:
```
export CHECKPOINT_PATH=$PARENT/openfold_output/test1/ExperimentalFold/None/checkpoints/18-38.ckpt
# export CHECKPOINT_PATH=LATEST
export EXPERIMENT_NAME=test1
export WANDB_ID=id_1
mkdir -p ${OUTPUT_DIR}/${EXPERIMENT_NAME}

srun --ntasks-per-node=4 --gpus $TOTAL_GPUS --nodes $NUM_NODES python3 train_openfold.py $TRAIN_DATA_DIR $DATA_DIR/alignment_data/alignments $TEMPLATE_MMCIF_DIR ${OUTPUT_DIR}/${EXPERIMENT_NAME} 2021-10-10 --train_chain_data_cache_path $DATA_DIR/pdb_data/data_caches/chain_data_cache.json --template_release_dates_cache_path $DATA_DIR/pdb_data/data_caches/mmcif_cache.json --config_preset finetuning --seed 42 --obsolete_pdbs_file_path $DATA_DIR/pdb_data/obsolete.dat --num_nodes $NUM_NODES --gpus 4 --precision bf16-mixed --resume_from_ckpt $CHECKPOINT_PATH --resume_model_weights_only False --use_experimental_loss True --log_performance False --wandb --experiment_name ${EXPERIMENT_NAME} --wandb_id $WANDB_ID --wandb_project ExperimentalFold --wandb_entity $WANDD_ID --log_every_n_steps 1 --log_lr --checkpoint_every_epoch --max_epochs 25 --train_epoch_len 20
```

Notes:
- Code adds to previous training run on wandb server, but has a separate folder in the output with the training run logs. The ckpts are added to the main folder
- Checkpoints are saved as epoch#-global_step#.ckpt

### Compute Nodes

To submit jobs:
```
source ~/env_openfold # if starting from fresh terminal

export NUM_NODES=10
export NUM_SUBMISSIONS=3
export NERSC_GPU_ALLOCATION=${PROJECT}_g
export WANDB_ENTITY=wandb_entity
export TIME=3:00:00
export CHECKPOINT_PATH=$CFS/m3562/users/vidyagan/openfold/openfold/resources/openfold_params/finetuning_5.pt 

cd $PARENT/openfold

. scripts/slurm_scripts/main_perlmutter.sh $NUM_NODES $NUM_SUBMISSIONS $NERSC_GPU_ALLOCATION $WANDB_ENTITY $TIME $CHECKPOINT_PATH
```

Results saved in `${PARENT}/openfold_output`.

### Known issues in training

- Memory errors on certain proteins during training. 
- Need to log by rank to pinpoint the exact errors.

## Experiments

The structure factor negative log loss is added to the existing OpenFold losses. 

TODO: Different relative weightings of the losses can be tested. 

## Evaluation

The code trains only with $R_{work}$ structure factors. 

TODO: evaluate on the following metrics in addition to the existing OpenFold metrics:
- $R_{free}$ on training or validation datasets
- $R_{work}$ on validation and test datasets
- $R_{free}$ on test dataset

## TODOs

- Upgrade to most recent version of OpenFold
- Upgrade to most recent version of SFCalculator
- Move to AlphaFold 3 as opensource versions in PyTorch with a training loop become available.
- Retrain with plinder/pinder datasets
- Retrain with PDB-Redo
- Improve solvent model in SFCalculator to match Phenix solvent model
- Dimer/Trimer: right now taking replacing a single structure with a prediction and using ground truth coordinates of everything else, switch to using the prediction for all instances of the structure