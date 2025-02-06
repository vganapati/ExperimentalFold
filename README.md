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

### OpenFold

Clone this [fork]() of OpenFold into your parent directory
The parent folder of the OpenFold repo should be the same as the parent folder of the SFCalculator repo.

```
cd $PARENT
git clone repo
```

See installation instructions [here]() for OpenFold on NERSC Perlmutter.

Also install matplotlib:
```
python -m pip install -U matplotlib
```

Create the following environment source file:


Following can be sourced with `source ~/env_openfold`:

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

### SFCalculator

Clone this [fork]() of SFCalculator into your parent directory

```
source ~/env_openfold # if not already in environment
pip install gemmi
pip install reciprocalspaceship
pip install tqdm
git clone https://github.com/vganapati/SFcalculator_torch.git
cd $PARENT/SFcalculator_torch
pip install .
```

Test installation:

```
cd $PARENT/openfold
python3 SFcalculator_loss.py
```

### Weights and Biases

Setup an account on weights and biases for experiment tracking.

```
wandb login
```
## Training
Documentation on [training OpenFold](https://openfold.readthedocs.io/en/latest/Training_OpenFold.html)

Recommended procedure for training is to start from fewer/shared nodes and scale up to more nodes.

```
export TOTAL_GPUS=1
export SLURM_NTASKS=$TOTAL_GPUS # needed for deepspeed

salloc --qos shared_interactive --time 03:00:00 --constraint gpu --gpus $TOTAL_GPUS --account=m4734_g
```

### Login Node

```
cd $PARENT/openfold
```

Command to run training from scratch on login node (note that you can use [pdb breakpoints]() for debugging, except in the data loading):
```
python3 train_openfold.py $TRAIN_DATA_DIR $DATA_DIR/alignment_data/alignments $TEMPLATE_MMCIF_DIR $OUTPUT_DIR 2021-10-10 --train_chain_data_cache_path $DATA_DIR/pdb_data/data_caches/chain_data_cache.json --template_release_dates_cache_path $DATA_DIR/pdb_data/data_caches/mmcif_cache.json --config_preset initial_training --seed 42 --obsolete_pdbs_file_path $DATA_DIR/pdb_data/obsolete.dat --num_nodes 1 --gpus 1 --precision bf16-mixed
```

Run this command for finetuning:

```
export CHECKPOINT_PATH=$CFS/m3562/users/vidyagan/openfold/openfold/resources/openfold_params/finetuning_5.pt
python3 train_openfold.py $TRAIN_DATA_DIR $DATA_DIR/alignment_data/alignments $TEMPLATE_MMCIF_DIR $OUTPUT_DIR 2021-10-10 --train_chain_data_cache_path $DATA_DIR/pdb_data/data_caches/chain_data_cache.json --template_release_dates_cache_path $DATA_DIR/pdb_data/data_caches/mmcif_cache.json --seed 42 --obsolete_pdbs_file_path $DATA_DIR/pdb_data/obsolete.dat --num_nodes 1 --gpus 1 --precision bf16-mixed --config_preset finetuning --resume_from_ckpt $CHECKPOINT_PATH --resume_model_weights_only True --use_experimental_loss True
```

Note: 
This is where the config params are: `openfold/openfold/config.py`

### Interactive Node

To test the training loop, use an interactive node.

#### Single Shared Interactive Node

single GPU
option for regular/high memory


```
export TOTAL_GPUS=1
export SLURM_NTASKS=$TOTAL_GPUS # needed for deepspeed

salloc --qos shared_interactive --time 01:30:00 --constraint gpu --gpus $TOTAL_GPUS --account=m4734_g
```

Run same commands as in login node instructions.

#### Single Interactive Node

4 GPUs, single node
option for regular/high memory

using deepspeed:

```
export NUM_NODES=1
export TOTAL_GPUS=$((${NUM_NODES}*4))
export SLURM_NTASKS=$TOTAL_GPUS # needed for deepspeed

salloc --nodes $NUM_NODES --qos interactive --time 01:00:00 --constraint gpu --gpus $TOTAL_GPUS --account=m4734_g

srun --ntasks-per-node=4 --gpus $TOTAL_GPUS --nodes $NUM_NODES python3 train_openfold.py $TRAIN_DATA_DIR $DATA_DIR/alignment_data/alignments $TEMPLATE_MMCIF_DIR $OUTPUT_DIR 2021-10-10 --train_chain_data_cache_path $DATA_DIR/pdb_data/data_caches/chain_data_cache.json --template_release_dates_cache_path $DATA_DIR/pdb_data/data_caches/mmcif_cache.json --config_preset initial_training --seed 42 --obsolete_pdbs_file_path $DATA_DIR/pdb_data/obsolete.dat --num_nodes 1 --gpus 1 --precision bf16-mixed --deepspeed_config deepspeed_config.json
```

#### Multiple Interactive Nodes

Now use multiple interactive nodes to test parallelization of the batch across GPUs and nodes
option for regular/high memory

```
Using interactive node for multi-GPU/multi-node training

export NUM_NODES=4
export TOTAL_GPUS=$((${NUM_NODES}*4))

salloc --nodes $NUM_NODES --qos interactive --time 04:00:00 --constraint gpu --gpus $TOTAL_GPUS --account=m4734_g

srun --ntasks-per-node=4 --gpus $TOTAL_GPUS --nodes $NUM_NODES python3 train_openfold.py $TRAIN_DATA_DIR $DATA_DIR/alignment_data/alignments $TEMPLATE_MMCIF_DIR $OUTPUT_DIR 2021-10-10 --train_chain_data_cache_path $DATA_DIR/pdb_data/data_caches/chain_data_cache.json --template_release_dates_cache_path $DATA_DIR/pdb_data/data_caches/mmcif_cache.json --config_preset initial_training --seed 42 --obsolete_pdbs_file_path $DATA_DIR/pdb_data/obsolete.dat --num_nodes $NUM_NODES --gpus 4 --precision bf16-mixed

finetuning:

export CHECKPOINT_PATH=$PARENT/openfold/openfold/resources/openfold_params/finetuning_5.pt
export EXPERIMENT_NAME=test1
mkdir -p ${OUTPUT_DIR}/${EXPERIMENT_NAME}

srun --ntasks-per-node=4 --gpus $TOTAL_GPUS --nodes $NUM_NODES python3 train_openfold.py $TRAIN_DATA_DIR $DATA_DIR/alignment_data/alignments $TEMPLATE_MMCIF_DIR ${OUTPUT_DIR}/${EXPERIMENT_NAME} 2021-10-10 --train_chain_data_cache_path $DATA_DIR/pdb_data/data_caches/chain_data_cache.json --template_release_dates_cache_path $DATA_DIR/pdb_data/data_caches/mmcif_cache.json --config_preset finetuning --seed 42 --obsolete_pdbs_file_path $DATA_DIR/pdb_data/obsolete.dat --num_nodes $NUM_NODES --gpus 4 --precision bf16-mixed --resume_from_ckpt $CHECKPOINT_PATH --resume_model_weights_only True --use_experimental_loss True --log_performance False --wandb --experiment_name ${EXPERIMENT_NAME} --wandb_id None --wandb_project ExperimentalFold --wandb_entity $WANDB_ENTITY --log_every_n_steps 1 --log_lr --checkpoint_every_epoch --max_epochs 10 --train_epoch_len 20



Finetuning from saved run:

export CHECKPOINT_PATH=$PARENT/openfold_output/test1/ExperimentalFold/None/checkpoints/18-38.ckpt
export EXPERIMENT_NAME=test1
export WANDB_ID=id_1
mkdir -p ${OUTPUT_DIR}/${EXPERIMENT_NAME}

srun --ntasks-per-node=4 --gpus $TOTAL_GPUS --nodes $NUM_NODES python3 train_openfold.py $TRAIN_DATA_DIR $DATA_DIR/alignment_data/alignments $TEMPLATE_MMCIF_DIR ${OUTPUT_DIR}/${EXPERIMENT_NAME} 2021-10-10 --train_chain_data_cache_path $DATA_DIR/pdb_data/data_caches/chain_data_cache.json --template_release_dates_cache_path $DATA_DIR/pdb_data/data_caches/mmcif_cache.json --config_preset finetuning --seed 42 --obsolete_pdbs_file_path $DATA_DIR/pdb_data/obsolete.dat --num_nodes $NUM_NODES --gpus 4 --precision bf16-mixed --resume_from_ckpt $CHECKPOINT_PATH --resume_model_weights_only False --use_experimental_loss True --log_performance False --wandb --experiment_name ${EXPERIMENT_NAME} --wandb_id $WANDB_ID --wandb_project ExperimentalFold --wandb_entity $WANDD_ID --log_every_n_steps 1 --log_lr --checkpoint_every_epoch --max_epochs 25 --train_epoch_len 20

[NOTE: Code adds to previous training run on wandb server, but has a separate folder in the output with the training run logs. The ckpts are added to the main folder]
[NOTE: Checkpoints are saved as epoch#-global_step#.ckpt]


Try with "LATEST" checkpoint:
export CHECKPOINT_PATH=LATEST

Try command above.

```

### Compute Nodes

High memory
option for regular/high memory

```
source ~/env_openfold # if starting from fresh terminal

export NUM_NODES=10
export NUM_SUBMISSIONS=3
export NERSC_GPU_ALLOCATION=m4734_g
export WANDB_ENTITY=wandb_entity
export TIME=3:00:00
export CHECKPOINT_PATH=$CFS/m3562/users/vidyagan/openfold/openfold/resources/openfold_params/finetuning_5.pt 

cd $PARENT/openfold

. scripts/slurm_scripts/main_perlmutter.sh $NUM_NODES $NUM_SUBMISSIONS $NERSC_GPU_ALLOCATION $WANDB_ENTITY $TIME $CHECKPOINT_PATH
```

Results saved in `PARENT/openfold_output`
### Known issues

Memory errors on certain proteins during training. 
TODO log by rank to pinpoint the exact errors.
Check on validation data

## Experiments

TODO
try different weighting of new loss/old loss

## Experiment Tracking

Weights and Biases 

## Evaluation

The code trains only with R_{work} structure factors. Can evaluate on the following metrics in addition to the existing OpenFold metrics:
- R_{free} on training or validation datasets
- R_{work} on validation and test datasets
- R_{free} on test dataset

Check on test data
TODO: script for evaluation
Check OpenFold documentation

TODO
R_{free} and R_{work}

## Issues

Upgrade to most recent version of OpenFold
Upgrade to most recent version of SFCalculator
Move to AlphaFold 3 as opensource versions in PyTorch with a training loop become available.
Switch to NVIDIA dataset
PDB-Redo
Improve solvent model in SFCalculator to match Phenix solvent model
Dimer/Trimer: right now taking the structure of a single instance and using coordinates of everything else
Switch to Multimer