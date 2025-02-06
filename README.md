# ExperimentalFold

Retraining OpenFold with structure factor amplitudes.

## Overview

AlphaFold 2 (corresponding open source version OpenFold), is trained on the protein structures that consist of a list of atomic coordinates from the Protein Data Bank. To obtain the atomic coordinates from crystallography experiments, the crystallographer must analyze raw collected data with tools like the Computational Crystallography Toolbox (to get structure factor amplitudes) and Phenix (to phase the structure factor amplitudes and build an atomic model from the resulting electron density). There is a potential for inaccuracies in model building, where the final atomic coordinates may not perfectly represent the true model. This code trains or finetunes OpenFold directly with the structure factor amplitudes and sees if performance can be improved.

We modify and incorporate code from 2 GitHub repos, OpenFold and SFCalculator. The code works by having OpenFold predict the structure of a monomer corresponding to a PDB entry. The monomer is rotated by the Kabsch algorithm to the same orientation of the protein in the PDB entry. Any atoms not in the monomer but in the PDB entry (e.g. ligands, other proteins) are taken directly from the PDB entry. The corresponding MTZ file containing structure factor amplitudes is found, and the unit cell is extracted. From the atom coordinates, the unit cell, and assumed solvent model, the structure factor amplitudes are computed from SFCalculator. We compute the negative log likelihood of these computed structure factor amplitudes given the experimentally derived structure factor amplitude mean and variance. The negative log likelihood is added to the OpenFold loss. 

This code has been tested only on monomers with OpenFold/AlphaFold 2 but can be straightforwardly extended to AlphaFold-multimer or AlphaFold 3.

## Setup & Installation

### SFCalculator

Clone this [fork]() of SFCalculator

### OpenFold

Clone this [fork]() of OpenFold
The parent folder of the OpenFold repo should be the same as the parent folder of the SFCalculator repo.
See installation instructions [here]() for OpenFold on NERSC Perlmutter.

### Weights and Biases

Setup an account on weights and biases for experiment tracking.

## Training

Recommended procedure for training is to start from fewer/shared nodes and scale up to more nodes.

### Interactive Node

To test the training loop, use an interactive node.

#### Single Shared Interactive Node

single GPU
option for regular/high memory

#### Single Interactive Node

4 GPUs, single node
option for regular/high memory

#### Multiple Interactive Nodes

Now use multiple interactive nodes to test parallelization of the batch across GPUs and nodes
option for regular/high memory

### Compute Nodes

High memory
option for regular/high memory

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


The code trains only with R_{work} structure factors. Can evaluate on the following metrics:
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