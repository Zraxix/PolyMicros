# PolyMicros: Bootstrapping a Foundation Model for Polycrystalline Material Structure

Currently the codebase is complete but is WIP, and likely won't run on your system due to path names specific to my computer in the code (namely loading datasets, and the GRFUtils class). Datasets are too large to be shared here. Please contact Michael Buzzy for assistance. 

### TO-DO
1) Standardize function interface (no more swapping between channel first and channel last conventions).
2) Improve import structure such that this can be run on other machines.
3) Eliminate diffusers package requirement (we are just using one function. randn_tensor, for no reason at this point).
4) Retrain neighboorhood models with EMA (Current weights won't run on new codebase without tweaking, easier to retrain)
5) Package utility functions so they can be imported easier. 
6) fix dataset schema to be consistent across files

## This repositiory contains the PolyMicros code base. The code is broken up into several main sections. 

### Diffusion Code

diffusion/     - Contains a pytorch lightning module for running EDM style diffusion models

DMRunner.py    - A runner script for training and running inference on diffusion models

Unet.py        - ddpm style unet modified for 3D data

UnetP.py       - ddpm style unet modified for 3D data with perodic boundary conditions

### Multi-Output Gaussain Random Field Code

MOSMKernel.py  - Code allowing for the generation of auto- and cross- correlations using the MOSM kernel

GRFSampler     - Code for sampling from the MOGRF given a covariance

### Utility code

CUBEPlot.py    - Code for plotting 3D cubes and othogonal 2PS in matplotlib

MKFIGS.py      - Code to plot results for the paper (a mess, stay awyay)

HDF5PartialLoader.py  - A dataloader which loads datasets from HDF5 files one batch at a time (for when cpu memory is to small for the whole dataset)

ROGSHUtils.py  - Tools for handling managing ROGSH values and their conversion to euler angles and vice versa. 


### Workflow code

PolyMicros.py  - The main dataset generation code. Contains the MOGRF Sampling, and local refinment steps.

CaseStudies.py - Conditioning the model for Super-Resolution, and Dimensionality Expansion. Contains code for both case studies. 


