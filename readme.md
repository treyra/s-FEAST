# Online Tree-based Planning for Active Spacecraft Fault Estimation and Collision Avoidance

James Ragan[^maintainer],
Benjamin Riviere,
Fred Y. Hadaegh, and
Soon-Jo Chung[^corresponding]

[^maintainer]: code base maintainer - jragan@caltech.edu
[^corresponding]: corresponding author - sjchung@caltech.edu

The failurePy codebase is provided as part of our Science Robotics research article "[Online tree-based planning for active spacecraft fault estimation and collision avoidance](https://www.science.org/doi/10.1126/scirobotics.adn4722)". It contains code for simulating spacecraft models subject various faults and algorithms for diagnosing them. Our safe Fault Estimation via Active Sensing Tree search (s-FEAST) algorithm provides a method to actively estimate these faults, even when subject to safety constraints that will be violated shortly.

## Dependencies

failurePy's depends on numpy, matplotlib, pyyaml, tqdm, cvxpy and opencv-python. It also depends on Google's JAX project, which can be installed here https://github.com/google/jax#installation (the CPU version is sufficient). Linux and WSL2 in Windows are tested and supported. MacOS is untested, but should work.

## Examples
To run an example of our binary or general fault models, run `python pipeline.py ./config/binaryFaultModelExample.yaml` or `python pipeline.py ./config/generalFaultModelExample.yaml` in the failurePy directory. These examples demonstrate how our tree search algorithm works, and the configuration file provides an example for how to adapt failurePy to run other simulations.


## Data

Output data is stored in the `failurePy/SavedData/` folder by default. Data used to generate the paper figures can be found this Dryad depository https://doi.org/10.5061/dryad.xgxd254r1. All data is stored in the form of:

    experimentName/solver/nSimulationsPerTree/trialNumber/trialData.dict
    experimentName/solver/nSimulationsPerTree/trialNumber/trialData.txt
    experimentName/solver/nSimulationsPerTree/averageData.dict
    experimentName/config.yaml
    experimentName/render.pdf
    experimentName/version.txt

Where there can be multiple solvers per experimentName, multiple nSimulationsPerTrees per solver, and multiple trialNumbers per nSimulationsPerTree. The data in `trialData.dict` and `trialData.txt` are the same, but are machine vs. human readable. Each `.dict` file is a python dictionary with the following keys:

| `field` | description |
|---------|-------------|
| `'physicalStateList'` | state of the system at each time step |
| `'failureStateList'` | constant failure state at each time step |
| `'beliefList'` | tuple of weights on each fault and an 3D array representing each conditional filter on these faults |
| `'rewards'` | rewards at each time step |
| `'actionList'` | action taken to arrive at each time step |
| `'possibleFailures'` | the faults considered in this experiment, known to the estimator and solver |
| `'success'` | Whether the algorithm converged to the correct fault (and stayed safe) |
| `'steps'` | number of time steps taken. Relevant if the simulation can end early |
| `'wallClockTime'` | average time to select each action |
| `'treeList'` | if saved, the search tree for each time step. Not present in runs with more than one trial per solver and nSimulationsPerTree|

The data in `averageData.dict` is also a python dictionary and represents the average over all trials for the following parameters:

| `field` | description |
|---------|-------------|
| `'avgRewards'` | array of the average rewards at each time step |
| `'cumulativeAvgRewards'` | array of the average cumulative rewards at each time step  |
| `'avgSuccessRate'` | average success rate |
| `'avgWallClockTime'` | average time in all trials to select each action |
| `'avgSteps'` | average steps taken |
| `'varRewards'` | array of the variance in the rewards at each time step |
| `'cumulativeVarRewards'` | array of the variance in the cumulative rewards at each time step |
| `'varWallClockTime'` |  the variance in the average time taken to select each action |

The `config.yaml` file provides the parameters used to run this experiment. Combined with `version.txt`, which logs when the experiment was run, the version of failurePy used, and a hash of the estimator functions used, this allows for repeatability. Finally, the `render.pdf` file renders the first trial run in each experiment for visualization. Other visualizations can be made as follows:

The code used to create Fig.5 and Fig. S3, is provided in `failurePy/visualization/figure5andFigureS3.ipynb`. Data is stored in SavedData/figure5 and SavedData/figureS3 in the Dryad depository.

The overlays shown in Figs. 1, S4, S5, and S6 are created by using `failurePy/visualization/renderSavedRun.py` to render the experimental data, which is also uploaded to the data repository. The same module can be used to re-create Fig. 6 from the raw simulation data provided. Data is stored in SavedData/figure1 in the Dryad depository, the other figures use the same data.

Figs. S1 and S2 can be reproduced from the provided data using `failurePy/visualization/figuresS1andS2.ipynb`. Data is stored in SavedData/figure1, SavedData/figureS1, SavedData/figureS2 in the Dryad depository.

The visualization directory contains other utilities for visualizing the data that were not used in our paper.

## Videos

We have made an overview video that can be seen here:

[![Full overview video](https://img.youtube.com/vi/olPOyCbhWG4/mqdefault.jpg)](https://youtu.be/olPOyCbhWG4 "Online Tree-based Planning for Active Spacecraft Fault Estimation and Collision Avoidance")


## Citation

The data and code here are for personal and educational use only and provided without warranty; written permission from the authors is required for further use. Please cite our work as follows:

> @article{
doi:10.1126/scirobotics.adn4722,
author = {James Ragan  and Benjamin Riviere  and Fred Y. Hadaegh  and Soon-Jo Chung },
title = {Online tree-based planning for active spacecraft fault estimation and collision avoidance},
journal = {Science Robotics},
volume = {9},
number = {93},
pages = {eadn4722},
year = {2024},
doi = {10.1126/scirobotics.adn4722},
URL = {https://www.science.org/doi/abs/10.1126/scirobotics.adn4722},
eprint = {https://www.science.org/doi/pdf/10.1126/scirobotics.adn4722}}
