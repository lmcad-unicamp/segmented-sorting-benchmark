# Segmented sorting benchmark (Dataset, Code, and Results)

This bundle contains the dataset and the code used to evaluate the six segmented sorting strategies in the manuscript ``An Evaluation of Segmented Sorting Strategies on GPUs''.
It also contains the results produced by the evaluation of these six strategies on seven GPUs.
The bundle is organized as follows:

* datasets: contain the dataset files.
  - parallel-computing.ds: dataset used in our experiments.
  - small.ds: small dataset to perform quick experiments with the infrastructure.

* libs: copy of third-party libraries used by our application.

* organized-results: the results produced by each GPU system and each dataset distribution.
  - each sub-directory contains the results for a given GPU. The results are organized by segment sizes distribution. Ex: equal => segments have the same size, -5.0 => segment sizes follow a power-law distribution with alpha = -5.0.

* run-exp.sh: script used to run our experiments. 

* systems: scripts used to set the environemnt in each GPU system. They are invoked by the run-exp.sh script.

* utils: tools used to gather information about the computing infrastructure.


## Executing the benchmark application

In order to execute the benchmark, one may run the run-exp.sh script. 
Example:

    ./run-exp.sh lmcad-1080


This command will invoke the "systems/config_environment-lmcad-1080.sh" script to setup the environment, build the src/segort-benchmark.x application, and execute it using the datasets/parallel-computing.ds dataset with different power-law distributions and equal segment sizes. 
