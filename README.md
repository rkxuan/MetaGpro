## MetaGpro

This is the official implementation of "Prompt as a Double-Edged Sword: A Dynamic Equilibrium Gradient-Assigned Attack against Graph Prompt Learning"

We will further fix the bugs, update the more robustness tests of GPL scenarios in the future.

We refer to the <ins>DeepRobust</ins> library to build our attack baselines **src/**. [https://github.com/DSE-MSU/DeepRobust](https://github.com/DSE-MSU/DeepRobust).

We refer to the <ins>ProG</ins> library to build our GPL scenarios **src/**. [https://github.com/sheldonresearch/ProG](https://github.com/sheldonresearch/ProG).


## Basic Environment
* `CUDA == 12.1`
* `python == 3.10` 
* `PyTorch == 2.1.0`


## Dataset

We use seven widely used benchmark datasets: Cora, CoraFull, CiteSeer, PubMed, Computers, Photo, DBLP

## Code Structure

main/run.py: The definition of main() function to run the experiments

prompt_graph/attack: Various meta-gradient attacks and our MetaGpro

prompt_graph/data: The graphic datasets and how to load graph as pytorch tensor

prompt_graph/defense: The defense strategy (i.e. PruneG)

prompt_graph/evaluation: The evaluator for GPL tasks

prompt_graph/model: The graph encoders

prompt_graph/pretrain: Pretraining models

prompt_graph/prompt: Graph prompt paradigms

prompt_graph/tasker: GPL tasks

prompt_graph/utils: Other needed functions or classes

## Implementation  

Mettack: !python main/run.py --pretrain_type=GraphMAE --gnn_type=GraphTransformer --pretrain_dataset=PubMed --target_dataset=PubMed --Mettack_type=mettack

MetaGpro: !python main/run.py --pretrain_type=GraphMAE --gnn_type=GraphTransformer --pretrain_dataset=PubMed --target_dataset=PubMed --Mettack_type=MetaGpro --surrogate_prompt=GPF
