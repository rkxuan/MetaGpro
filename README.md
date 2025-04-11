## MetaGpro

This is the official implementation of "Prompt as a Double-Edged Sword: A Dynamic Equilibrium Gradient-Assigned Attack against Graph Prompt Learning"

We will further fix the bugs, update the more robustness tests of GPL scenarios in the future.

We refer to the <ins>DeepRobust</ins> library to build our attack baselines **src/**. [https://github.com/DSE-MSU/DeepRobust](https://github.com/DSE-MSU/DeepRobust).

We refer to the <ins>ProG</ins> library to build our GPL scenarios **src/**. [https://github.com/sheldonresearch/ProG](https://github.com/sheldonresearch/ProG).

## Dataset

We use seven widely used benchmark datasets: Cora, CoraFull, CiteSeer, PubMed, Computers, Photo, DBLP

## Structure

prompt_graph/attack: various meta-gradient attacks and out MetaGpro
prompt_graph/data: the dataset and how to load graph as tensor
prompt_graph/defense: the defense strategy (i.e. PruneG)
prompt_graph/evaluation: the evaluator
prompt_graph/model: the graph encoders
prompt_graph/pretrain: pretraining models
prompt_graph/prompt: graph prompt paradigms
prompt_graph/tasker: GPL tasks
prompt_graph/utils: other needed functions or classes

## Implementation  

mettack: !python main/run.py --pretrain_type=GraphMAE --gnn_type=GraphTransformer --pretrain_dataset=PubMed --target_dataset=PubMed --Mettack_type=mettack

MetaGpro: !python main/run.py --pretrain_type=GraphMAE --gnn_type=GraphTransformer --pretrain_dataset=PubMed --target_dataset=PubMed --Mettack_type=MetaGpro --surrogate_prompt=GPF
