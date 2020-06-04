#!/bin/sh
gpu_id=0
cd ..

mkdir log_ablation

#w/o mid
python main_SBMs_node_classification.py --gpu_id $gpu_id --config './configs/SBMs_node_clustering_AUTOGCN_CLUSTER.json' --L 8 --runs 5 --opt over --hidden_dim 110 --num_high 1 --num_low 1 --num_mid 0 > ./log_ablation/AUTOGCN-cluster-womiddle.log

#w/o high
python main_SBMs_node_classification.py --gpu_id $gpu_id --config './configs/SBMs_node_clustering_AUTOGCN_CLUSTER.json' --L 8 --runs 5 --opt over --hidden_dim 110 --num_high 0 --num_low 1 --num_mid 1 > ./log_ablation/AUTOGCN-cluster-wohigh.log

#w/o low
python main_SBMs_node_classification.py --gpu_id $gpu_id --config './configs/SBMs_node_clustering_AUTOGCN_CLUSTER.json' --L 8 --runs 5 --opt over --hidden_dim 110 --num_high 1 --num_low 0 --num_mid 1 > ./log_ablation/AUTOGCN-cluster-wolow.log

#w/o over
python main_SBMs_node_classification.py --gpu_id $gpu_id --config './configs/SBMs_node_clustering_AUTOGCN_CLUSTER.json' --L 8 --runs 5 --opt single > ./log_ablation/AUTOGCN-cluster-single.log

#w/o par
python main_SBMs_node_classification.py --gpu_id $gpu_id --config './configs/SBMs_node_clustering_AUTOGCN_CLUSTER.json' --L 8 --runs 5 --opt fix > ./log_ablation/AUTOGCN-cluster-fix.log

#w/o gate
python main_SBMs_node_classification.py --gpu_id $gpu_id --config './configs/SBMs_node_clustering_AUTOGCN_CLUSTER.json' --L 8 --runs 5 --gate False > ./log_ablation/AUTOGCN-cluster-wogate.log







