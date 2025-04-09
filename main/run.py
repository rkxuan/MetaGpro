import torch
from prompt_graph.attack import BaseAttack, Metattack, Metacon_s, Metacon_d, MetaGraD, MetaGpro, MetaGpro_Approx, DICE, AtkSE
from prompt_graph.defense import prune
import argparse
from prompt_graph.utils import edge_adding, edge_dropping, feature_masking, edge_weighted_dropping
from prompt_graph.data import load4node,load4graph, split_induced_graphs, induced_graphs
from prompt_graph.tasker import NodeTask
import os
import pickle
from prompt_graph.utils import seed_everything, visualize_topology_attack_influence, add_delete_statistics
import numpy as np


def transform_feature_dim(data, pca, input_dim):   # this function is also used in prompt_graph.attack.BaseAttack
    if pca is False:  # 不用pca降维
        data.x = data.x[:, :input_dim]
    else:    #pca指定了降维维度
        _, _, V = torch.pca_lowrank(data.x, input_dim)
        data.x = torch.matmul(data.x, V[:, :input_dim])

def load_induced_graph_and_save(folder_path='./Experiment/induced_graph', dataset_name='Cora', data=None, small=100, large=300, device='cpu'):

    folder_path = os.path.join(folder_path , dataset_name)
    if not os.path.exists(folder_path):
            os.makedirs(folder_path)

    file_path = os.path.join(folder_path, 'induced_graph_min'+ str(small) +'_max'+str(large)+'.pkl')
    if os.path.exists(file_path):
            with open(file_path, 'rb') as f:
                print('loading induced graph...')
                graphs_list = pickle.load(f)
                print('Done!!!')
    else:
        print('Begin split_induced_graphs.')
        split_induced_graphs(data, folder_path, device, smallest_size=small, largest_size=large)
        with open(file_path, 'rb') as f:
            graphs_list = pickle.load(f)
    graphs_list = [graph.to(device) for graph in graphs_list]
    return graphs_list

def load_induced_graph(data=None, small=100, large=300, device='cpu'):
    """
    w.r.t there are so many hyper-parameters for these attacks
    call this function to load induced graphs after adversarial attack
    """
    graphs_list = induced_graphs(data, device, small, large)
    print("done")
    graphs_list = [graph.to(device) for graph in graphs_list]
    return graphs_list


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

if __name__ == '__main__':
    # Written in SEU in 2024/10, based on ProG and Deeprobust
    # ProG: https://github.com/sheldonresearch/ProG
    # Deeprobust: http://github.com/DSE-MSU/DeepRobust

    attack_dict = {'base':BaseAttack, 'mettack':Metattack, 'metacon_s':Metacon_s, 'metacon_d':Metacon_d,  'dice': DICE,
                   'atkse':AtkSE, 'GraD':MetaGraD, 'MetaGpro':MetaGpro, 'MetaGpro_app':MetaGpro_Approx}
    augmentation_dict = {'edge_dropping':edge_dropping, 'edge_adding':edge_adding, 'feature_masking':feature_masking, "edge_weighted_dropping":edge_weighted_dropping} # define in prompt_graph.utils.simple_augmentation
    prompt_list = ['All-in-one-softmax', 'All-in-one-mean', 'All-in-one','All-in-one-token','GPF', 'GPF-plus', 'Gprompt', 'GPPT', 'MultiGprompt']
    attack_loss_list = ['CE', 'GraD', 'Tanh', 'Bias_Tanh', 'MCE']
    surrogate_prompt_list = ['Two-views', 'All-in-one-mean', 'All-in-one-softmax', 'All-in-one', 'GPF', 'GPF-plus', 'Gprompt', 'GPF-GNN', 'Gprompt-GNN']   # Sparse-All-in-one is the All-in-one-mean in the paper
    model_list = ['GCN', 'GIN', 'GAT', "GraphTransformer"]  # define model in prompt_graph.model 
    dataset_list = ['PubMed', 'CiteSeer', 'Cora', 'Computers', 'Photo', 'DBLP', 'CoraFull']  # define dataset by prompt_graph.load4data.load4node_sparse function
    pretrain_list = ['GraphCL', 'SimGRACE','GraphMAE', 'DGI', 'ADGCL']


    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--seed', type=int,  default=42)
    parser.add_argument('--Mettack_type', type=str, choices=attack_dict.keys(), default='mettack')

    # Pretrain_stage 
    parser.add_argument('--pretrain_type', choices=pretrain_list, default='GraphMAE')
    parser.add_argument('--gnn_type', type=str, choices=model_list, default='GCN')  # Benchmark shows GCN is good enough, GT complex but not such suitable in downstream 
    parser.add_argument('--hid_dim', type=int, default=128)
    parser.add_argument('--gln', type=int, default=2)
    parser.add_argument('--num_pretrain_epoch', type=int, default=500)
    parser.add_argument('--pretrain_dataset', type=str, choices=dataset_list, default='PubMed')
    parser.add_argument('--pre_train_model_file_path', type=str, default='/root/autodl-tmp/ProG/pre_trained_model/')
    

    # Attack_base
    parser.add_argument('--attack_structure', type=str2bool, default=True)
    parser.add_argument('--attack_features', type=str2bool, default=False)
    parser.add_argument('--target_dim', type=int, default=100)
    parser.add_argument('--pca', type=bool, default='True')
    parser.add_argument('--target_dataset', type=str, choices=dataset_list, default='Cora')
    parser.add_argument('--device', type=str, default='auto')


    # Meta-based Attack
    parser.add_argument('--train_iter', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.01)                 # set 0.1 for Cora, CiteSeer, PubMed; set 0.01 for Computers, Photo
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--lambda_', type=float, default=0.5)
    parser.add_argument('--budget', type=float, default=0.05)
    parser.add_argument('--save_fold_path',type=str, default='/root/autodl-tmp/deeprobust')
    parser.add_argument('--with_bias', type=str2bool, default=False)
    parser.add_argument('--with_relu', type=str2bool, default=False)
    parser.add_argument('--ll_constraint', type=str2bool, default=False)

    # GCL loss for unlabeled data 
    parser.add_argument('--vic_coef1', type=float, default=1.0)
    parser.add_argument('--vic_coef2', type=float, default=1.0)
    parser.add_argument('--vic_coef3', type=float, default=0.04)
    parser.add_argument('--beta_', type=float, default=0.1)  
    parser.add_argument('--aug_ratio',type=float, default=0.2)
    parser.add_argument('--augmentation', choices=augmentation_dict.keys(), type=str, default='random')

    # MetaGpro
    parser.add_argument('--all_in_one_threshold', type=float, default=0.5) # weight mask threshold for All-in-one and Sparse-All-in-one
    parser.add_argument('--surrogate_token_num', type=int, default=10)
    parser.add_argument('--surrogate_prompt', type=str, choices=surrogate_prompt_list, default='Two-views')
    parser.add_argument('--attack_loss', type=str, choices=attack_loss_list, default='CE')
    parser.add_argument('--lenda_1', type=float, default=0.0)              # weight for two views's similarity loss
    parser.add_argument('--lenda_2', type=float, default=0.0)              # weight for over-robust learning loss

    # downstream task
    parser.add_argument('--labeled_each_class', type=int, default=100)    # random sample and attack many times will cost much, so sample once
    parser.add_argument('--task_num', type=int, default = 1)              # shot_num = labeled_each_class // task_num
    parser.add_argument('--batch_size',type=int, default = 128)
    parser.add_argument('--task_epochs', type=int, default = 500)
    parser.add_argument('--task_lr', type=float, default = 0.001)         
    parser.add_argument('--prompt_type', type=str, choices=prompt_list, default='GPF-plus')
    parser.add_argument('--shot_folder', type=str, default='/root/autodl-tmp/ProG/Node')
    parser.add_argument('--token_num', type=int, default=10)


    # defense task
    parser.add_argument('--dropout', type=float, default=0.1)

    args, _ = parser.parse_known_args()
    

    seed_everything(args.seed)
    attack_model = attack_dict[args.Mettack_type]   
    attack_model = attack_model(**vars(args))
    attack_model.attack()
    modified_adj = attack_model.modified_adj
    modified_features = attack_model.modified_features
    input_dim = attack_model.input_dim
    output_dim = attack_model.out_dim
    train_mask = attack_model.train_mask
    pre_train_model_path = attack_model.pre_train_model_path
    device = attack_model.device
    data = attack_model.data.to(device)
    adj_ori = attack_model.adj_ori
    
    # downstream task, and we only focus on NodeTask
    seed_everything(args.seed)           
    # seed_everything not always works
    # the reason here see https://github.com/pyg-team/pytorch_geometric/issues/3175
    # and one solution is to replace edge_index of SparseTensor, but not all function support SparseTensor
    # so we have tried out best to add seed_everything, and set --task_num=1, then record the best result

    edge_index_0, edge_index_1 = torch.where(adj_ori == 1)
    edge_index =  torch.stack([edge_index_0, edge_index_1], dim=0)
    data.edge_index = edge_index
    shot_num = int(args.labeled_each_class // args.task_num)

    # you can use the following one line to see the Share of the perturbation
    # add_delete_statistics(train_mask, data.y, adj_ori, modified_adj)

    if args.prompt_type in ['All-in-one-mean','All-in-one-softmax', 'Gprompt','All-in-one','All-in-one-token', 'GPF', 'GPF-plus']:
        graphs_list = load_induced_graph(data, 100, 300, device)
    else:  # I dont know whether other prompt functions can work, but ProG write this code in Tutorial
        graphs_list = None 

    tasker = NodeTask(data=data, input_dim=input_dim, output_dim=output_dim, task_num=args.task_num, shot_num=shot_num, 
            graphs_list=graphs_list, train_mask=train_mask, shot_folder=args.shot_folder,
            pre_train_model_path=pre_train_model_path, gnn_type=args.gnn_type, 
            hid_dim=args.hid_dim, gln=args.gln,target_dataset_name = args.target_dataset, 
            prompt_type=args.prompt_type, task_epochs=args.task_epochs,task_lr = args.task_lr, 
            batch_size = args.batch_size, device=device, token_num=args.token_num)

    _, test_acc_, std_test_acc_, f1_, std_f1_, roc_, std_roc_, _, _, pros, labels = tasker.run()
    # you can use following lines to save the decision margin of training data
    # pro: softmax(z), you can change the code to log_softmax(z)
    # if pros is not None and labels is not None: 
    #    sorted = pros.argsort(-1)
    #    best_non_target_class = sorted[sorted != labels[:, None]].reshape(pros.size(0), -1)[:, -1]
    #    margin = (
    #                pros[np.arange(pros.size(0)), labels]
    #                - pros[np.arange(pros.size(0)), best_non_target_class]
    #        )
    #    margin = margin.cpu().numpy()
    #    save_folder_path = "./result"+ "/prompt_" + args.prompt_type + "/dataset_" + args.target_dataset
    #    if not os.path.exists(save_folder_path):
    #        os.makedirs(save_folder_path)
    #    file_name = "clean_margin.npy"
    #    file_path = os.path.join(save_folder_path, file_name)
    #    np.save(file_path, margin)"""

    if args.Mettack_type != 'base':  # you can use '--Mettack_type==base' to make attack silent and NodeTask above will be 'normal' on target dataset
        seed_everything(args.seed)
        if args.attack_structure:
            edge_index_0, edge_index_1 = torch.where(modified_adj == 1)
            modified_edge_index =  torch.stack([edge_index_0, edge_index_1], dim=0)
            data.edge_index = modified_edge_index
        if args.attack_features:
            data.x = modified_features
        if args.prompt_type in ['All-in-one-mean','All-in-one-softmax', 'Gprompt','All-in-one','All-in-one-token', 'GPF', 'GPF-plus']:
            graphs_list = load_induced_graph(data, 100, 300, device)
        else:  # I dont know whether other prompt functions can work, but ProG write this code in Tutorial
            graphs_list = None

        #prune(data, args.dropout)

        tasker = NodeTask(data=data, input_dim=input_dim, output_dim=output_dim, task_num=args.task_num, shot_num=shot_num, 
                graphs_list=graphs_list, train_mask=train_mask, shot_folder=args.shot_folder,
                pre_train_model_path=pre_train_model_path, gnn_type=args.gnn_type, 
                hid_dim=args.hid_dim, gln=args.gln,target_dataset_name = args.target_dataset, 
                prompt_type=args.prompt_type, task_epochs=args.task_epochs,task_lr = args.task_lr, 
                batch_size = args.batch_size, device=device, token_num=args.token_num)
        
        _, test_acc, std_test_acc, f1, std_f1, roc, std_roc, _, _, pros, labels= tasker.run()

        print("/n------------------------After attack------------------------/n")

        print("In {} dataset".format(args.target_dataset))
        print("Final Accuracy {:.4f}±{:.4f}(std)".format(test_acc, std_test_acc)) 
        print("Final F1 {:.4f}±{:.4f}(std)".format(f1,std_f1)) 
        print("Final AUROC {:.4f}±{:.4f}(std)".format(roc, std_roc))

        # you can use the following two lines to see the attack influence on original graph and prompted graph
        # prompted_features = tasker.get_prompted_features()
        # visualize_topology_attack_influence(data.x, adj_ori, modified_adj, prompted_features, args.surrogate_prompt, args.prompt_type, args.target_dataset, args.Mettack_type, args.all_in_one_threshold, args.token_num)

        # you can use following lines to save the decision margin of training data
        # pro: softmax(z), you can change the code to log_softmax(z)
        # if pros is not None and labels is not None: 
        #    sorted = pros.argsort(-1)
        #    best_non_target_class = sorted[sorted != labels[:, None]].reshape(pros.size(0), -1)[:, -1]
        #    margin = (
        #                pros[np.arange(pros.size(0)), labels]
        #               - pros[np.arange(pros.size(0)), best_non_target_class]
        #        )
        #    margin = margin.cpu().numpy() 
        #    save_folder_path = "./result"+ "/prompt_" + args.prompt_type + "/dataset_" + args.target_dataset
        #    if not os.path.exists(save_folder_path):
        #        os.makedirs(save_folder_path)
        #    file_name = f"{args.Mettack_type}_{args.surrogate_prompt}_{args.attack_loss}_margin.npy"
        #    file_path = os.path.join(save_folder_path, file_name)
        #    np.save(file_path, margin)"""
    else:
        print("Attack is None")

    print("/n------------------------Before attack------------------------/n")
    print("In {} dataset".format(args.target_dataset))
    print("Final Accuracy {:.4f}±{:.4f}(std)".format(test_acc_, std_test_acc_)) 
    print("Final F1 {:.4f}±{:.4f}(std)".format(f1_, std_f1_))
    print("Final AUROC {:.4f}±{:.4f}(std)".format(roc_, std_roc_))




