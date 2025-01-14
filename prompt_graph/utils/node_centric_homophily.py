import torch
import os
import matplotlib.pyplot as plt


def node_centric_homophily(x, adj):
    # define in "UNDERSTANDING AND IMPROVING GRAPH INJECTION ATTACK BY PROMOTING UNNOTICEABILITY ICLR2022"
    
    diag = torch.diag(adj)
    a_diag = torch.diag_embed(diag)
    adj = adj - a_diag


    D = torch.sum(adj, dim=1)
    D_inv = torch.pow(D, -1/2)
    D_inv[torch.isinf(D_inv)] = 0.
    D_mat_inv = torch.diag(D_inv)

    adj_norm = D_mat_inv @ adj @ D_mat_inv   # GCN的归一化方式

    r = adj_norm @ x

    homophily = torch.nn.functional.cosine_similarity(x, r)
    homophily = 0.5 * (homophily + 1)        # from (-1, 1) to (0, 1)
    return homophily


def visualize_topology_attack_influence(x, adj_ori, modified_adj, prompted_features, surrogate_prompt, 
                                        task_prompt, target_dataset, Mettack_type, all_in_one_threshold=0.5, token_num=10):
    folder_path = "./result"+ "/surrogate_" + surrogate_prompt
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    node_centric_homophily_ori = node_centric_homophily(x, adj_ori)
    node_centric_homophily_attack = node_centric_homophily(x, modified_adj)
    node_centric_homophily_ori = node_centric_homophily_ori.to('cpu').detach().numpy()
    node_centric_homophily_attack = node_centric_homophily_attack.to('cpu').detach().numpy()
        
    prompt_centric_homophily_ori = node_centric_homophily(prompted_features, adj_ori)
    prompt_centric_homophily_attack = node_centric_homophily(prompted_features, modified_adj)
    prompt_centric_homophily_ori = prompt_centric_homophily_ori.to('cpu').detach().numpy()
    prompt_centric_homophily_attack = prompt_centric_homophily_attack.to('cpu').detach().numpy()


    plt.xlabel("Homophily between features", fontsize=14)
    plt.ylabel("The number of nodes", fontsize=14)
    plt.hist(node_centric_homophily_ori, bins=20, alpha=0.5, edgecolor='black', label='Unattacked')
    plt.hist(node_centric_homophily_attack, bins=20, alpha=0.5, edgecolor='black', label="Attacked")
    plt.legend(fontsize=12)
    save_file_name = f"homophily_features_{task_prompt}_{target_dataset}_{Mettack_type}.png"
    path = os.path.join(folder_path, save_file_name)
    plt.savefig(path)
    plt.close()

    prompt_centric_homophily_ori = prompt_centric_homophily_ori[prompt_centric_homophily_ori>0.9]
    prompt_centric_homophily_attack = prompt_centric_homophily_attack[prompt_centric_homophily_attack>0.9]
    plt.xlabel("Homophily between prompted features", fontsize=14)
    plt.ylabel("The number of nodes", fontsize=14)
    plt.hist(prompt_centric_homophily_ori, bins=20, alpha=0.5, edgecolor='black', label='Unattacked')
    plt.hist(prompt_centric_homophily_attack, bins=20, alpha=0.5, edgecolor='black', label="Attacked")
    plt.legend(fontsize=12)
    save_file_name = f"homophily_prompted_{task_prompt}_{target_dataset}_{Mettack_type}.png"
    path = os.path.join(folder_path, save_file_name)
    plt.savefig(path)
    plt.close()