import torch
import os
import numpy as np


def add_delete_statistics(train_mask, y, adj_ori, modified_adj):
    adj_changes = modified_adj - adj_ori
    add_edge_index_0, add_edge_index_1 = torch.where(adj_changes > 0)
    delete_edge_index_0, delete_edge_index_1 =  torch.where(adj_changes < 0)
    total_edge_index_0, total_edge_index_1 = torch.where(adj_changes != 0)

    assert total_edge_index_0.shape[0] == add_edge_index_0.shape[0] + delete_edge_index_0.shape[0], 'please check something wrong'

    total_num = total_edge_index_0.shape[0]
    # UU UL LL
    # UL and LU is equal in undirected graph
    UU_num = 0
    UL_num = 0
    LL_num = 0


    filp_num = np.array([0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0,
                        0, 0, 0, 0, 0], dtype=int)

    num_record = np.zeros(adj_ori.shape[0], dtype=int)
    
    # 想不出更高效的算法 还是写了一个for循环 O(n)
    for i, j in zip(total_edge_index_0, total_edge_index_1):
        if train_mask[i] == 1  and train_mask[j] == 1:
            LL_num += 1
        elif train_mask[i] == 0 and train_mask[j] == 0:
            UU_num += 1
        else:
            UL_num += 1
        
        filp_num[num_record[int(i)]] -= 1
        if num_record[int(i)] < 19:
            num_record[int(i)] += 1 
            filp_num[num_record[int(i)]] += 1 
        else:
            filp_num[num_record[int(i)]] += 2


    print("Statistics about UU UL LL:")
    print("LL: {:.3f}".format(LL_num/total_num))
    print("UL: {:.3f}".format(UL_num/total_num))
    print("UU: {:.3f}".format(UU_num/total_num))


    A_equ = 0
    A_nequ = 0
    D_equ = 0
    D_nequ = 0
    for i, j in zip(add_edge_index_0, add_edge_index_1):
        if y[i] == y[j]:
            A_equ += 1
        else:
            A_nequ += 1
    
    for i, j in zip(delete_edge_index_0, delete_edge_index_1):
        if y[i] == y[j]:
            D_equ += 1
        else:
            D_nequ += 1

    print("Statistics about A/D:")
    print("A_E: {:.3f}".format(A_equ/total_num))
    print("A_N: {:.3f}".format(A_nequ/total_num))
    print("D_E: {:.3f}".format(D_equ/total_num))
    print("D_N: {:.3f}".format(D_nequ/total_num))

    print("Statistics about the number of nodes whose n edges are filped:")
    print(filp_num[1:])




    