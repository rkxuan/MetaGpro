import torch
import random


def sample_soft_labels(output:torch.Tensor, labeled_index:torch.Tensor, threshold:float=0.4):
    

    # Output is the model predictions
    # label_index is the index with labeled data 
    # if argmax possibility<threhold then ignore it
    

    mask = torch.zeros([output.shape[0]])
    Max_possibility = torch.max(output,dim=1)[0]

    mask[Max_possibility<0.4] = 1

    mask[label_index] = 0

    return mask

    
    