import torch
from prompt_graph.utils import seed_everything


def split_train_val_test(data, class_num, nnodes_each):
        # class_num: the number of multiclass
        # nnodes_each: the number of nodes will sample for each class

        seed_everything(42)
        train_index = None

        for i in range(class_num):
            index_ =  torch.where(data.y==i)[0]

            assert index_.shape[0] >= nnodes_each, 'sample few nodes each class'


            sample = torch.randperm(index_.shape[0])[:nnodes_each]

            index_ = index_[sample]

            if train_index is None:
                train_index = index_
            else:
                train_index = torch.cat([train_index, index_])
        
        train_mask = torch.zeros(data.y.shape[0],dtype=data.y.dtype)
        train_mask[train_index] = 1

        remain_index = torch.where(train_mask==0)[0]


        validation_sample = torch.randperm(remain_index.shape[0])[:int(remain_index.shape[0]*0.2)]
        validation_index = remain_index[validation_sample]
        val_mask = torch.zeros(data.y.shape[0],dtype=data.y.dtype)
        val_mask[validation_index] =  1

        index_in_learning = torch.cat([train_index, validation_index])
        test_mask = torch.ones(data.y.shape[0],dtype=data.y.dtype)
        test_mask[index_in_learning] = 0

        return train_mask, val_mask, test_mask

class Test_data:
    def __init__(self, y):
        self.y = y


def Test():
    data = Test_data(torch.randint(0,3,[100]))
    a, b, c = split_train_val_test(data, 3, 4)
    print(torch.sum(a), torch.sum(b), torch.sum(c))
    print(data.y[a==1])
    print(a+b+c)

if __name__ == '__main__':
    Test()