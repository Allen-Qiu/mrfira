"""
创建数据集
"""
import torch, os, pickle, random
from torch.utils.data import DataLoader


class QPDataset(torch.utils.data.Dataset):      # (query,passage)对数据集
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):   # 返回query向量和passage向量
        return self.data[index][0], self.data[index][1]

    def __len__(self):
        return len(self.data)

# 针对Title Abstract数据集
class TitleAbstractDataset(torch.utils.data.Dataset):
    def __init__(self, data_list):
        self.data = data_list

    def __getitem__(self, idx):   # 返回title的编号和对应的negatives abstract的编号
        qid, nids = self.data[idx]
        return qid, torch.tensor(nids)

    def __len__(self):
        return len(self.data)

# 针对财报 negative数据集
class ReportNegativeDataset(torch.utils.data.Dataset):
    def __init__(self, data_list, drop_last=True):
        self.data = data_list

    def __getitem__(self, idx):   # 返回(qvec, pvecs)
        qvec, pvecs = self.data[idx]
        return qvec, pvecs

    def __len__(self):
        return len(self.data)

def test1():
    SAMPLE_PATH = "/root/autodl-tmp/pos-samples"
    pos_list = []
    files = os.listdir(SAMPLE_PATH)

    for f in files:
        samples = pickle.load(open(os.path.join(SAMPLE_PATH, f), "rb"))
        pos_list.extend(samples)

    dlist = []
    for p in pos_list:
        dlist.append((torch.tensor(p[0], dtype=float), torch.tensor(p[1], dtype=float)))
    random.shuffle(dlist)
    dataset = QPDataset(dlist)
    dataloader = DataLoader(dataset, batch_size=30)

    for Q, P in dataloader:
        print(Q.shape, P.shape)

if __name__ == '__main__':
    test1()