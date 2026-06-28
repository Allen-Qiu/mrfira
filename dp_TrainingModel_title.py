"""
训练title-abstract数据集上的多向量检索模型
"""
import torch, random, time
import os, pickle, math
import numpy as np

from torch.utils.data import random_split, DataLoader
from fn_SearchModule import SearchModule_title
from fn_Dataset import TitleAbstractDataset
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset
import torch.multiprocessing as mp
from fn_EvaluateModule import EvaluateModule_Title_Contrastive as Evaluator
import os

os.environ["OMP_NUM_THREADS"] = "1"

mp.set_sharing_strategy('file_system')

# 1. hyper-parameters
batch_size       = 100
learning_rate    = 0.0001
epochs           = 100
train_test_ratio = 0.9
has_posweight    = True    # 针对不平衡数据集进行设置
NUM_M, NUM_N, GATE_NUM = 4, 10, 64
PATIENCE    = 3
LAM = 0
ortho_weight = 0.1
temperature = 0.07
model_name = 'mvsmodule-title'

TITLE_ABSTRACT_TRAIN = "/root/autodl-tmp/title_abstract_trainset.pl"
NEGATIVES = "/root/autodl-tmp/negatives.pl"
TITLE_EMBED = "/root/autodl-tmp/title-embedding.pl"
ABSTRACT_EMBED = "/root/autodl-tmp/abstract-embedding.pl"

print("模型参数：", batch_size, learning_rate, has_posweight, NUM_M, NUM_N, model_name, PATIENCE, temperature)

device = torch.device("cpu") if not torch.cuda.is_available() else torch.device("cuda")

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True  # 保证 GPU 计算结果一致
    torch.backends.cudnn.benchmark = False

seed_everything(42)

# 2. building dataset
def build_dataset(file):    # query的编号和hard negatives
    samples = pickle.load(open(file, "rb"))
    dataset = TitleAbstractDataset(samples)
    return dataset

# 用于negatives数据集
def get_dataloader2(dataset, train_test_ratio, batch_size):
    # 1. 直接使用 dataset 的长度生成索引
    train_indices, test_indices = train_test_split(
        range(len(dataset)),
        test_size=1 - train_test_ratio,
        random_state=42
    )

    # 2. 创建 Subset
    train_dataset = Subset(dataset, train_indices)
    test_dataset = Subset(dataset, test_indices)

    # 3. 创建 DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=1
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=1
    )
    return train_loader, test_loader, train_indices, test_indices

def build_embedding_layer(embeds_file):
    data = torch.tensor(pickle.load(open(embeds_file, "rb")))
    weight_matrix = data.detach()
    num_embeddings, embedding_dim = weight_matrix.shape
    emb = torch.nn.Embedding(num_embeddings, embedding_dim, padding_idx=0)
    emb.weight = torch.nn.Parameter(weight_matrix)
    emb.weight.requires_grad = False

    return emb

# 4. training
# InfoNCE损失函数的训练
def train_loop(train_dataloader, model, optimizer, title_embedding_layer, abstracts_embedding_layer):
    total_loss = 0
    total_correct = 0
    num_batches = len(train_dataloader)
    total_samples = len(train_dataloader.dataset)

    criterion = torch.nn.CrossEntropyLoss()

    for batch_qid, batch_nids in train_dataloader:  # query的id，和对应的它的negatives的id。 qid (B), nids (B, n)
        logits_list = []
        labels_list = []

        for qid, nids in zip(batch_qid,batch_nids): # 一个query id和它的negatives的id
            indices = torch.tensor([qid], dtype=torch.long)
            query_embeddings = title_embedding_layer(indices).unsqueeze(1)        # query shape=(1,1,d)
            idx_list = [qid]+nids.tolist()
            indices = torch.tensor(idx_list, dtype=torch.long)
            abstract_embeddings = abstracts_embedding_layer(indices).unsqueeze(1) # abstract shape = (B,1,d)
            logits, _ = model(query_embeddings.to(device), abstract_embeddings.to(device))        # logits shape: (B, 1)
            logits = logits.squeeze()

            # 创建 label tensor
            labels = torch.zeros(logits.shape[0]).to(device)
            labels[0] = 1.0

            # 保持为 Tensor 对象放入列表
            logits_list.append(logits)      # logits shape=(11,)
            labels_list.append(labels)      # labels shape=(11,)

        logits_batch = torch.stack(logits_list) # (100,11) batch, negative
        labels_batch = torch.stack(labels_list) # (100,11)
        scaled_logits_batch = logits_batch/temperature
        loss = criterion(scaled_logits_batch, labels_batch)

        # 计算correct
        pred_idx = torch.argmax(logits_batch, dim=1)
        target_idx = torch.argmax(labels_batch, dim=1)
        correct = (pred_idx == target_idx).sum().item()
        total_correct += correct

        # 辅助损失 (MoE Gate Loss)
        # aux_loss = get_moe_loss(model.q_gate)

        # 计算正交损失
        ortho_loss = ortho_weight * model.get_orthogonal_loss()
        loss += ortho_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # total_loss += loss.item() + aux_loss.item()
        total_loss += loss.item()

    return total_loss / num_batches, total_correct / total_samples

# 测试程序
def test_loop(test_dataloader, model, title_embedding_layer, abstracts_embedding_layer):
    model.eval()  # 评估模式
    total_loss = 0
    total_correct = 0
    total_samples = len(test_dataloader.dataset)
    num_batches = len(test_dataloader)

    criterion = torch.nn.CrossEntropyLoss()

    with torch.no_grad():
        for batch_qid, batch_nids in test_dataloader:  # query的id，和对应的它的negatives的id。 qid (B), nids (B, n)
            logits_list = []
            labels_list = []

            for qid, nids in zip(batch_qid, batch_nids):  # 一个query id和它的negatives的id
                indices = torch.tensor([qid], dtype=torch.long)
                query_embeddings = title_embedding_layer(indices).unsqueeze(1)  # query shape=(B,1,d)
                idx_list = [qid] + nids.tolist()
                indices = torch.tensor(idx_list, dtype=torch.long)
                abstract_embeddings = abstracts_embedding_layer(indices).unsqueeze(1)  # abstract shape = (B,1,d)
                logits, _ = model(query_embeddings.to(device), abstract_embeddings.to(device))  # logits shape: (B, 1)
                logits = logits.squeeze()

                # 创建 label tensor
                labels = torch.zeros(logits.shape[0]).to(device)
                labels[0] = 1.0

                logits_list.append(logits)
                labels_list.append(labels)

            logits_batch = torch.stack(logits_list)
            scaled_logits_batch = logits_batch/temperature
            labels_batch = torch.stack(labels_list)
            loss = criterion(scaled_logits_batch, labels_batch)

            # 计算正交损失
            ortho_loss = ortho_weight * model.get_orthogonal_loss()
            loss += ortho_loss

            total_loss += loss.item()

            pred_idx = torch.argmax(logits_batch, dim=1)
            target_idx = torch.argmax(labels_batch, dim=1)
            correct = (pred_idx == target_idx).sum().item()
            total_correct += correct

    avg_loss = total_loss / num_batches
    accuracy = total_correct / total_samples

    return avg_loss, accuracy


# 在加入了MoE gaiting network的模型上计算辅助损失
def get_moe_loss(gate_module, w_importance=0.001):
    """
    计算负载均衡损失
    w_importance: 控制平衡强度的系数，通常设为 0.01 ~ 0.1
    """
    prob = gate_module.last_prob  # (batch, m)
    mask = gate_module.last_mask  # (batch, m)

    # 1. 负载重要性：每个特征收到的总概率均值
    importance = prob.mean(dim=0)  # shape: (m,)

    # 2. 选择频率：每个特征实际被选中的频率
    load = mask.mean(dim=0)  # shape: (m,)

    # 3. 负载均衡损耗：变异系数的平方 (CV^2)
    # 计算公式：m * sum(importance * load)
    # 当 importance 和 load 都是均匀分布时，该值最小
    loss = prob.shape[1] * torch.sum(importance * load)

    return loss * w_importance

def print_metrics(res):
    mrr = res[-1]
    print(f"Recall@1: {res[0].item() if isinstance(res[0], torch.Tensor) else res[0]:.6f}, "
          f"@5: {res[1].item() if isinstance(res[1], torch.Tensor) else res[1]:.6f}, "
          f"@10: {res[2].item() if isinstance(res[2], torch.Tensor) else res[2]:.6f}, "
          f"@20: {res[3].item() if isinstance(res[3], torch.Tensor) else res[3]:.6f}, "
          f"@50: {res[4].item() if isinstance(res[4], torch.Tensor) else res[4]:.6f}, "
          f"rank1: {res[5].item() if isinstance(res[5], torch.Tensor) else res[5]:.6f},"
          f"rank2: {res[6].item() if isinstance(res[6], torch.Tensor) else res[6]:.6f},"
          f"mrr: {mrr.item():.6f}")

def run():
    seed_everything(42)
    title_embed_layer = build_embedding_layer(TITLE_EMBED)
    abstract_embed_layer = build_embedding_layer(ABSTRACT_EMBED)
    model = SearchModule_title(GATE_NUM, NUM_M, NUM_N).to(device)

    for param in model.parameters():
        param.requires_grad = True

    # optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,  weight_decay=1e-2)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,  # 初始学习率
        weight_decay=1e-2,  # 权重衰减系数，防止权重爆炸
        eps=1e-8  # 数值稳定性常数
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,  # 触发时学习率减半 (new_lr = lr * 0.5)
        patience=PATIENCE,  # 如果 PATIENCE 个 epoch 以后验证集 Loss 还没改善，就降速
        min_lr=1e-6  # 学习率降到这个程度就不再降了
    )
    dataset = build_dataset(NEGATIVES)   # (query_id, (negatives_ids))
    valid_loss_min = math.inf
    train_loader, test_loader, train_indices, test_indices = get_dataloader2(dataset, train_test_ratio, batch_size)
    evaluator = Evaluator(test_indices)

    for t in range(epochs):
        print(f"\nEpoch {t + 1}/{epochs}:")
        model.train()
        train_loss, train_accuracy = train_loop(train_loader, model, optimizer, title_embed_layer, abstract_embed_layer)
        model.eval()
        valid_loss, valid_accuracy = test_loop(test_loader, model, title_embed_layer, abstract_embed_layer)
        scheduler.step(valid_loss)

        curr_lr = optimizer.param_groups[0]['lr']
        print(f"  Train Loss: {train_loss:.6f} | Valid Loss: {valid_loss:.6f}")
        print(f"  Train Accuracy: {train_accuracy:.6f} | Valid Accuracy: {valid_accuracy:.6f}")
        print(f"  Learning Rate: {curr_lr:.2e}")
        print("-"*30)

        if t % 5 == 0:
            res = evaluator.evaluate(model)
            print_metrics(res)

        print("*" * 30)

        if valid_loss < valid_loss_min:
            valid_loss_min = valid_loss
            torch.save(model, f"{model_name}-m.pth")

if __name__ == "__main__":
    run()