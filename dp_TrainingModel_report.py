"""
在财报数据集上训练模型
"""
import torch, random
import os, pickle, math
import numpy as np

import torch.nn.functional as F
from torch.utils.data import random_split, DataLoader
from fn_SearchModule import SearchModule_report
from fn_Dataset import QPDataset
from experiment3 import evaluate_mrfira3

# 1. hyper-parameters
batch_size       = 200
learning_rate    = 0.00005
epochs           = 60
train_test_ratio = 0.8
NUM_M, NUM_N, GATE_NUM = 4, 10, 64
PATIENCE    = 2
ortho_weight = 50
aux_weight = 0.5
temperature = 0.05
model_name = 'mvsmodule-new-4'

SAMPLE_PATH = "/root/autodl-tmp/pos-samples"
print("模型参数：", batch_size, learning_rate,  NUM_M, NUM_N, model_name, PATIENCE, SAMPLE_PATH)

TEST_PATH = "/root/autodl-tmp"              # testloader
device = torch.device("cpu") if not torch.cuda.is_available() else torch.device("cuda")

# 2. building dataset from pos examples
def build_dataset():
    pos_list = []
    files = os.listdir(SAMPLE_PATH)

    for f in files:
        try:
            samples = pickle.load(open(os.path.join(SAMPLE_PATH, f), "rb"))
        except:
            print(f)
            continue
        for p in samples:
            # pos_list.append((torch.tensor(p[0], dtype=torch.float32).unsqueeze(0),
            #                  torch.tensor(p[1], dtype=torch.float32).unsqueeze(0))) # pos-sample2
            pos_list.append((torch.tensor(p[0][0], dtype=torch.float32).unsqueeze(0),
                             torch.tensor(p[1][0], dtype=torch.float32).unsqueeze(0)))   # pos-sample
    random.shuffle(pos_list)
    dataset = QPDataset(pos_list)
    return dataset

def get_dataloader(dataset, train_test_ratio):
    train_size = math.floor(len(dataset) * train_test_ratio)
    test_size = len(dataset) - train_size
    train_data, test_data = random_split(dataset, [train_size, test_size])
    train_loader = DataLoader(train_data,
                              batch_size=batch_size,
                              shuffle=True,
                              drop_last=True
                              )
    test_loader = DataLoader(test_data,
                             batch_size=batch_size,
                             shuffle=True,
                             drop_last=True
                             )
    return train_loader, test_loader

# 4. training
# 最新改进后的训练程序，26.05.28
def train_loop(train_dataloader, model, optimizer):
    total_loss = 0
    num_batches = len(train_dataloader)
    criterion = torch.nn.CrossEntropyLoss()
    for Q, P in train_dataloader:
        aux_loss = 0
        B = Q.shape[0]

        logits_list = []
        labels_list = []

        for i in range(B):
            qvec = Q[i:i + 1]
            logits, _ = model(qvec.to(device), P.to(device))  # logits shape: (B, 1)
            logits = logits.squeeze()

            q_gatelayer = model.q_gate
            p_gatelayer = model.p_gate
            aux_loss += get_moe_loss(q_gatelayer)
            aux_loss += get_moe_loss(p_gatelayer)

            # 创建 label tensor
            labels = torch.zeros(B).to(device)
            labels[i] = 1.0

            # 保持为 Tensor 对象放入列表
            logits_list.append(logits)
            labels_list.append(labels)

        logits_batch = torch.stack(logits_list)  # (100,11) batch, negative
        labels_batch = torch.stack(labels_list)  # (100,11)
        scaled_logits_batch = logits_batch / temperature
        loss = criterion(scaled_logits_batch, labels_batch)
        ortho_loss = ortho_weight * model.get_orthogonal_loss()
        loss += ortho_loss
        loss += aux_weight*aux_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / num_batches

# 加入了进度比例，用于门控网络的动态噪声机制
def train_loop2(train_dataloader, model, optimizer, epoch, total_epochs):
    total_loss = 0
    num_batches = len(train_dataloader)
    criterion = torch.nn.CrossEntropyLoss()
    model.train()

    for batch_idx, (Q, P) in enumerate(train_dataloader):
        aux_loss = 0
        B = Q.shape[0]

        #  计算当前 batch 的全局进度比例 (0.0 ~ 1.0)
        current_global_step = epoch * num_batches + batch_idx
        max_global_steps = total_epochs * num_batches
        epoch_ratio = current_global_step / max_global_steps

        logits_list = []
        labels_list = []

        for i in range(B):
            qvec = Q[i:i + 1]
            logits, _ = model(qvec.to(device), P.to(device), epoch_ratio=epoch_ratio)  # logits shape: (B, 1)
            logits = logits.squeeze()

            q_gatelayer = model.q_gate
            p_gatelayer = model.p_gate
            aux_loss += get_moe_loss(q_gatelayer)
            aux_loss += get_moe_loss(p_gatelayer)

            # 创建 label tensor
            labels = torch.zeros(B).to(device)
            labels[i] = 1.0

            # 保持为 Tensor 对象放入列表
            logits_list.append(logits)
            labels_list.append(labels)

        logits_batch = torch.stack(logits_list)  # (100,11) batch, negative
        labels_batch = torch.stack(labels_list)  # (100,11)
        scaled_logits_batch = logits_batch / temperature
        loss = criterion(scaled_logits_batch, labels_batch)
        ortho_loss = ortho_weight * model.get_orthogonal_loss()
        loss += ortho_loss
        loss += aux_weight*aux_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / num_batches


# 测试程序
def test_loop2(test_dataloader, model):
    model.eval()
    total_loss = 0
    num_batches = len(test_dataloader)

    criterion = torch.nn.CrossEntropyLoss()
    with torch.no_grad():
        for Q, P in test_dataloader:
            aux_loss = 0
            B = Q.shape[0]
            logits_list = []
            labels_list = []

            for i in range(B):
                qvec = Q[i:i + 1]
                logits, _ = model(qvec.to(device), P.to(device), epoch_ratio=0.0)
                logits = logits.squeeze()

                q_gatelayer = model.q_gate
                p_gatelayer = model.p_gate
                aux_loss += get_moe_loss(q_gatelayer)
                aux_loss += get_moe_loss(p_gatelayer)

                labels = torch.zeros(B).to(device)
                labels[i] = 1.0

                logits_list.append(logits)  # 这里存入的是 Tensor
                labels_list.append(labels)

            logits_batch = torch.stack(logits_list)  # (B,B) batch, negative
            labels_batch = torch.stack(labels_list)  # (B,B)
            scaled_logits_batch = logits_batch / temperature
            loss = criterion(scaled_logits_batch, labels_batch)
            ortho_loss = ortho_weight * model.get_orthogonal_loss()
            loss += ortho_loss
            loss += aux_weight*aux_loss

            total_loss += loss.item()

    return total_loss / num_batches


# --- 1. 定义评估函数
def evaluate_metrics(model, dataloader, device):
    model.eval()
    recall_1 = 0
    total_samples = 0
    mrr = 0.0

    with torch.no_grad():
        for Q, P in dataloader:
            Q, P = Q.to(device), P.to(device)
            B = Q.shape[0]

            for i in range(B):
                qvec = Q[i:i + 1]
                logits, _ = model(qvec.to(device), P.to(device), epoch_ratio=1.0)
                logits = logits.squeeze()
                _, indices = torch.sort(logits, descending=True)
                if indices[0] == i:
                    recall_1 += 1
                rank = torch.argwhere(indices == i) + 1
                mrr += 1.0 / rank

            total_samples += B

    return recall_1 / total_samples, mrr.item() / total_samples


# 在加入了MoE gaiting network的模型上计算负载均衡损失损失
def get_moe_loss(gate_module, w_importance=0.001):
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

    return loss * w_importance  # w_importance: 控制平衡强度的系数，通常设为 0.01 ~ 0.1

def print_metrics(t, r1, mrr):
    print(f"\n[Evaluation - Epoch {t + 1}]")
    print(f"Recall@1: {r1:.4%}, MRR: {mrr:.4f}")
    print("-" * 30)

def run():
    model = SearchModule_report(GATE_NUM, NUM_M, NUM_N).to(device)

    for param in model.parameters():
        param.requires_grad = True

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,  # 初始学习率
        weight_decay=1e-4,  # 权重衰减系数，防止权重爆炸
        eps=1e-8
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,  # 触发时学习率减半 (new_lr = lr * 0.5)
        patience=PATIENCE,
        verbose=True,
        min_lr=1e-6  # 学习率降到这个程度就不再降了
    )
    dataset = build_dataset()
    valid_loss_min = math.inf
    MRR_MAX = 0
    train_loader, test_loader = get_dataloader(dataset, train_test_ratio)

    # r1, m_rank = evaluate_metrics(model, test_loader, device)
    # r1, m_rank = evaluate_metrics2(model, test_loader, device)
    # print_metrics(0, r1, m_rank)
    evaluate_mrfira3(model)
    for t in range(epochs):
        model.train()
        # train_loss = train_loop(train_loader, model, optimizer)
        train_loss = train_loop2(train_loader, model, optimizer, t, epochs)
        model.eval()
        # valid_loss = test_loop(test_loader, model)
        valid_loss = test_loop2(test_loader, model)
        scheduler.step(valid_loss)

        curr_lr = optimizer.param_groups[0]['lr']
        # curr_temp = model.logit_scale.exp().item()
        print(f"Epoch {t + 1}/{epochs}:")
        print(f"  Train Loss: {train_loss:.6f} | Valid Loss: {valid_loss:.6f}")
        print(f"  Learning Rate: {curr_lr:.2e}")
        # print(f"  Learning Rate: {curr_lr:.2e} | Temperature: {curr_temp:.2f}")
        print("-" * 30)

        # r1, m_rank = evaluate_metrics(model, test_loader, device)
        # r1, m_rank = evaluate_metrics2(model, test_loader, device)
        # print_metrics(t, r1, m_rank)

        evaluate_mrfira3(model)

        if valid_loss < valid_loss_min:
            valid_loss_min = valid_loss
            torch.save(model, f"{model_name}-v.pth")

        if t==2:
            torch.save(model, f"{model_name}-t.pth")

if __name__ == "__main__":
    run()