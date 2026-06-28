"""
重排序模块，计算query和passage多向量检索的相似度
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

# 注意力探针方案
class AttentionExtractor(nn.Module):
    def __init__(self, input_dim=1024, num_vectors=64, output_dim=256, dropout_p=0.1):
        super().__init__()
        # 每个探针去感应 1024 维中的不同部分
        self.probes = nn.Parameter(torch.randn(num_vectors, output_dim))
        self.kv_proj = nn.Linear(input_dim, output_dim)
        self.attn = nn.MultiheadAttention(embed_dim=output_dim,
                                          num_heads=8,
                                          dropout=dropout_p,
                                          batch_first=True)
        self.kv_dropout = nn.Dropout(p=dropout_p)
        self.q_dropout = nn.Dropout(p=dropout_p)

    def get_orthogonal_loss(self):
        # 归一化探针，确保长度一致
        probes_norm = F.normalize(self.probes, p=2, dim=1)
        # 计算相关性矩阵 [16, 16]
        correlation_matrix = torch.matmul(probes_norm, probes_norm.t())
        # 创建单位矩阵
        identity = torch.eye(self.probes.size(0)).to(self.probes.device)
        # 计算与单位矩阵的差距（只惩罚非对角线部分）
        loss = torch.mean((correlation_matrix - identity) ** 2)
        return loss

    def forward(self, x):
        B = x.size(0)
        #  x 是 [B, 1, 1024]， kv 变换后变成 [B, 1, 128]
        kv = self.kv_proj(x)
        kv = self.kv_dropout(kv)

        # probes 是 [16, 128]，扩展为 [B, 16, 128]
        q = self.probes.unsqueeze(0).expand(B, -1, -1)
        q = self.q_dropout(q)

        # 交叉注意力提取：探针(Q) 去观察 1个特征向量(K,V)
        out, _ = self.attn(q, kv, kv)
        # return F.normalize(out, p=2, dim=-1)
        return out

# 针对title-abstract数据集
class SearchModule_title(nn.Module):
    def __init__(self, total, num_m, num_n): # total是初始query和passagr分别总共要提取的特征数，num_m是最终从查询query选择的特征数；num_n是从段落中选择的特征数
        super(SearchModule_title, self).__init__()
        self.num_m = num_m
        self.num_n = num_n
        self.total = total

        self.q_extractor = self.get_extractor(total)
        self.p_extractor = self.get_extractor(total)
        self.q_gate = self.get_gating_module(total, total, self.num_m)
        self.p_gate = self.get_gating_module(total, total, self.num_n)
        self.q_extractor.apply(self.extractor_weights_init)
        self.p_extractor.apply(self.extractor_weights_init)

    def forward(self, qvec, pvec):
        Q_feature = self.q_extractor(qvec) # B,m,d
        P_feature = self.p_extractor(pvec) # B,m,d
        if qvec.shape[0] == 1:
            Q_feature = Q_feature.repeat(P_feature.shape[0], 1, 1)
            X = torch.cat((Q_feature, P_feature), dim=2) # B,m,2d
        else:
            X = torch.cat((Q_feature, P_feature), dim=2)
        Q = self.q_gate(X, Q_feature)  # B,m,d
        P = self.p_gate(X, P_feature)   # X用于产生gate, P的shape=(B,m,d)

        # 2. 计算跨 Batch 的相似度矩阵 S
        # Q = F.normalize(Q, p=2, dim=2)
        # P = F.normalize(P, p=2, dim=2)
        # S = torch.einsum('bmd, bnd -> bmn', Q_feature, P_feature)  # shape: (B, m, n)
        S = torch.einsum('bmd, bnd -> bmn', Q, P)  # shape: (B, m, n)

        # S shape: (B, m, n)
        # 1. 沿着行 (dim=2, 即 n_dim) 找最大值
        row_max, _ = torch.max(S, dim=2)  # shape: (B, m)

        # 2. 对得到的最大值向量求平均 (dim=1, 即 m_dim)
        # keepdim=True 可以保证结果是 (B, 1) 而不是 (B,)
        sim = torch.mean(row_max, dim=1, keepdim=True) * self.total/self.num_m  # shape: (B, 1)

        return sim, None

    def get_extractor(self, total, dropout=0.1):
        # extractor = FeatureExtractor(total)
        extractor = AttentionExtractor(num_vectors=total)
        # extractor = ConvTransposeExtractor(total)
        # extractor = ConvTransformerExtractor(total)
        return extractor

    def get_orthogonal_loss(self):
        return self.p_extractor.get_orthogonal_loss() + self.q_extractor.get_orthogonal_loss()

    # gating network
    def get_gating_module(self, input, output, topk):
        # return HardGatingModule(input, output, topk=topk)
        return NoisyHardGatingModule(input, output, topk)

    def extractor_weights_init(self, m):
        if isinstance(m, (nn.Conv1d, nn.ConvTranspose1d, nn.Linear)):
            if m.weight is not None:
                nn.init.kaiming_normal_(m.weight.data, a=0.2, mode='fan_in', nonlinearity='leaky_relu')
            if m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)

        elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
            if m.weight is not None:
                nn.init.constant_(m.weight.data, 1.0)
            if m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)

# 针对财报数据集
class SearchModule_report(nn.Module):
    def __init__(self, total, num_m, num_n): # total是初始query和passagr分别总共要提取的特征数，num_m是最终从查询query选择的特征数；num_n是从段落中选择的特征数
        super(SearchModule_report, self).__init__()
        self.num_m = num_m
        self.num_n = num_n
        self.total = total

        self.q_extractor = self.get_extractor(total)
        self.p_extractor = self.get_extractor(total)
        self.q_gate = self.get_gating_module(total, total, self.num_m)
        self.p_gate = self.get_gating_module(total, total, self.num_n)
        self.q_extractor.apply(self.extractor_weights_init)
        self.p_extractor.apply(self.extractor_weights_init)
        self.feature_dropout = nn.Dropout(p=0.1)

    def forward(self, qvec, pvec, epoch_ratio=1.0):
        # qvec: (B, 1, d) 变为 (B, total, d)
        Q_feature = self.q_extractor(qvec) # B,total,d
        # pvec: (B, 1, d) 变为 (B, n, d)
        P_feature = self.p_extractor(pvec) # B,total,d
        if qvec.shape[0] == 1:
            Q_feature = Q_feature.repeat(P_feature.shape[0], 1, 1)
            X = torch.cat((Q_feature, P_feature), dim=2) # B,total,2d
        else:
            X = torch.cat((Q_feature, P_feature), dim=2) # B,total,2d
        X = self.feature_dropout(X)
        Q = self.q_gate(X, Q_feature, epoch_ratio)  # B,m,d
        P = self.p_gate(X, P_feature, epoch_ratio)   # X用于产生gate, P的shape=(B,m,d)

        # 2. 计算跨 Batch 的相似度矩阵 S
        # Q = F.normalize(Q, p=2, dim=2)
        # P = F.normalize(P, p=2, dim=2)
        # S = torch.einsum('bmd, bnd -> bmn', Q_feature, P_feature)  # shape: (B, m, n)
        S = torch.einsum('bmd, bnd -> bmn', Q, P)  # shape: (B, m, n)

        # S shape: (B, m, n)
        # 1. 沿着行 (dim=2, 即 n_dim) 找最大值
        row_max, _ = torch.max(S, dim=2)  # shape: (B, m)

        # 2. 对得到的最大值向量求平均 (dim=1, 即 m_dim)
        # keepdim=True 可以保证结果是 (B, 1) 而不是 (B,)
        sim = torch.mean(row_max, dim=1, keepdim=True) * self.total/self.num_m  # shape: (B, 1)

        return sim, None

    def get_extractor(self, total, dropout=0.1):
        # extractor = FeatureExtractor(total)
        extractor = AttentionExtractor(num_vectors=total)
        # extractor = ConvTransposeExtractor(total)
        # extractor = ConvTransformerExtractor(total)
        return extractor

    def get_orthogonal_loss(self):
        return self.p_extractor.get_orthogonal_loss() + self.q_extractor.get_orthogonal_loss()

    # gating network
    def get_gating_module(self, input, output, topk):
        return DynamicNoisyHardGatingModule(input, output, topk=topk)
        # return NoisyHardGatingModule(input, output, topk)

    def extractor_weights_init(self, m):
        if isinstance(m, (nn.Conv1d, nn.ConvTranspose1d, nn.Linear)):
            if m.weight is not None:
                nn.init.kaiming_normal_(m.weight.data, a=0.2, mode='fan_in', nonlinearity='leaky_relu')
            if m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)

        elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
            if m.weight is not None:
                nn.init.constant_(m.weight.data, 1.0)
            if m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)


# 注入噪声的Hard Gating
class NoisyHardGatingModule(nn.Module):
    def __init__(self, input, output, topk):
        super(NoisyHardGatingModule, self).__init__()
        self.topk = topk
        self.noise_epsilon = 1e-2
        # 门控得分网络
        self.gate_score = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(input, output),
            nn.Dropout(0.2)
        )
        # 噪声权重网络
        self.noise_w = nn.Linear(input, output)
        # 状态保存
        self.last_prob = None
        self.last_mask = None
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # 使用 Kaiming 正态分布初始化权重
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    # 偏置项通常初始化为 0
                    init.constant_(m.bias, 0)

    def forward(self, x, input):    # X用于产生gate, 然后在input上进行选择
        # 1. 计算原始得分
        clean_scores = self.gate_score(x)
        # 2. 注入噪声 (仅训练模式)
        if self.training:
            noise_std = F.softplus(self.noise_w(x.mean(dim=-1))) + self.noise_epsilon
            noise = torch.randn_like(clean_scores) * noise_std
            scores = clean_scores + noise
        else:
            scores = clean_scores

        # 3. 归一化得分
        soft_prob = F.softmax(scores, dim=1)

        # 4. 硬选择 Top-K
        values, indices = torch.topk(scores, k=self.topk, dim=1)
        mask = torch.zeros_like(scores).scatter(1, indices, 1.0)

        # 5. 保存状态供 Loss 使用
        self.last_prob = soft_prob
        self.last_mask = mask

        # 6. STE 梯度直通技巧
        hard_gate = (mask - soft_prob).detach() + soft_prob
        # 7. 应用门控
        return input * hard_gate.unsqueeze(-1)

# 退火自适应探索（Annealed Adaptive Exploration）
class DynamicNoisyHardGatingModule(nn.Module):
    def __init__(self, input, output, topk):
        super(DynamicNoisyHardGatingModule, self).__init__()
        self.topk = topk
        self.noise_epsilon = 1e-3  # 基础保底噪声

        # 门控得分网络
        self.gate_score = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(input, output),
            nn.Dropout(0.2)
        )
        # 噪声权重网络
        self.noise_w = nn.Linear(input, output)

        self.last_prob = None
        self.last_mask = None
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x, input_feat, epoch_ratio=0.0):  # epoch_ratio: 当前训练进度 (0.0 ~ 1.0)
        # 1. 计算原始得分
        clean_scores = self.gate_score(x)

        # 2. 注入双重动态噪声 (仅训练模式)
        if self.training:
            # A: 全局进度衰减因子 (余弦退火)
            progress = torch.clamp(torch.tensor(epoch_ratio), 0.0, 1.0)
            decay_factor = 0.5 * (1.0 + torch.cos(progress * 3.14159265))

            # B: 局部自适应不确定性 (熵控制)
            clean_prob = F.softmax(clean_scores, dim=1)
            # 计算香农熵
            entropy = -torch.sum(clean_prob * torch.log(clean_prob + 1e-8), dim=1, keepdim=True)
            max_entropy = torch.log(torch.tensor(clean_scores.size(1), dtype=torch.float, device=x.device))
            # 计算自信度 (模型越自信，confidence 越接近 1.0)
            confidence = 1.0 - (entropy / max_entropy)
            # 极度自信时放大噪声，不自信时保持原样
            adaptive_scale = 1.0 + confidence * 2.0

            # 融合：总噪声 = 基础噪声 * 熵自适应系数 * 全局衰减因子
            base_noise_std = F.softplus(self.noise_w(x.mean(dim=-1))) + self.noise_epsilon
            final_noise_std = base_noise_std * adaptive_scale * decay_factor
            noise = torch.randn_like(clean_scores) * final_noise_std
            # noise = torch.randn_like(clean_scores) * base_noise_std

            scores = clean_scores + noise
        else:
            scores = clean_scores

        # 3. 归一化得分与 Top-K 过滤
        soft_prob = F.softmax(scores, dim=1)
        values, indices = torch.topk(scores, k=self.topk, dim=1)
        mask = torch.zeros_like(scores).scatter(1, indices, 1.0)

        # 4. 保存状态供 Loss（如负载均衡 Loss）使用
        self.last_prob = soft_prob
        self.last_mask = mask

        # 5. STE 梯度直通技巧
        hard_gate = (mask - soft_prob).detach() + soft_prob

        return input_feat * hard_gate.unsqueeze(-1)
