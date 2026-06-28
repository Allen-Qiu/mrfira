"""
搜索模块，计算query和passage多向量检索的相似度
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

# 该版本用Gate机制从Pvec抽取特征
# class SearchModule(nn.Module):
#     def __init__(self, num_m, num_n, d, dropout=0.1): # num_m是从查询query提取的特征数；num_n是从段落中提取的特征数
#         super(SearchModule, self).__init__()
#         self.num_m = num_m
#         self.num_n = num_n
#
#         self.extractor = self.get_decoder(self.num_m)
#         self.gates = nn.ModuleList([
#             nn.Sequential(
#                 # 1. 归一化：保证在大参数量下的训练稳定性
#                 nn.LayerNorm(1024),
#
#                 # 2. 升维投影：将容量扩展 2 倍 (1024 -> 2048)
#                 nn.Linear(1024, 2048),
#                 nn.SiLU(),
#                 nn.Dropout(dropout),
#
#                 # 3. 深度特征加工：增加一层同维度的非线性变换
#                 nn.Linear(2048, 2048),
#                 nn.SiLU(),
#                 nn.Dropout(dropout),
#
#                 # 4. 降维投影：映射回原始维度 (2048 -> 1024)
#                 nn.Linear(2048, 1024),
#
#                 # 5. 门控激活：输出 0-1 之间的权重
#                 nn.Sigmoid()
#             ) for _ in range(num_n)
#         ])
#
#         self.extractor.apply(self.extractor_weights_init)
#
#         for gate in self.gates:
#             for m in gate.modules():
#                 if isinstance(m, nn.Linear):
#                     nn.init.xavier_uniform_(m.weight)
#                     if m.bias is not None:
#                         nn.init.constant_(m.bias, 0)
#
#
#     def forward(self, qvec, pvec=None):
#         # qvec: (B, 1, d) -> 假设经过 tconv 和 conv 后变为 (B, m, d)
#         M = self.extractor(qvec)
#
#         results = []
#         for i in range(self.num_n):
#             res = self.gates[i](qvec)
#             results.append(res)
#         G = torch.cat(results, dim=1)  # shape: (B, n, d)
#         R = M.unsqueeze(2) * G.unsqueeze(1)
#
#         if pvec is None:
#             return None, R
#
#         # 1. 准备 P特征 (针对 pvec)
#         # pvec 原本是 (B, 1, d)，扩展为 (B, n, d)
#         P = pvec.repeat(1, self.num_n, 1)  # shape: (B, n, d)
#         T = P * G  # shape: (B, n, d)
#
#         # 2. 计算跨 Batch 的相似度矩阵 S
#         # M 是 (B, m, d), T 是 (B, n, d)
#         # 我们想要的结果是 (B_q, B_p, m, n)
#         # 使用 einsum 是最清晰的方法：
#         # b: batch_q, p: batch_p, m: m_dim, n: n_dim, d: feature_dim
#         S = torch.einsum('bmd, pnd -> bpmn', M, T)  # shape: (B, B, m, n)
#
#         # 3. 在 m, n 维度上进行 Max-Mean 操作
#         # 对每个 query 词，找到段落中最相似的词 (Max)
#         # S 的最后两个维度分别是 m 和 n
#         max_vals = torch.max(S, dim=-1).values  # shape: (B, B, m)
#         sim = torch.mean(max_vals, dim=-1)  # shape: (B, B)
#
#         return sim, R
#
#
#     def get_decoder(self, m, dropout=0.2):
#         # 定义中间通道数，通过增加 hidden_channels 来提升 Capacity
#         # 建议设为 m 的 2-4 倍
#         mid_channels = m * 4
#
#         return nn.Sequential(
#             # --- 第一层：大幅度扩张通道 ---
#             # kernel_size=5 增加感受野，padding=2 保证长度 d 不变
#             nn.ConvTranspose1d(1, mid_channels, kernel_size=5, stride=1, padding=2),
#             nn.BatchNorm1d(mid_channels),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Dropout(dropout),
#
#             # --- 第二层：深层转置卷积（精炼特征） ---
#             # 再次使用转置卷积增加非线性映射深度
#             nn.ConvTranspose1d(mid_channels, mid_channels, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm1d(mid_channels),
#             nn.LeakyReLU(0.2, inplace=True),
#
#             # --- 第三层：跨通道局部整合 ---
#             # 使用普通卷积进行特征融合，进一步提升模型对特征的提取能力
#             nn.Conv1d(mid_channels, mid_channels // 2, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm1d(mid_channels // 2),
#             nn.ReLU(inplace=True),
#             nn.Dropout(dropout),
#
#             # --- 第四层：最终映射层 ---
#             # 将通道数平滑地压缩到目标 m
#             nn.Conv1d(mid_channels // 2, m, kernel_size=1)
#     )
#
#     def extractor_weights_init(self, m):
#         classname = m.__class__.__name__
#
#         # 处理卷积层和转置卷积层
#         if 'Conv' in classname:
#             # 使用 Kaiming 正态分布，适合 LeakyReLU
#             nn.init.kaiming_normal_(m.weight.data, a=0.2, mode='fan_in', nonlinearity='leaky_relu')
#             if m.bias is not None:
#                 nn.init.constant_(m.bias.data, 0)
#
#         # 处理 BatchNorm 层
#         elif 'BatchNorm' in classname:
#             nn.init.constant_(m.weight.data, 1.0)
#             nn.init.constant_(m.bias.data, 0.0)

# 没有在pvec上抽取特征，适合searching阶段
# class SearchModule2(nn.Module):
#     def __init__(self, num_m, num_n, d, dropout=0.1): # num_m是从查询query提取的特征数；num_n是从段落中提取的特征数
#         super(SearchModule2, self).__init__()
#         self.num_m = num_m
#         self.num_n = num_n
#
#         self.extractor = self.get_decoder(self.num_m)
#         self.extractor.apply(self.extractor_weights_init)
#
#     def forward(self, qvec, pvec=None):
#         # qvec: (B, 1, d) -> 假设经过 tconv 和 conv 后变为 (B, m, d)
#         M = self.extractor(qvec)
#         R = M[0]
#         if pvec is None:
#             return None, R
#
#         # 1. 准备 P特征 (针对 pvec)
#         # pvec 原本是 (B, 1, d)，扩展为 (B, n, d)
#         P = pvec
#         # P = pvec.repeat(1, self.num_n, 1)  # shape: (B, n, d)
#         # T = P * G  # shape: (B, n, d)
#
#         # 2. 计算跨 Batch 的相似度矩阵 S
#         # M 是 (B, m, d), T 是 (B, n, d)
#         # 我们想要的结果是 (B_q, B_p, m, n)
#         # 使用 einsum 是最清晰的方法：
#         # b: batch_q, p: batch_p, m: m_dim, n: n_dim, d: feature_dim
#         S = torch.einsum('bmd, pnd -> bpmn', M, P)  # shape: (B, B, m, n)
#
#         # 3. 在 m, n 维度上进行 Max-Mean 操作
#         # 对每个 query 词，找到段落中最相似的词 (Max)
#         # S 的最后两个维度分别是 m 和 n
#         max_vals = torch.max(S, dim=-1).values  # shape: (B, B, m)
#         sim = torch.mean(max_vals, dim=-1)  # shape: (B, B)
#
#         return sim, R
#
#
#     def get_decoder(self, m, dropout=0.2):
#         # 定义中间通道数，通过增加 hidden_channels 来提升 Capacity
#         # 建议设为 m 的 2-4 倍
#         mid_channels = m * 4
#
#         return nn.Sequential(
#             # --- 第一层：大幅度扩张通道 ---
#             # kernel_size=5 增加感受野，padding=2 保证长度 d 不变
#             nn.ConvTranspose1d(1, mid_channels, kernel_size=5, stride=1, padding=2),
#             nn.BatchNorm1d(mid_channels),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Dropout(dropout),
#
#             # --- 第二层：深层转置卷积（精炼特征） ---
#             # 再次使用转置卷积增加非线性映射深度
#             nn.ConvTranspose1d(mid_channels, mid_channels, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm1d(mid_channels),
#             nn.LeakyReLU(0.2, inplace=True),
#
#             # --- 第三层：跨通道局部整合 ---
#             # 使用普通卷积进行特征融合，进一步提升模型对特征的提取能力
#             nn.Conv1d(mid_channels, mid_channels // 2, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm1d(mid_channels // 2),
#             nn.ReLU(inplace=True),
#             nn.Dropout(dropout),
#
#             # --- 第四层：最终映射层 ---
#             # 将通道数平滑地压缩到目标 m
#             nn.Conv1d(mid_channels // 2, m, kernel_size=1)
#     )
#
#     def extractor_weights_init(self, m):
#         classname = m.__class__.__name__
#
#         # 处理卷积层和转置卷积层
#         if 'Conv' in classname:
#             # 使用 Kaiming 正态分布，适合 LeakyReLU
#             nn.init.kaiming_normal_(m.weight.data, a=0.2, mode='fan_in', nonlinearity='leaky_relu')
#             if m.bias is not None:
#                 nn.init.constant_(m.bias.data, 0)
#
#         # 处理 BatchNorm 层
#         elif 'BatchNorm' in classname:
#             nn.init.constant_(m.weight.data, 1.0)
#             nn.init.constant_(m.bias.data, 0.0)

# passge也使用Tconv+conv来抽取特征，即做成Reranker
class SearchModule3(nn.Module):
    def __init__(self, total, num_m, num_n): # total是初始query和passagr分别总共要提取的特征数，num_m是最终从查询query选择的特征数；num_n是从段落中选择的特征数
        super(SearchModule3, self).__init__()
        self.num_m = num_m
        self.num_n = num_n

        self.q_extractor = self.get_extractor(total)
        self.p_extractor = self.get_extractor(total)
        self.q_gate = self.get_gating_module(total, total, self.num_m)
        self.p_gate = self.get_gating_module(total, total, self.num_n)
        self.q_extractor.apply(self.extractor_weights_init)
        self.p_extractor.apply(self.extractor_weights_init)

    def forward(self, qvec, pvec):
        # qvec: (1, 1, d) -> 假设经过 tconv 和 conv 后变为 (1, m, d)
        Q_feature = self.q_extractor(qvec) # 1,m,d
        Q = self.q_gate(Q_feature, Q_feature)
        # pvec: (B, 1, d) -> 假设经过 tconv 和 conv 后变为 (B, n, d)
        P_feature = self.p_extractor(pvec) # B,n,d
        Q_repeat = Q_feature.repeat(P_feature.shape[0], 1, 1)
        P_cat = torch.cat((Q_repeat, P_feature), dim=2)
        P = self.p_gate(P_cat, P_feature)   # P_cat用于产生gate

        # 2. 计算跨 Batch 的相似度矩阵 S
        # M 是 (B, m, d), T 是 (B, n, d)
        # 我们想要的结果是 (B_q, B_p, m, n)
        # 使用 einsum 是最清晰的方法：
        # b: batch_q, p: batch_p, m: m_dim, n: n_dim, d: feature_dim
        S = torch.einsum('bmd, pnd -> bpmn', Q, P)  # shape: (B, B, m, n)

        # 3. 在 m, n 维度上进行 Max-Mean 操作
        # 对每个 query 词，找到段落中最相似的词 (Max)
        # S 的最后两个维度分别是 m 和 n
        max_vals = torch.max(S, dim=-1).values  # shape: (B, B, m)
        # sim = torch.mean(max_vals, dim=-1)  # shape: (B, B)
        sim = torch.sum(max_vals, dim=-1)
        return sim, None


    def get_extractor(self, total, dropout=0.1):
        # 定义中间通道数，通过增加 hidden_channels 来提升 Capacity
        # 建议设为 m 的 2-4 倍
        mid_channels = total * 4

        return nn.Sequential(
            # --- 第一层：大幅度扩张通道 ---
            # kernel_size=5 增加感受野，padding=2 保证长度 d 不变
            nn.ConvTranspose1d(1, mid_channels, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(mid_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(dropout),

            # --- 第二层：深层转置卷积（精炼特征） ---
            # 再次使用转置卷积增加非线性映射深度
            nn.ConvTranspose1d(mid_channels, mid_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(mid_channels),
            nn.LeakyReLU(0.2, inplace=True),

            # --- 第三层：跨通道局部整合 ---
            # 使用普通卷积进行特征融合，进一步提升模型对特征的提取能力
            nn.Conv1d(mid_channels, mid_channels // 2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(mid_channels // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),

            # --- 第四层：最终映射层 ---
            # 将通道数平滑地压缩到目标 m
            nn.Conv1d(mid_channels // 2, total, kernel_size=1),

    )

    # gating network
    def get_gating_module(self, input, output, topk):
        # return HardGatingModule(input, output, topk=topk)
        return NoisyHardGatingModule(input, output, topk)

    def extractor_weights_init(self, m):
        classname = m.__class__.__name__

        # 处理卷积层和转置卷积层
        if 'Conv' in classname:
            # 使用 Kaiming 正态分布，适合 LeakyReLU
            nn.init.kaiming_normal_(m.weight.data, a=0.2, mode='fan_in', nonlinearity='leaky_relu')
            if m.bias is not None:
                nn.init.constant_(m.bias.data, 0)

        # 处理 BatchNorm 层
        elif 'BatchNorm' in classname:
            nn.init.constant_(m.weight.data, 1.0)
            nn.init.constant_(m.bias.data, 0.0)


# 基于转置卷积的特征抽取方案
class ConvTransposeExtractor(nn.Module):
    def __init__(self, total, dropout=0.1):
        super().__init__()

        # 建议设为 m (total) 的 2-4 倍，提升 Capacity
        mid_channels = total * 4

        # 将原先的 nn.Sequential 整合为类内部的 network 属性
        self.network = nn.Sequential(
            # --- 第一层：大幅度扩张通道 ---
            # 输入 x 形状预期为 [B, 1, input_dim]
            nn.ConvTranspose1d(1, mid_channels, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(mid_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(dropout),

            # --- 第二层：深层转置卷积（精炼特征） ---
            nn.ConvTranspose1d(mid_channels, mid_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(mid_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(dropout),

            # --- 第三层：跨通道局部整合 ---
            nn.Conv1d(mid_channels, mid_channels // 2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(mid_channels // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),

            # --- 第四层：最终映射层 ---
            # 最终输出通道数为 total，形状为 [B, total, output_dim]
            nn.Conv1d(mid_channels // 2, total, kernel_size=1),
            nn.BatchNorm1d(total),
            nn.Tanh(),
            nn.Dropout(dropout)
        )

        # 用于缓存前向传播的输出，方便计算正交损失
        self._current_output = None

    def forward(self, x):
        """
        x 的输入形状可以是:
        - [B, 1, input_dim] (标准 Conv1d 格式)
        - [B, input_dim] (如果是扁平化向量，在内部自动 unsqueeze)
        """
        if x.dim() == 2:
            x = x.unsqueeze(1)

        out = self.network(x)

        # 缓存输出用于计算损失
        self._current_output = out
        return out

    def get_orthogonal_loss(self):
        """
        计算抽取的 total 个特征向量之间的正交性损失。
        期望同一 Batch 内，不同的特征通道之间相互正交。
        """
        if self._current_output is None:
            raise RuntimeError("请先执行 forward 前向传播，再计算正交损失。")

        # self._current_output 形状: [B, total, Dim]
        features = self._current_output
        batch_size, total_features, feat_dim = features.shape

        # 1. 归一化特征向量，确保计算的是余弦相似度
        # 对最后一个维度（特征维度）进行 L2 归一化
        features_norm = F.normalize(features, p=2, dim=-1)

        # 2. 计算每个 Batch 内部的特征相关性矩阵
        # [B, total, Dim] @ [B, Dim, total] -> [B, total, total]
        correlation_matrix = torch.bmm(features_norm, features_norm.transpose(1, 2))

        # 3. 创建单位矩阵
        identity = torch.eye(total_features, device=features.device)
        # 扩展到和 correlation_matrix 一样的 batch 维度 -> [B, total, total]
        identity = identity.unsqueeze(0).expand(batch_size, -1, -1)

        # 4. 计算与单位矩阵的差距（只惩罚非对角线部分，即不同特征之间的相关性）
        loss = torch.mean((correlation_matrix - identity) ** 2)

        return loss


# 卷积 + Transformer 混合多向量特征抽取方案(这一方案还每调通，可以看看学习率等是不是要降低)
class ConvTransformerExtractor(nn.Module):
    def __init__(self, total, dropout=0.1):
        """
        total: 目标生成的向量数量 (N)
        """
        super().__init__()
        self.total = total

        # --- 1. 前端：轻量化转置卷积生成器 (负责快速维度切分与通道扩充) ---
        # 这里的输入 input_dim 预期为 1024 维
        self.conv_generator = nn.Sequential(
            # 第一层：将通道从 1 扩展到 total * 2
            nn.ConvTranspose1d(1, total * 2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(total * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(dropout),

            # 第二层：精准映射到目标通道数 total，输出形状: [B, total, 1024]
            nn.ConvTranspose1d(total * 2, total, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(total),
        )

        # --- 2. 中端：Transformer 交互探针精炼器 (消除向量间的冗余，填补 Recall 空白) ---
        # total 在这里作为序列长度 (Sequence Length)，特征维度固定为 1024
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=1024,
            nhead=8,
            dim_feedforward=2048,
            dropout=dropout,
            activation='gelu',
            batch_first=True  # 确保直接接收和输出 [B, total, 1024]
        )
        self.transformer_refiner = nn.TransformerEncoder(encoder_layer, num_layers=2)

        # --- 3. 后端：最终映射与规范层 ---
        self.output_layer = nn.Sequential(
            nn.Tanh(),
            nn.Dropout(dropout)
        )

        # 用于缓存前向传播的输出，方便外部调用 get_orthogonal_loss
        self._current_output = None

    def forward(self, x):
        """
        输入格式兼容：支持 [B, 1, 1024] 或 [B, 1024]
        """
        if x.dim() == 2:
            x = x.unsqueeze(1)

        # 步骤 1：通过转置卷积，快速切分出 total 个向量的雏形
        cnn_features = self.conv_generator(x)

        # 步骤 2：直接送入 Transformer 进行多向量间的全局自注意力交互，进行去冗余和精炼
        refined_features = self.transformer_refiner(cnn_features)

        # 步骤 3：最终 Tanh 规范化
        out = self.output_layer(refined_features)

        # 缓存输出，输出形状完全适配原系统: [B, total, 1024]
        self._current_output = out
        return out

    def get_orthogonal_loss(self):
        """
        计算抽取的 total 个特征向量之间的正交性损失。
        """
        if self._current_output is None:
            raise RuntimeError("请先执行 forward 前向传播，再计算正交损失。")

        features = self._current_output
        batch_size, total_features, feat_dim = features.shape

        # 1. 对最后一个维度（特征维度）进行 L2 归一化
        features_norm = F.normalize(features, p=2, dim=-1)

        # 2. 计算每个 Batch 内部的特征相关性矩阵
        correlation_matrix = torch.bmm(features_norm, features_norm.transpose(1, 2))

        # 3. 创建单位矩阵并扩展到 batch 维度
        identity = torch.eye(total_features, device=features.device)
        identity = identity.unsqueeze(0).expand(batch_size, -1, -1)

        # 4. 计算与单位矩阵的差距
        loss = torch.mean((correlation_matrix - identity) ** 2)

        return loss

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


if __name__ == '__main__':
    # checkpoint = encoder = '/root/autodl-tmp/bge-m3'
    # encoder = TextEncoder(checkpoint)
    # query = ["hello world"]
    # qvec = encoder.get_embedding(query)
    #
    # sentences = ["What is BGE M3?", "Defination of BM25"]
    # pvecs = encoder.get_embedding(sentences)
    #
    # model = SearchModule(3, 5, qvec.shape[0])
    # res  = model(qvec, pvecs[0])
    # print(res.shape)
    # print(res)

    pass
