#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F

import math
from typing import Optional, List


class LoRALayer():
    def __init__(self, r: int, lora_alpha: int, lora_dropout: float, merge_weights: bool):
        self.r = r                          # 降低维度后的秩 r（低秩分解的维度）
        self.lora_alpha = lora_alpha        # LoRA 的缩放系数
        # dropout 用于在训练中随机丢弃输入，增强泛化
        if lora_dropout > 0.:
            self.lora_dropout = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout = lambda x: x  # 如果 dropout=0，则保持输入不变
        self.merged = False                  # 标记权重是否已经合并到原始权重
        self.merge_weights = merge_weights   # 是否在推理时合并 LoRA 权重



class Embedding(nn.Embedding, LoRALayer):
    # LoRA 实现在 Embedding 层

    def __init__(
            self,
            num_embeddings: int,     # 词表大小（embedding 矩阵行数）
            embedding_dim: int,      # 每个词的向量维度（embedding 矩阵列数）
            r: int = 0,              # LoRA 的秩（低秩矩阵的维度）
            lora_alpha: int = 1,     # LoRA 的缩放系数
            merge_weights: bool = True,  # 是否支持权重合并
            **kwargs
    ):
        nn.Embedding.__init__(self, num_embeddings, embedding_dim, **kwargs)
        # 初始化原始的 nn.Embedding (正常词嵌入矩阵)

        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=0,
                           merge_weights=merge_weights)
        # 初始化 LoRA 基础参数（秩、缩放因子、dropout等）

        if r > 0:
            # 如果 r > 0，说明启用了 LoRA
            self.lora_A = nn.Parameter(self.weight.new_zeros((r, num_embeddings)))
            # LoRA 矩阵 A: (r, vocab_size)，将 one-hot 投影到低秩空间

            self.lora_B = nn.Parameter(self.weight.new_zeros((embedding_dim, r)))
            # LoRA 矩阵 B: (embed_dim, r)，将低秩向量映射回 embedding 空间

            self.scaling = self.lora_alpha / self.r
            # 缩放因子，保持更新幅度与 alpha 成比例

            self.weight.requires_grad = False
            # 冻结原始 embedding 权重，只训练 LoRA 的 A 和 B

        self.reset_parameters()
        # 初始化参数（A、B 矩阵）

    def reset_parameters(self):
        nn.Embedding.reset_parameters(self)
        # 调用 nn.Embedding 的默认初始化

        if hasattr(self, 'lora_A'):
            nn.init.zeros_(self.lora_A)     # A 初始化为 0
            nn.init.normal_(self.lora_B)    # B 正态分布初始化（更适合学习）

    def train(self, mode: bool = True):
        nn.Embedding.train(self, mode)  # 调用父类的 train()
        if mode:
            if self.merge_weights and self.merged:
                # 如果在训练模式下，且之前合并过权重，需要先减去 LoRA 部分
                if self.r > 0:
                    self.weight.data -= (self.lora_B @ self.lora_A).transpose(0, 1) * self.scaling
                    # 还原原始权重
                self.merged = False
        else:
            if self.merge_weights and not self.merged:
                # 如果切换到 eval 模式，且还没合并，就把 LoRA 权重合进去
                if self.r > 0:
                    self.weight.data += (self.lora_B @ self.lora_A).transpose(0, 1) * self.scaling
                    # 合并 LoRA 增量到原始 embedding 矩阵
                self.merged = True

    def forward(self, x: torch.Tensor):
        if self.r > 0 and not self.merged:
            # 如果启用了 LoRA 且还没合并
            result = nn.Embedding.forward(self, x)  # 原始 embedding 查表

            after_A = F.embedding(
                x, self.lora_A.transpose(0, 1), self.padding_idx, self.max_norm,
                self.norm_type, self.scale_grad_by_freq, self.sparse
            )
            # 用 lora_A 做一次 embedding 查表，把 token 映射到低秩空间 (batch, seq_len, r)

            result += (after_A @ self.lora_B.transpose(0, 1)) * self.scaling
            # 将低秩向量乘 B 投影回 embedding 维度，并加到原始 embedding 上

            return result
        else:
            return nn.Embedding.forward(self, x)  # 没有 LoRA，直接走原始 embedding


class LoRA_Linear(nn.Linear, LoRALayer):
    # 在全连接层 (nn.Linear) 上实现 LoRA

    def __init__(
            self,
            in_features: int,          # 输入特征维度
            out_features: int,         # 输出特征维度
            r: int = 0,                # LoRA 的秩（低秩分解的维度）
            lora_alpha: int = 1,       # LoRA 的缩放因子
            lora_dropout: float = 0.,  # LoRA 输入 dropout，增强泛化
            fan_in_fan_out: bool = False,  # True 时，权重存储为 (in, out)，而不是 (out, in)
            merge_weights: bool = True,    # 是否允许合并 LoRA 权重
            **kwargs
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        # 初始化原始的全连接层（正常的 W: (out_features, in_features)）

        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                           merge_weights=merge_weights)
        # 初始化 LoRA 层的基础参数

        self.fan_in_fan_out = fan_in_fan_out

        if r > 0:
            # 如果启用了 LoRA（r>0）
            self.lora_A = nn.Parameter(self.weight.new_zeros((r, in_features)))
            # LoRA 矩阵 A: (r, in_dim)，先把输入降维到 r 维

            self.lora_B = nn.Parameter(self.weight.new_zeros((out_features, r)))
            # LoRA 矩阵 B: (out_dim, r)，再把 r 维映射回输出维度

            self.scaling = self.lora_alpha / self.r
            # 缩放因子，确保更新幅度与 lora_alpha 成比例

            # ⚠️ 注意：这里没有显式冻结 self.weight（原始权重），
            # 但通常实际应用里会在外部调用 mark_only_lora_as_trainable 来冻结

        self.reset_parameters()  # 初始化 LoRA 参数

    def reset_parameters(self):
        nn.Linear.reset_parameters(self)  # 调用 nn.Linear 的初始化方法
        if hasattr(self, 'lora_A'):
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))  # A 用 Kaiming 初始化
            nn.init.zeros_(self.lora_B)                            # B 全 0 初始化
            # 这样一开始 LoRA 的增量为 0，不会干扰原模型的输出

    def forward(self, x: torch.Tensor):
        # 注意：这里没有直接用原始 nn.Linear.forward(x)，
        # 而是只计算 LoRA 的增量部分

        result = (self.lora_dropout(x) @ self.lora_A.transpose(0, 1)
                  @ self.lora_B.transpose(0, 1)) * self.scaling
        # step1: 输入 x 先经过 dropout
        # step2: x @ A^T → 将输入从 in_dim 降到 r 维
        # step3: 再 @ B^T → 把 r 维映射回 out_dim
        # step4: 乘 scaling 缩放
        # 最终得到 LoRA 的增量 (delta_y)

        return result  # 返回的只是 LoRA 增量，而不是完整输出




class MergedLinear(nn.Linear, LoRALayer):
    # 在 nn.Linear 基础上实现的 LoRA，支持部分通道启用 LoRA

    def __init__(
            self,
            in_features: int,        # 输入维度
            out_features: int,       # 输出维度
            r: int = 0,              # LoRA 秩（低秩分解的维度）
            lora_alpha: int = 1,     # LoRA 缩放因子
            lora_dropout: float = 0.,# dropout 概率
            enable_lora: List[bool] = [False],  # 每个“分组”是否启用 LoRA（如多头注意力的不同 head）
            fan_in_fan_out: bool = False,       # 权重矩阵存储格式
            merge_weights: bool = True,         # 是否支持合并权重
            **kwargs
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        # 初始化标准 Linear 层

        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha,
                           lora_dropout=lora_dropout, merge_weights=merge_weights)

        assert out_features % len(enable_lora) == 0, \
            'enable_lora 的长度必须能整除 out_features'
        self.enable_lora = enable_lora
        self.fan_in_fan_out = fan_in_fan_out

        if r > 0 and any(enable_lora):  # 只有在启用了 LoRA 的情况下才初始化参数
            self.lora_A = nn.Parameter(
                self.weight.new_zeros((r * sum(enable_lora), in_features))
            )
            # A: (r * 有效组数, in_dim)，输入先降维到 r

            self.lora_B = nn.Parameter(
                self.weight.new_zeros((out_features // len(enable_lora) * sum(enable_lora), r))
            )
            # B: (每组输出维度 × 有效组数, r)，再映射回输出空间
            # 这里用分组 Conv1D 的方式实现多个 LoRA 组并行计算

            self.scaling = self.lora_alpha / self.r
            self.weight.requires_grad = False  # 冻结原始权重

            # 计算哪些输出通道启用了 LoRA
            self.lora_ind = self.weight.new_zeros(
                (out_features,), dtype=torch.bool
            ).view(len(enable_lora), -1)
            self.lora_ind[enable_lora, :] = True
            self.lora_ind = self.lora_ind.view(-1)
            # 得到一个布尔 mask，标记哪些输出通道有 LoRA

        self.reset_parameters()

        if fan_in_fan_out:
            self.weight.data = self.weight.data.transpose(0, 1)
            # 如果存储格式不同，进行转置

    def reset_parameters(self):
        nn.Linear.reset_parameters(self)
        if hasattr(self, 'lora_A'):
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))  # A 用 Kaiming 初始化
            nn.init.zeros_(self.lora_B)                            # B 全 0 初始化

    def zero_pad(self, x):
        # 将 LoRA 输出对齐到原始输出维度（未启用 LoRA 的通道用 0 填充）
        result = x.new_zeros((len(self.lora_ind), *x.shape[1:]))
        result[self.lora_ind] = x
        return result

    def merge_AB(self):
        # 计算 LoRA 权重增量 ΔW = B @ A
        def T(w):  # 根据 fan_in_fan_out 决定是否转置
            return w.transpose(0, 1) if self.fan_in_fan_out else w

        delta_w = F.conv1d(
            self.lora_A.unsqueeze(0),    # A 变形 (1, r*组数, in_dim)
            self.lora_B.unsqueeze(-1),   # B 变形 (out_dim, r, 1)
            groups=sum(self.enable_lora) # 分组卷积，模拟多个 LoRA 分组
        ).squeeze(0)
        return T(self.zero_pad(delta_w))  # 填充到完整维度

    def train(self, mode: bool = True):
        # 切换 train / eval 模式时，决定是否合并 LoRA 权重
        def T(w): return w.transpose(0, 1) if self.fan_in_fan_out else w

        nn.Linear.train(self, mode)
        if mode:  # 训练模式
            if self.merge_weights and self.merged:
                # 如果之前合并过权重，先减去 ΔW，还原原始权重
                if self.r > 0 and any(self.enable_lora):
                    self.weight.data -= self.merge_AB() * self.scaling
                self.merged = False
        else:     # 推理模式
            if self.merge_weights and not self.merged:
                # 如果还没合并过，把 ΔW 加到原始权重里，加速推理
                if self.r > 0 and any(self.enable_lora):
                    self.weight.data += self.merge_AB() * self.scaling
                self.merged = True

    def forward(self, x: torch.Tensor):
        def T(w): return w.transpose(0, 1) if self.fan_in_fan_out else w

        if self.merged:
            # 如果权重已经合并，直接用合并后的权重做一次普通 Linear
            return F.linear(x, T(self.weight), bias=self.bias)
        else:
            # 否则：原始输出 + LoRA 增量
            result = F.linear(x, T(self.weight), bias=self.bias)
            if self.r > 0:
                result += self.lora_dropout(x) @ T(self.merge_AB().T) * self.scaling
            return result



class ConvLoRA(nn.Module, LoRALayer):
    def __init__(self, conv_module, in_channels, out_channels, kernel_size, r=0, lora_alpha=1, lora_dropout=0.,
                 merge_weights=True, **kwargs):
        super(ConvLoRA, self).__init__()
        self.conv = conv_module(in_channels, out_channels, kernel_size, **kwargs)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout, merge_weights=merge_weights)
        assert isinstance(kernel_size, int)
        # Actual trainable parameters
        if r > 0:
            self.lora_A = nn.Parameter(
                self.conv.weight.new_zeros((r * kernel_size, in_channels * kernel_size))
            )
            self.lora_B = nn.Parameter(
                self.conv.weight.new_zeros((out_channels // self.conv.groups * kernel_size, r * kernel_size))
            )
            self.scaling = self.lora_alpha / self.r
            # Freezing the pre-trained weight matrix
            self.conv.weight.requires_grad = False
        self.reset_parameters()
        self.merged = False

    def reset_parameters(self):
        self.conv.reset_parameters()
        if hasattr(self, 'lora_A'):
            # initialize A the same way as the default for nn.Linear and B to zero
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

    def train(self, mode=True):
        super(ConvLoRA, self).train(mode)
        if mode:
            if self.merge_weights and self.merged:
                if self.r > 0:
                    # Make sure that the weights are not merged
                    self.conv.weight.data -= (self.lora_B @ self.lora_A).view(self.conv.weight.shape) * self.scaling
                self.merged = False
        else:
            if self.merge_weights and not self.merged:
                if self.r > 0:
                    # Merge the weights and mark it
                    self.conv.weight.data += (self.lora_B @ self.lora_A).view(self.conv.weight.shape) * self.scaling
                self.merged = True

    def forward(self, x):
        if self.r > 0 and not self.merged:
            return self.conv._conv_forward(
                x,
                self.conv.weight + (self.lora_B @ self.lora_A).view(self.conv.weight.shape) * self.scaling,
                self.conv.bias
            )
        return self.conv(x)


class Conv2d(ConvLoRA):
    def __init__(self, *args, **kwargs):
        super(Conv2d, self).__init__(nn.Conv2d, *args, **kwargs)


class Conv1d(ConvLoRA):
    def __init__(self, *args, **kwargs):
        super(Conv1d, self).__init__(nn.Conv1d, *args, **kwargs)


# Can Extend to other ones like this

class Conv3d(ConvLoRA):
    def __init__(self, *args, **kwargs):
        super(Conv3d, self).__init__(nn.Conv3d, *args, **kwargs)

#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import torch
import torch.nn as nn

from typing import Dict


def mark_only_lora_as_trainable(model: nn.Module, bias: str = 'none') -> None:
    for n, p in model.named_parameters():
        if 'adapter' not in n:
            p.requires_grad = False
    if bias == 'none':
        return
    elif bias == 'all':
        for n, p in model.named_parameters():
            if 'bias' in n:
                p.requires_grad = True
    elif bias == 'lora_only':
        for m in model.modules():
            if isinstance(m, LoRALayer) and \
                    hasattr(m, 'bias') and \
                    m.bias is not None:
                m.bias.requires_grad = True
    else:
        raise NotImplementedError


def lora_state_dict(model: nn.Module, bias: str = 'none') -> Dict[str, torch.Tensor]:
    my_state_dict = model.state_dict()
    if bias == 'none':
        return {k: my_state_dict[k] for k in my_state_dict if 'lora_' in k}
    elif bias == 'all':
        return {k: my_state_dict[k] for k in my_state_dict if 'lora_' in k or 'bias' in k}
    elif bias == 'lora_only':
        to_return = {}
        for k in my_state_dict:
            if 'lora_' in k:
                to_return[k] = my_state_dict[k]
                bias_name = k.split('lora_')[0] + 'bias'
                if bias_name in my_state_dict:
                    to_return[bias_name] = my_state_dict[bias_name]
        return to_return
    else:
        raise NotImplementedError
