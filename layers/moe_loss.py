"""
MoE损失函数模块

功能：
- 实现MoE专家网络的平衡损失
- 实现MoE专家网络的稀疏性损失
- 实现MoE专家网络的多样性损失

作者：基于MoE多尺度特征融合设计
日期：2024
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MoELoss(nn.Module):
    """
    🔥 MoE损失函数模块
    
    实现MoE专家网络的三种损失函数：
    1. 平衡损失 (Balance Loss)：促进专家使用平衡，使用软 L2 惩罚。
    2. 稀疏性损失 (Sparsity Loss)：促进专家选择稀疏性，使用 Gini 不纯度。
    3. 多样性损失 (Diversity Loss)：促进专家分工多样性，保持不变。
    """
    
    def __init__(self, balance_weight=0.01, sparsity_weight=0.001, diversity_weight=0.01, balance_threshold=0.3):
        """
        初始化MoE损失函数
        """
        super(MoELoss, self).__init__()
        self.balance_weight = balance_weight
        self.sparsity_weight = sparsity_weight
        self.diversity_weight = diversity_weight
        self.balance_threshold = balance_threshold
        
    def balance_loss(self, expert_weights):
        """
        ✅ 最终修正：平衡损失（软 L2 惩罚）
        
        目的：只惩罚极端不平衡，使用 L2 范数限制极端情况下的梯度幅度。
        
        Args:
            expert_weights: [B, num_experts] - 专家权重分布（Softmax 输出）
        Returns:
            balance_loss: 平衡损失值
        """
        # 计算每个专家的平均使用频率
        expert_usage = expert_weights.mean(dim=0)  # [num_experts]
        num_experts = expert_usage.size(0)
        expected_usage = 1.0 / num_experts  # 期望使用频率 (e.g., 1/3 = 0.333)
        
        # 计算相对偏差：衡量实际使用频率与期望频率的偏离程度
        relative_deviation = torch.abs(expert_usage - expected_usage) / expected_usage
        
        # 🎯 核心修正：使用 relu(deviation - threshold).pow(2)
        # 1. relu(...) 只对超过阈值的偏差进行惩罚 (软平衡)
        # 2. .pow(2) 使用 L2 范数，有效限制了极端不平衡时 (如 1.3333) 梯度的爆炸性增长
        deviation_to_penalize = torch.relu(relative_deviation - self.balance_threshold)
        
        balance_loss = deviation_to_penalize.pow(2).mean()
        
        return balance_loss
    
    def sparsity_loss(self, expert_weights):
        """
        ✅ 最终修正：稀疏性损失（基于 Gini 不纯度/平方和）
        
        目的：惩罚均匀分布，奖励稀疏分布。
        原理：当权重稀疏 (如 [0, 1, 0]) 时，平方和等于 1 (最大值)；当权重均匀时 (如 [1/3, 1/3, 1/3])，平方和最小 (1/E)。
        
        Args:
            expert_weights: [B, num_experts] - 专家权重分布（已归一化）
        Returns:
            sparsity_loss: 稀疏性损失值（越小越稀疏）
        """
        # 计算每个样本的权重平方和
        squared_weights = expert_weights.pow(2).sum(dim=1)  # [B] - 每个样本的平方和
        
        # 修正：使用 (1.0 - squared_weights) / (1.0 - 1.0/num_experts) 将损失归一化到 [0, 1]
        num_experts = expert_weights.size(1)
        
        # 当 E=3 时，1.0 - 1.0/E = 2/3
        # 稀疏损失 = (1 - sum(w^2)) / (最大可能损失)
        max_possible_loss = 1.0 - (1.0 / num_experts)
        
        # 🔧 修复：使用浮点数比较，避免精度问题
        # 确保分母不为零，防止除法错误
        if abs(max_possible_loss) < 1e-8:
             # E=1 时（单专家），损失应为 0（因为只有一个专家，无法衡量稀疏性）
             return torch.zeros_like(squared_weights).mean()
             
        # 损失值范围：[0, 1]。0 表示完全稀疏，1 表示完全均匀。
        # 数学公式：L_sparsity = (1 - Σw²) / (1 - 1/E)
        sparsity_loss = ((1.0 - squared_weights) / max_possible_loss).mean()
        
        return sparsity_loss
    
    def diversity_loss(self, expert_weights):
        """
        多样性损失：保持不变 (原始实现正确)
        
        🔧 修复：调整执行顺序，先检查边界情况，避免不必要的计算
        """
        # 🔧 修复：先检查边界情况，避免不必要的计算
        num_experts = expert_weights.size(1)
        if num_experts <= 1:
            # 单专家或无专家时，无法衡量多样性，返回0
            return torch.tensor(0.0, device=expert_weights.device, dtype=expert_weights.dtype)
        
        # 计算专家权重之间的相关性
        # 使用L2归一化后的权重计算余弦相似度
        expert_weights_norm = F.normalize(expert_weights, p=2, dim=1)  # [B, num_experts]
        correlation_matrix = torch.mm(expert_weights_norm.t(), expert_weights_norm)  # [num_experts, num_experts]
        
        # 计算非对角线元素的平均值（专家间的平均相关性）
        # 数学公式：L_diversity = Σ_{i≠j} corr(i,j) / (E * (E-1))
        mask = 1 - torch.eye(num_experts, device=expert_weights.device)
        diversity_loss = (correlation_matrix * mask).sum() / (num_experts * (num_experts - 1))
        
        return diversity_loss
    
    def forward(self, expert_weights):
        """
        计算MoE总损失
        """
        # 计算各项损失
        balance_loss = self.balance_loss(expert_weights)
        sparsity_loss = self.sparsity_loss(expert_weights)
        diversity_loss = self.diversity_loss(expert_weights)
        
        # 加权求和
        total_loss = (self.balance_weight * balance_loss + 
                     self.sparsity_weight * sparsity_loss + 
                     self.diversity_weight * diversity_loss)
        
        # 返回损失字典
        loss_dict = {
            'moe_balance_loss': balance_loss.item(),
            'moe_sparsity_loss': sparsity_loss.item(),
            'moe_diversity_loss': diversity_loss.item(),
            'moe_total_loss': total_loss.item()
        }
        
        return total_loss, loss_dict


def make_moe_loss(cfg):
    """
    创建MoE损失函数
    """
    # 从配置文件读取MoE损失权重
    balance_weight = getattr(cfg.SOLVER, 'MOE_BALANCE_LOSS_WEIGHT', 0.01)
    sparsity_weight = getattr(cfg.SOLVER, 'MOE_SPARSITY_LOSS_WEIGHT', 0.001)
    diversity_weight = getattr(cfg.SOLVER, 'MOE_DIVERSITY_LOSS_WEIGHT', 0.01)
    balance_threshold = getattr(cfg.SOLVER, 'MOE_BALANCE_THRESHOLD', 0.3)
    
    # 创建MoE损失函数
    moe_loss = MoELoss(
        balance_weight=balance_weight,
        sparsity_weight=sparsity_weight,
        diversity_weight=diversity_weight,
        balance_threshold=balance_threshold
    )
    
    print(f"🔥 MoE损失函数初始化完成（已修复模式坍塌问题）:")
    print(f"   - 平衡损失权重: {balance_weight}")
    print(f"   - 稀疏性损失权重: {sparsity_weight}")
    print(f"   - 多样性损失权重: {diversity_weight}")
    print(f"   - 平衡损失阈值: {balance_threshold} (允许{balance_threshold*100:.0f}%偏差，防止强制平均)")
    
    return moe_loss