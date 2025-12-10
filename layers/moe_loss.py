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
    1. 平衡损失：促进专家使用平衡
    2. 稀疏性损失：促进专家选择稀疏性
    3. 多样性损失：促进专家分工多样性
    """
    
    def __init__(self, balance_weight=0.01, sparsity_weight=0.001, diversity_weight=0.01, balance_threshold=0.3):
        """
        初始化MoE损失函数
        
        Args:
            balance_weight (float): 平衡损失权重
            sparsity_weight (float): 稀疏性损失权重
            diversity_weight (float): 多样性损失权重
            balance_threshold (float): 平衡损失阈值，允许的偏差比例（默认0.3，即允许30%偏差）
        """
        super(MoELoss, self).__init__()
        self.balance_weight = balance_weight
        self.sparsity_weight = sparsity_weight
        self.diversity_weight = diversity_weight
        self.balance_threshold = balance_threshold
        
    def balance_loss(self, expert_weights):
        """
        平衡损失：促进专家使用平衡（改进版 - 防止模式坍塌）
        
        🔧 修复：使用"软平衡"而不是"硬平衡"
        - 不强制完全平均（33.3%, 33.3%, 33.3%）
        - 只惩罚极端不平衡（如 90%, 5%, 5%）
        - 允许合理的偏差（如 40%, 30%, 30%）
        
        Args:
            expert_weights: [B, num_experts] - 专家权重分布
        Returns:
            balance_loss: 平衡损失值
        """
        # 计算每个专家的平均使用频率
        expert_usage = expert_weights.mean(dim=0)  # [num_experts]
        num_experts = expert_usage.size(0)
        expected_usage = 1.0 / num_experts  # 期望使用频率（如 1/3 = 0.333）
        
        # 🔧 改进：使用相对偏差而不是绝对方差
        # 只惩罚超过阈值的偏差，允许合理的偏差
        relative_deviation = torch.abs(expert_usage - expected_usage) / expected_usage  # 相对偏差
        
        # 🔧 改进：使用阈值，只惩罚极端不平衡
        # 如果相对偏差 < threshold（即使用频率在合理范围内），不惩罚
        # 如果相对偏差 >= threshold，才进行惩罚
        penalty_mask = (relative_deviation > self.balance_threshold).float()
        
        # 只对超过阈值的偏差进行惩罚
        balance_loss = (relative_deviation * penalty_mask).mean()
        
        # 🔧 备选方案：使用平方惩罚，但只对极端情况
        # balance_loss = torch.clamp(relative_deviation - threshold, min=0.0).pow(2).mean()
        
        return balance_loss
    
    def sparsity_loss(self, expert_weights):
        """
        稀疏性损失：促进专家选择稀疏性
        
        Args:
            expert_weights: [B, num_experts] - 专家权重分布
        Returns:
            sparsity_loss: 稀疏性损失值
        """
        # 计算每个样本的专家选择稀疏性
        # 使用L1正则化促进稀疏性
        sparsity_loss = expert_weights.sum(dim=1).mean()
        
        return sparsity_loss
    
    def diversity_loss(self, expert_weights):
        """
        多样性损失：促进专家分工多样性
        
        Args:
            expert_weights: [B, num_experts] - 专家权重分布
        Returns:
            diversity_loss: 多样性损失值
        """
        # 计算专家权重之间的相关性
        # 使用余弦相似度计算专家之间的相关性
        expert_weights_norm = F.normalize(expert_weights, p=2, dim=1)
        correlation_matrix = torch.mm(expert_weights_norm.t(), expert_weights_norm)
        
        # 计算非对角线元素的和（相关性越高，多样性越低）
        num_experts = expert_weights.size(1)
        mask = 1 - torch.eye(num_experts, device=expert_weights.device)
        diversity_loss = (correlation_matrix * mask).sum() / (num_experts * (num_experts - 1))
        
        return diversity_loss
    
    def forward(self, expert_weights):
        """
        计算MoE总损失
        
        Args:
            expert_weights: [B, num_experts] - 专家权重分布
        Returns:
            total_loss: 总损失值
            loss_dict: 各项损失的详细信息
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
    
    Args:
        cfg: 配置对象
    Returns:
        moe_loss: MoE损失函数
    """
    # 从配置文件读取MoE损失权重
    balance_weight = getattr(cfg.SOLVER, 'MOE_BALANCE_LOSS_WEIGHT', 0.01)
    sparsity_weight = getattr(cfg.SOLVER, 'MOE_SPARSITY_LOSS_WEIGHT', 0.001)
    diversity_weight = getattr(cfg.SOLVER, 'MOE_DIVERSITY_LOSS_WEIGHT', 0.01)
    balance_threshold = getattr(cfg.SOLVER, 'MOE_BALANCE_THRESHOLD', 0.3)  # 🔧 新增：平衡损失阈值
    
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
