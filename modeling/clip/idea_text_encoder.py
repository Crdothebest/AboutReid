import torch
import torch.nn as nn
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

"""
IDEA项目的CLIP文本编码器实现
完全复制IDEA项目的文本编码逻辑
"""


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)

    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)


class TextEncoder(nn.Module):
    """
    IDEA项目的CLIP文本编码器
    完全复制IDEA项目中的实现
    """
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        """
        IDEA风格的文本编码前向传播

        Args:
            prompts: 文本嵌入 [batch_size, n_ctx, transformer.width]
            tokenized_prompts: 分词后的文本 [batch_size, n_ctx]

        Returns:
            torch.Tensor: 文本特征 [batch_size, transformer.width]
        """
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection
        return x


class IDEATextEncoder(nn.Module):
    """
    IDEA项目的完整文本编码器
    包含文本分词、嵌入和编码的完整流程
    """
    def __init__(self, clip_model):
        super().__init__()
        self.token_embedding = clip_model.token_embedding
        self.text_encoder = TextEncoder(clip_model)

    def forward(self, text, modality=None):
        """
        完整的文本编码流程

        Args:
            text: 输入文本 [batch_size, max_length]
            modality: 模态信息 (可选，用于未来扩展)

        Returns:
            torch.Tensor: 文本特征 [batch_size, embed_dim]
        """
        # Token embedding
        x = self.token_embedding(text).type(self.text_encoder.dtype)  # [batch_size, n_ctx, embed_dim]

        # 调用TextEncoder进行编码
        text_features = self.text_encoder(x, text)

        return text_features


def create_idea_text_encoder(clip_model):
    """
    创建IDEA风格的文本编码器

    Args:
        clip_model: 预加载的CLIP模型

    Returns:
        IDEATextEncoder: IDEA风格的文本编码器
    """
    return IDEATextEncoder(clip_model)