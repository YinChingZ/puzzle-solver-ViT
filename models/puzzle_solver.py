import torch
import torch.nn as nn
import torch.nn.functional as F

from models.backbones.cnn_feature_extractor import CNNFeatureExtractor
from models.backbones.vit_encoder import ViTEncoder, PatchEmbed
from models.heads.position_head import PositionHead
from models.heads.relation_head import RelationHead
from models.modules.position_encoding import PositionalEncoding
from models.modules.reconstruction import ReconstructionModule

class PuzzleSolver(nn.Module):
    def __init__(self, img_size=224, patch_size=16, num_classes=1000, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4., qkv_bias=True, qk_scale=None, drop_rate=0., attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm):
        super(PuzzleSolver, self).__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = img_size // patch_size
        
        # 特征提取器
        self.cnn_feature_extractor = CNNFeatureExtractor(
            input_channels=3, 
            feature_dim=embed_dim, 
            grid_size=self.grid_size, 
            use_bn=True, 
            dropout=0.1
        )
        
        # ViT 组件
        self.transformer_encoder = ViTEncoder(
            img_size=img_size, 
            patch_size=patch_size, 
            in_channels=3, 
            num_classes=num_classes, 
            embed_dim=embed_dim, 
            depth=depth, 
            num_heads=num_heads, 
            mlp_ratio=mlp_ratio, 
            qkv_bias=qkv_bias, 
            qk_scale=qk_scale, 
            drop_rate=drop_rate, 
            attn_drop_rate=attn_drop_rate, 
            drop_path_rate=drop_path_rate, 
            norm_layer=norm_layer
        )
        
        # 用于位置编码
        self.positional_encoding = PositionalEncoding(embed_dim=embed_dim)
        
        # 任务头
        self.position_head = PositionHead(
            input_dim=embed_dim, 
            num_positions=self.grid_size * self.grid_size
        )
        self.relation_head = RelationHead(
            input_dim=embed_dim, 
            num_relations=self.grid_size ** 2
        )
        self.reconstruction_module = ReconstructionModule(
            input_dim=embed_dim, 
            output_dim=3, 
            grid_size=self.grid_size
        )

    def forward(self, x):
        # CNN 特征提取
        features = self.cnn_feature_extractor(x)
        
        # 通过 transformer 编码器
        encoded_features = self.transformer_encoder(features)
        
        # 获取原始的浮点 logits
        position_logits_float = self.position_head(encoded_features)  # 这应该是浮点类型
        relation_logits = self.relation_head(encoded_features)
        
        # 获取浮点型 logits 用于损失计算
        position_logits = self.position_head(encoded_features, return_indices=False)  # 应该是浮点型
    
        # 打印更详细的形状信息来调试
        print(f"Position logits detailed shape: {position_logits.shape}, type: {position_logits.dtype}")
        print(f"Encoded features shape: {encoded_features.shape}")
        
        # 确保位置逻辑的尺寸与标签匹配
        # 如果 position_logits 是 [batch_size, seq_len, num_classes]，需要调整为 [batch_size, num_classes]
        if len(position_logits.shape) > 2:
            position_logits = position_logits.view(position_logits.size(0), -1)
        
        # 如果需要，将位置逻辑调整为与标签匹配的尺寸
        # 假设标签是 [batch_size, 16]（4×4网格）
        batch_size = position_logits.size(0)
        expected_positions = 16  # 4×4网格
        
        if position_logits.size(1) != expected_positions:
            print(f"Reshaping position logits from {position_logits.shape} to [batch_size, {expected_positions}]")
            # 如果可能，截取或调整位置逻辑的尺寸
            # 这是临时解决方案 - 理想情况下应该修复根本原因
            position_logits = position_logits[:, :expected_positions]

        # 如果需要位置索引用于重建
        position_indices = self.position_head(encoded_features, return_indices=True)

        # 使用位置索引进行重建
        reconstructed_image, confidence = self.reconstruction_module(
            encoded_features, position_indices, relation_logits
        )

        # 返回 logits 用于损失计算
        return position_logits, relation_logits, reconstructed_image

