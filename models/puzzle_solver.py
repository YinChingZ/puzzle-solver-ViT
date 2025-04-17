import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint

from models.backbones.cnn_feature_extractor import CNNFeatureExtractor
from models.backbones.vit_encoder import ViTEncoder, PatchEmbed
from models.heads.position_head import PositionHead
from models.heads.relation_head import RelationHead
from models.modules.position_encoding import PositionalEncoding
from models.modules.reconstruction import ReconstructionModule

class PuzzleSolver(nn.Module):
    def __init__(self, img_size=224, patch_size=16, num_classes=1000, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4., qkv_bias=True, qk_scale=None, drop_rate=0., attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm, grid_size=None):
        super(PuzzleSolver, self).__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = grid_size if grid_size is not None else img_size // patch_size
        
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
            grid_size=self.grid_size,
            img_size=img_size  # 传递图像尺寸
        )

        # 在__init__方法中添加
        self.use_checkpointing = True  # 启用梯度检查点

    # 修改 forward 方法

    def forward(self, x, return_features=False):
        # 检查输入格式并进行预处理
        if x.dim() > 4:  # 如果输入是拼图块格式
            # 获取批次大小和网格大小
            batch_size, num_patches = x.shape[0], x.shape[1]
            grid_size = int(num_patches ** 0.5)  # 假设是方形网格
            
            # 重新组合拼图块为一个完整图像
            if x.dim() == 5:  # [B, N, C, H, W]
                # 获取相关维度
                _, _, channels, patch_h, patch_w = x.shape
                
                # 创建输出张量
                img_h, img_w = grid_size * patch_h, grid_size * patch_w
                x_reshaped = torch.zeros(batch_size, channels, img_h, img_w, device=x.device)
                
                # 重新排列拼图块
                for b in range(batch_size):
                    for i in range(grid_size):
                        for j in range(grid_size):
                            patch_idx = i * grid_size + j
                            y_start = i * patch_h
                            y_end = (i+1) * patch_h
                            x_start = j * patch_w
                            x_end = (j+1) * patch_w
                            
                            x_reshaped[b, :, y_start:y_end, x_start:x_end] = x[b, patch_idx]
                
                # 更新x为重组后的图像
                x = x_reshaped
                # print(f"重组拼图块为完整图像: {x.shape}")
        
        # 原有的前向传播逻辑
        features = self.cnn_feature_extractor(x)
        
        # 使用梯度检查点优化内存使用
        if hasattr(self, 'use_checkpointing') and self.use_checkpointing:
            encoded_features = checkpoint.checkpoint(
                self.transformer_encoder, 
                features, 
                use_reentrant=False,
                # 添加以下参数以修复警告
                preserve_rng_state=True
            )
        else:
            encoded_features = self.transformer_encoder(features)
        
        # 获取原始的浮点 logits
        position_logits = self.position_head(encoded_features, return_indices=False)
        relation_logits = self.relation_head(encoded_features)
        
        # 打印更详细的形状信息来调试
        # print(f"Position logits detailed shape: {position_logits.shape}, type: {position_logits.dtype}")
        # print(f"Encoded features shape: {encoded_features.shape}")
        
        # 检查维度并适当处理
        if len(position_logits.shape) == 3:  # 如果是3D: [batch_size, seq_len, num_classes]
            # 正确处理3D张量
            expected_positions = self.grid_size * self.grid_size
            if position_logits.size(1) > expected_positions:
                # 只保留需要的位置
                position_logits = position_logits[:, :expected_positions, :]
        elif len(position_logits.shape) == 2:  # 如果是2D: [batch_size, seq_len*num_classes]
            # 重塑为预期的grid_size
            batch_size = position_logits.size(0)
            expected_positions = self.grid_size * self.grid_size
            if position_logits.size(1) != expected_positions:
                # 裁剪或填充到预期大小
                if position_logits.size(1) > expected_positions:
                    position_logits = position_logits[:, :expected_positions]
                else:
                    # 如果太小，填充
                    padding = torch.zeros(batch_size, expected_positions - position_logits.size(1), device=position_logits.device)
                    position_logits = torch.cat([position_logits, padding], dim=1)
        
        # 获取位置索引用于重建
        position_indices = self.position_head(encoded_features, return_indices=True)

        # 使用位置索引进行重建
        reconstructed_image, confidence = self.reconstruction_module(
            encoded_features, position_indices, relation_logits
        )

        # 根据参数决定是否返回特征
        if return_features:
            return position_logits, relation_logits, reconstructed_image, encoded_features
        else:
            return position_logits, relation_logits, reconstructed_image

    # 添加预训练模型加载功能
    def load_pretrained(self, pretrained_path=None):
        """加载预训练模型权重"""
        if pretrained_path is None:
            # 使用timm库加载预训练ViT
            import timm
            pretrained_vit = timm.create_model('vit_base_patch16_224', pretrained=True)
            
            # 复制适配层和部分transformer块权重
            with torch.no_grad():
                # 复制patch embedding权重
                if hasattr(self, 'patch_embed') and hasattr(pretrained_vit, 'patch_embed'):
                    self.patch_embed.proj.weight.copy_(pretrained_vit.patch_embed.proj.weight)
                    self.patch_embed.proj.bias.copy_(pretrained_vit.patch_embed.proj.bias)
                
                # 复制transformer块权重(适应深度差异)
                if hasattr(self, 'transformer') and hasattr(pretrained_vit, 'blocks'):
                    for i in range(min(len(self.transformer.blocks), len(pretrained_vit.blocks))):
                        # 复制自注意力权重
                        self.transformer.blocks[i].attn.qkv.weight.copy_(
                            pretrained_vit.blocks[i].attn.qkv.weight[:self.embed_dim*3, :self.embed_dim])
                        
                        # 复制MLP权重
                        self.transformer.blocks[i].mlp.fc1.weight.copy_(
                            pretrained_vit.blocks[i].mlp.fc1.weight[:self.embed_dim*4, :self.embed_dim])
                        
            print("成功加载预训练ViT权重")
        else:
            # 加载自定义预训练权重
            state_dict = torch.load(pretrained_path, map_location='cpu')
            if isinstance(state_dict, dict) and 'model' in state_dict:
                state_dict = state_dict['model']
            
            # 过滤不匹配的键
            model_state_dict = self.state_dict()
            filtered_dict = {k: v for k, v in state_dict.items() 
                            if k in model_state_dict and v.shape == model_state_dict[k].shape}
            
            incompatible_keys = self.load_state_dict(filtered_dict, strict=False)
            print(f"成功加载预训练权重: {len(filtered_dict)}/{len(state_dict)} 参数")
            print(f"不兼容的键: {incompatible_keys}")

