import torch
import torch.nn as nn
import torch.nn.functional as F

from models.backbones.cnn_feature_extractor import CNNFeatureExtractor
from models.backbones.vit_encoder import ViTEncoder
from models.heads.position_head import PositionHead
from models.heads.relation_head import RelationHead
from models.modules.position_encoding import PositionalEncoding
from models.modules.reconstruction import ReconstructionModule

class PuzzleSolver(nn.Module):
    def __init__(self, img_size=224, patch_size=16, num_classes=1000, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4., qkv_bias=True, qk_scale=None, drop_rate=0., attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm):
        super(PuzzleSolver, self).__init__()
        self.cnn_feature_extractor = CNNFeatureExtractor(input_channels=3, feature_dim=embed_dim, grid_size=img_size // patch_size, use_bn=True, dropout=0.1)
        self.patch_embed = ViTEncoder.PatchEmbed(img_size=img_size, patch_size=patch_size, in_channels=3, embed_dim=embed_dim)
        self.positional_encoding = PositionalEncoding(embed_dim=embed_dim)
        self.transformer_encoder = ViTEncoder(img_size=img_size, patch_size=patch_size, in_channels=3, num_classes=num_classes, embed_dim=embed_dim, depth=depth, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop_rate=drop_rate, attn_drop_rate=attn_drop_rate, drop_path_rate=drop_path_rate, norm_layer=norm_layer)
        self.position_head = PositionHead(input_dim=embed_dim, num_positions=img_size // patch_size * img_size // patch_size)
        self.relation_head = RelationHead(input_dim=embed_dim, num_relations=(img_size // patch_size) ** 2)
        self.reconstruction_module = ReconstructionModule(input_dim=embed_dim, output_dim=3, grid_size=img_size // patch_size)

    def forward(self, x):
        x = self.cnn_feature_extractor(x)
        x = x.view(x.size(0), -1)  # Ensure compatibility with ViT encoder
        x = self.patch_embed(x)
        x = self.positional_encoding(x)
        x = self.transformer_encoder(x)
        position_logits = self.position_head(x)
        relation_logits = self.relation_head(x)
        reconstructed_image = self.reconstruction_module(x)
        return position_logits, relation_logits, reconstructed_image
