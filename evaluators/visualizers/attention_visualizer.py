import matplotlib.pyplot as plt
import torch
import numpy as np

class AttentionVisualizer:
    def __init__(self, model):
        self.model = model
        self.attention_maps = []
        
    def _attention_hook(self, module, input, output):
        """Hook to capture attention weights"""
        # Most attention implementations return attention weights as second item
        # or directly as attn property after forward pass
        if isinstance(output, tuple) and len(output) > 1:
            self.attention_maps.append(output[1])  # Usually (output, attention)
        elif hasattr(output, 'attentions'):
            self.attention_maps.append(output.attentions)
        elif hasattr(module, 'attn'):
            # Some implementations store attention in the module after forward
            self.attention_maps.append(module.attn)
            
    def extract_attention_weights(self, x, layer_idx=None):
        """Extract attention weights for specified layer(s)"""
        self.model.eval()
        self.attention_maps = []  # Reset stored maps
        hooks = []
        
        try:
            # Register hooks for specified layer(s)
            if hasattr(self.model, 'transformer_encoder') and hasattr(self.model.transformer_encoder, 'blocks'):
                blocks = self.model.transformer_encoder.blocks
                if layer_idx is None:
                    # Hook all layers
                    for block in blocks:
                        if hasattr(block, 'attn'):
                            hooks.append(block.attn.register_forward_hook(self._attention_hook))
                else:
                    # Hook specific layer
                    if 0 <= layer_idx < len(blocks) and hasattr(blocks[layer_idx], 'attn'):
                        hooks.append(blocks[layer_idx].attn.register_forward_hook(self._attention_hook))
            
            # Forward pass
            with torch.no_grad():
                _ = self.model(x)
                
            return self.attention_maps
        finally:
            # Ensure hooks are removed even if an error occurs
            for hook in hooks:
                hook.remove()
    
    def visualize_attention(self, x, head=0, layer_idx=-1, save_path=None, sample_idx=0):
        """Visualize attention map for a specific head and layer"""
        attention_weights = self.extract_attention_weights(x, layer_idx)
        
        if not attention_weights:
            print("No attention weights could be extracted. Check model compatibility.")
            return
        
        # Handle layer selection
        if layer_idx is None or layer_idx < 0:
            # Default to last layer if not specified or negative index
            attn = attention_weights[-1]
        elif layer_idx < len(attention_weights):
            attn = attention_weights[layer_idx]
        else:
            print(f"Layer index {layer_idx} out of range. Using last layer.")
            attn = attention_weights[-1]
        
        # Extract attention map for specified head
        if isinstance(attn, torch.Tensor):
            if len(attn.shape) == 4:  # [batch, heads, seq_len, seq_len]
                attention_map = attn[sample_idx, head].cpu().numpy()
            else:
                print(f"Unexpected attention tensor shape: {attn.shape}")
                return
        else:
            print(f"Unexpected attention type: {type(attn)}")
            return
            
        plt.figure(figsize=(8, 6))
        im = plt.imshow(attention_map, cmap='viridis')
        plt.colorbar(im)
        plt.title(f'Attention Map - Head {head}, Layer {layer_idx if layer_idx >= 0 else len(attention_weights)-1}')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        plt.show()
