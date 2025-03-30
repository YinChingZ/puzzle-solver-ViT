import matplotlib.pyplot as plt
import torch

class AttentionVisualizer:
    def __init__(self, model):
        self.model = model

    def extract_attention_weights(self, x):
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(x)
            attention_weights = self.model.transformer_encoder.blocks[-1].attn.attn
        return attention_weights

    def visualize_attention(self, x, head=0, layer=-1):
        attention_weights = self.extract_attention_weights(x)
        attention_map = attention_weights[0, head].cpu().numpy()

        plt.imshow(attention_map, cmap='viridis')
        plt.colorbar()
        plt.title(f'Attention Map - Head {head}, Layer {layer}')
        plt.show()
