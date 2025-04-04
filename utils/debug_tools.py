import torch
import numpy as np
import matplotlib.pyplot as plt
from torchviz import make_dot

def check_gradients(model, threshold=1e-5):
    """检查梯度是否过小或为零"""
    total_params = 0
    zero_grad_params = 0
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            total_params += 1
            if param.grad is None or torch.abs(param.grad).max() < threshold:
                zero_grad_params += 1
                print(f"警告: {name} 梯度接近零或为None")
    
    if total_params > 0:
        print(f"梯度检查: {zero_grad_params}/{total_params} 参数梯度接近零")
        return zero_grad_params / total_params
    return 0.0

def visualize_model_graph(model, dummy_input):
    """可视化模型计算图"""
    output = model(dummy_input)
    if isinstance(output, tuple):
        output = output[0]  # 取第一个输出
    
    dot = make_dot(output, params=dict(model.named_parameters()))
    dot.format = 'png'
    dot.render('model_graph')
    print("模型计算图已保存为 model_graph.png")