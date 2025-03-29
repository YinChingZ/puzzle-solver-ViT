import torch
import torch.nn as nn

def initialize_weights(model):
    """
    Initialize the weights of the model.
    """
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.01)
            nn.init.constant_(m.bias, 0)

def save_model(model, path):
    """
    Save the model to the specified path.
    """
    torch.save(model.state_dict(), path)

def load_model(model, path):
    """
    Load the model from the specified path.
    """
    model.load_state_dict(torch.load(path))
    return model

def count_parameters(model):
    """
    Count the number of parameters in the model.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def freeze_layers(model, layers_to_freeze):
    """
    Freeze the specified layers of the model.
    """
    for name, param in model.named_parameters():
        if any(layer in name for layer in layers_to_freeze):
            param.requires_grad = False

def unfreeze_layers(model, layers_to_unfreeze):
    """
    Unfreeze the specified layers of the model.
    """
    for name, param in model.named_parameters():
        if any(layer in name for layer in layers_to_unfreeze):
            param.requires_grad = True
