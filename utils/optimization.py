import torch
import torch.optim as optim

def get_optimizer(model, optimizer_name='adam', learning_rate=0.001, weight_decay=0.0001):
    if optimizer_name.lower() == 'adam':
        return optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_name.lower() == 'sgd':
        return optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=weight_decay)
    else:
        raise ValueError(f'Unsupported optimizer: {optimizer_name}')

def adjust_learning_rate(optimizer, epoch, initial_lr, lr_decay_epoch=30):
    """Sets the learning rate to the initial LR decayed by 10 every lr_decay_epoch epochs"""
    lr = initial_lr * (0.1 ** (epoch // lr_decay_epoch))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def warmup_learning_rate(optimizer, warmup_epochs, initial_lr, current_epoch):
    """Warmup learning rate for the first few epochs"""
    if current_epoch < warmup_epochs:
        lr = initial_lr * (current_epoch / warmup_epochs)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

def get_scheduler(optimizer, scheduler_name='cosine', T_max=50, eta_min=0):
    if scheduler_name.lower() == 'cosine':
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max, eta_min=eta_min)
    elif scheduler_name.lower() == 'step':
        return optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    else:
        raise ValueError(f'Unsupported scheduler: {scheduler_name}')
