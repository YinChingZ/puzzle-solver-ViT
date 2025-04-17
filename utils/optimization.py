import torch
import torch.optim as optim

def get_optimizer(model, optimizer_name='adam', learning_rate=0.001, weight_decay=0.0001):
    if optimizer_name.lower() == 'adam':
        return optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_name.lower() == 'sgd':
        return optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=weight_decay)
    elif optimizer_name.lower() == 'adamw':
        return optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
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

def get_scheduler(optimizer, scheduler_name='cosine', **kwargs):
    """获取学习率调度器
    
    参数:
        optimizer: 优化器
        scheduler_name: 调度器名称
        **kwargs: 调度器特定参数
    """
    if scheduler_name.lower() == 'cosine' or scheduler_name.lower() == 'cosineannealinglr':
        return optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=kwargs.get('T_max', 50), 
            eta_min=kwargs.get('eta_min', 0)
        )
    elif scheduler_name.lower() == 'step':
        return optim.lr_scheduler.StepLR(
            optimizer, 
            step_size=kwargs.get('step_size', 30), 
            gamma=kwargs.get('gamma', 0.1)
        )
    elif scheduler_name.lower() == 'onecycle' or scheduler_name.lower() == 'onecyclelr':
        # 从 train_loader 和 epochs 计算 total_steps
        from torch.utils.data import DataLoader
        import math
        
        # 尝试从全局或传入参数中获取训练加载器和轮次
        train_loader = kwargs.get('train_loader', None)
        epochs = kwargs.get('epochs', None)
        total_steps = kwargs.get('total_steps', None)
        
        # 如果没有明确的total_steps，尝试计算
        if total_steps is None:
            if train_loader is not None and epochs is not None:
                steps_per_epoch = len(train_loader)
                total_steps = steps_per_epoch * epochs
            else:
                # 如果无法计算，提供默认值
                total_steps = 10000
                print(f"Warning: Using default total_steps={total_steps} for OneCycleLR")
        
        return optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=kwargs.get('max_lr', 0.001),
            total_steps=total_steps,
            pct_start=kwargs.get('pct_start', 0.3),
            div_factor=kwargs.get('div_factor', 25.0),
            final_div_factor=kwargs.get('final_div_factor', 1e4)
        )
    else:
        raise ValueError(f'Unsupported scheduler: {scheduler_name}')
