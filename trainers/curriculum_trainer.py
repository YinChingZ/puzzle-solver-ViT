import torch
import os
from torch.utils.data import DataLoader
from trainers.base_trainer import BaseTrainer

class CurriculumTrainer(BaseTrainer):
    def __init__(self, model, optimizer, criterion, train_loader, val_loader, num_epochs, device, difficulty_scheduler=None):
        # 提取 dataset 和 batch_size
        train_dataset = train_loader.dataset
        val_dataset = val_loader.dataset
        batch_size = train_loader.batch_size
        
        # 调用基类初始化
        super(CurriculumTrainer, self).__init__(
            model, optimizer, criterion, train_dataset, val_dataset, 
            batch_size, num_epochs, device
        )
        
        # 保存其他属性
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.difficulty_scheduler = difficulty_scheduler
        self.current_difficulty = 'easy'  # 默认难度

    def train(self):
        for epoch in range(self.num_epochs):
            if self.difficulty_scheduler:
                current_difficulty, current_grid_size = self.difficulty_scheduler.step(epoch)
                print(f"Epoch {epoch+1}: Training with difficulty {current_difficulty}, grid size {current_grid_size}")
            
            self.model.train()
            train_loss = 0.0
            
            for images, position_labels in self.train_loader:
                images = images.to(self.device)
                position_labels = position_labels.to(self.device)
                
                self.optimizer.zero_grad()
                
                position_logits, relation_logits, reconstructed_image = self.model(images)

                print(f"Position logits shape: {position_logits.shape}, dtype: {position_logits.dtype}")
                print(f"Position labels shape: {position_labels.shape}, dtype: {position_labels.dtype}")
                
                # ===== 关键修改：重新调整 logits 的形状 =====
                
                # 问题：logits 和 labels 都是 [batch_size, 16]
                # 标准 CrossEntropyLoss 需要:
                # - 输入: [batch_size, num_classes, ...] (对于 1D 目标)
                # - 目标: [batch_size, ...] (整型类别索引)
                
                # 解决方案：添加类别维度，需要根据您模型的具体设计调整
                num_classes = position_logits.shape[-1]  # 假设这是您的类别数
                batch_size = position_logits.shape[0]
                
                # 情形1: 如果 logits 应该表示每个样本每个位置的类别预测
                if len(position_logits.shape) == 2:  # [batch_size, num_positions]
                    # 获取每个位置的可能类别数
                    num_positions = position_logits.shape[1]
                    
                    # 调整 logits 和 labels 的形状
                    # 假设每个位置只预测一个类别
                    # 把 position_labels 变形为 [batch_size*num_positions]
                    position_labels_reshaped = position_labels.view(-1)
                    
                    # 假设 position_logits 应该有类别维度，但目前没有
                    # 创建一个包含类别维度的新张量
                    position_logits_with_classes = torch.zeros(
                        batch_size, num_positions, num_classes, 
                        device=position_logits.device, dtype=position_logits.dtype
                    )
                    
                    # 使用现有的 logits 值填充
                    for b in range(batch_size):
                        for p in range(num_positions):
                            # 这里只是用一个简单的示例，您可能需要根据实际情况调整
                            class_idx = min(int(position_logits[b, p].item()), num_classes-1)
                            position_logits_with_classes[b, p, class_idx] = 1.0
                    
                    # 重塑为 CrossEntropyLoss 需要的格式
                    position_logits_reshaped = position_logits_with_classes.view(batch_size*num_positions, num_classes)
                    
                    # 计算损失
                    position_loss = self.criterion(position_logits_reshaped, position_labels_reshaped)
                else:
                    # 如果形状不是预期的，输出错误
                    print(f"未处理的 position_logits 形状: {position_logits.shape}")
                    raise ValueError(f"无法处理形状为 {position_logits.shape} 的 position_logits")
                
                loss = position_loss
            
            val_loss = self.validate()
            print(f'Epoch [{epoch+1}/{self.num_epochs}], Train Loss: {train_loss/len(self.train_loader)}, Val Loss: {val_loss/len(self.val_loader)}')

    def validate(self):
        self.model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, position_labels in self.val_loader:
                images = images.to(self.device)
                position_labels = position_labels.to(self.device)
                
                position_logits, relation_logits, reconstructed_image = self.model(images)
                
                # 应用与训练方法相同的形状转换
                if position_logits.shape == position_labels.shape:
                    if position_logits.dim() == 3:
                        position_logits = position_logits.permute(0, 2, 1)
                
                position_logits = position_logits.float()
                position_loss = self.criterion(position_logits, position_labels)
                loss = position_loss
                
                val_loss += loss.item()
                
        return val_loss

    def save_checkpoint(self, checkpoint_path):
        state = {
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epoch': self.num_epochs
        }
        torch.save(state, checkpoint_path)

    def load_checkpoint(self, checkpoint_path):
        if os.path.isfile(checkpoint_path):
            state = torch.load(checkpoint_path)
            self.model.load_state_dict(state['model'])
            self.optimizer.load_state_dict(state['optimizer'])
            self.num_epochs = state['epoch']
        else:
            print(f'No checkpoint found at {checkpoint_path}')
