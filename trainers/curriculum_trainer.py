import torch
import os
from torch.utils.data import DataLoader
from trainers.base_trainer import BaseTrainer
import torch.nn.functional as F
import math
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import time
from torch.cuda.amp import autocast, GradScaler

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

        # 添加对比损失
        self.contrastive_criterion = ContrastivePuzzleLoss(temperature=0.07)

        # 添加TensorBoard
        self.writer = SummaryWriter(log_dir=f"logs/tensorboard/{time.strftime('%Y%m%d-%H%M%S')}")

        self.best_val_loss = float('inf')
        self.patience = 5
        self.patience_counter = 0

        # 添加混合精度训练支持
        self.scaler = GradScaler()
        self.use_amp = True  # 使用混合精度

    def train(self, epoch=None):
        # 处理 epoch 为 None 的情况
        if epoch is None:
            epoch = getattr(self, 'current_epoch', 0)
            self.current_epoch = epoch + 1
        
        # 根据课程安排获取当前难度和网格大小
        current_difficulty = 'easy'
        current_grid_size = 4
        
        if self.difficulty_scheduler:
            for stage in reversed(self.difficulty_scheduler):
                if epoch >= stage['epoch']:
                    current_difficulty = stage['difficulty']
                    current_grid_size = stage['grid_size']
                    break
            
            print(f"Epoch {epoch+1}: 训练难度 {current_difficulty}, 网格大小 {current_grid_size}")
        
        # 更新数据集的网格大小参数
        if hasattr(self.train_dataset, 'set_grid_size'):
            self.train_dataset.set_grid_size(current_grid_size)
        
        # 重置数据加载器，使用新的网格大小
        self.train_loader = DataLoader(
            self.train_dataset, 
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        
        self.model.train()
        self.train_loss = 0.0
        
        # 使用tqdm进度条跟踪训练
        pbar = tqdm(self.train_loader)
        
        data_time = 0
        batch_time = 0
        end = time.time()
        
        for batch_idx, (images, positions) in enumerate(pbar):
            data_time += time.time() - end  # 记录数据加载时间
            
            images = images.to(self.device)
            positions = positions.to(self.device)
            
            # 使用混合精度训练
            self.optimizer.zero_grad()
            
            if self.use_amp:
                with autocast():
                    position_logits, relation_logits, reconstructed, features = self.model(images, return_features=True)
                    
                    # 计算主要损失
                    position_loss = self.criterion(position_logits, positions)
                    
                    # 计算对比损失
                    contrastive_loss = self.contrastive_criterion(features, positions)
                    
                    # 计算相邻块预测损失
                    adjacent_loss = self._compute_adjacent_loss(position_logits, positions)
                    
                    # 可选：重建损失
                    recon_loss = 0.0
                    if isinstance(reconstructed, torch.Tensor) and reconstructed.dim() > 3:
                        recon_loss = F.mse_loss(reconstructed, images)
                    
                    # 综合损失
                    loss = position_loss + 0.5 * contrastive_loss + 0.2 * adjacent_loss + 0.1 * recon_loss
                
                # 使用缩放器处理梯度
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                # 原始精度训练代码
                position_logits, relation_logits, reconstructed, features = self.model(images, return_features=True)
                # 计算主要损失
                position_loss = self.criterion(position_logits, positions)
                
                # 计算对比损失
                contrastive_loss = self.contrastive_criterion(features, positions)
                
                # 计算相邻块预测损失
                adjacent_loss = self._compute_adjacent_loss(position_logits, positions)
                
                # 可选：重建损失
                recon_loss = 0.0
                if isinstance(reconstructed, torch.Tensor) and reconstructed.dim() > 3:
                    recon_loss = F.mse_loss(reconstructed, images)
                
                # 综合损失
                loss = position_loss + 0.5 * contrastive_loss + 0.2 * adjacent_loss + 0.1 * recon_loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
            
            # 更新累计损失和进度条
            self.train_loss += loss.item()
            pbar.set_description(f"训练损失: {loss.item():.4f}")

            # 记录损失组件
            self.writer.add_scalar('Loss/train/total', loss.item(), epoch)
            self.writer.add_scalar('Loss/train/position', position_loss.item(), epoch)
            self.writer.add_scalar('Loss/train/contrastive', contrastive_loss.item(), epoch)
            self.writer.add_scalar('Loss/train/adjacent', adjacent_loss.item(), epoch)
            self.writer.add_scalar('Loss/train/reconstruction', recon_loss, epoch)
            
            # 记录梯度统计
            if batch_idx % 50 == 0:
                for name, param in self.model.named_parameters():
                    if param.requires_grad and param.grad is not None:
                        self.writer.add_histogram(f'Gradients/{name}', param.grad, epoch)
            
            batch_time += time.time() - end  # 记录总批处理时间
            end = time.time()
            
            if batch_idx % 10 == 0:
                print(f"Epoch: {epoch} [{batch_idx}/{len(self.train_loader)}] "
                      f"Data: {data_time/(batch_idx+1):.3f}s "
                      f"Batch: {batch_time/(batch_idx+1):.3f}s "
                      f"DataRatio: {data_time/batch_time*100:.1f}%")
        
        self.train_loss /= len(self.train_loader)
        return self.train_loss

    def _compute_adjacent_loss(self, position_logits, positions):
        """计算相邻块相似性损失"""
        # 获取预测概率
        position_probs = F.softmax(position_logits, dim=1)
        
        batch_size = positions.size(0)
        grid_size = int(math.sqrt(position_probs.size(1)))
        loss = 0.0
        
        # 对每个样本计算相邻块损失
        for i in range(batch_size):
            pos_idx = positions[i]
            
            # 获取真实位置的行列索引
            rows = pos_idx // grid_size
            cols = pos_idx % grid_size
            
            # 对于每个位置，寻找其真实的相邻块
            for p in range(len(pos_idx)):
                r, c = rows[p].item(), cols[p].item()
                
                # 查找上下左右四个方向的相邻块
                neighbors = []
                for dr, dc in [(0,1), (1,0), (0,-1), (-1,0)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < grid_size and 0 <= nc < grid_size:
                        neighbors.append(nr * grid_size + nc)
                
                # 如果没有相邻块，跳过
                if not neighbors:
                    continue
                    
                # 当前块的位置预测概率
                p_probs = position_probs[i, p]
                
                # 计算该块和其相邻块的预测概率相似度
                similarity_loss = 0.0
                for n in neighbors:
                    # 找到相邻块在当前批次中的索引
                    n_idx = (pos_idx == n).nonzero()
                    if len(n_idx) > 0:
                        n_idx = n_idx[0][0]
                        n_probs = position_probs[i, n_idx]
                        
                        # 计算相邻块位置预测概率的KL散度
                        similarity_loss += F.kl_div(
                            p_probs.log(), n_probs, reduction='sum')
                
                loss += similarity_loss / len(neighbors)
        
        return loss / batch_size if batch_size > 0 else 0.0

    def validate(self):
        self.model.eval()
        val_loss = 0.0
        batches_processed = 0
        
        with torch.no_grad():
            for images, position_labels in self.val_loader:
                images = images.to(self.device)
                position_labels = position_labels.to(self.device)
                
                try:
                    position_logits, relation_logits, reconstructed_image = self.model(images)
                    
                    # 统一张量形状 (与 train 方法相同的处理逻辑)
                    batch_size = position_labels.size(0)
                    num_positions = position_labels.size(1)
                    
                    if len(position_logits.shape) == 3:
                        position_logits = position_logits[:, :num_positions, :]
                        logits_reshaped = position_logits.reshape(batch_size*num_positions, -1)
                        labels_reshaped = position_labels.reshape(-1).long()
                    elif len(position_logits.shape) == 2:
                        if position_logits.size(1) == num_positions:
                            logits_reshaped = torch.zeros(
                                batch_size * num_positions, num_positions, 
                                device=position_logits.device,
                                dtype=torch.float32
                            )
                            
                            for b in range(batch_size):
                                for p in range(num_positions):
                                    pos_value = position_logits[b, p].item()
                                    if isinstance(pos_value, (int, float)) and 0 <= pos_value < num_positions:
                                        logits_reshaped[b * num_positions + p, int(pos_value)] = 1.0
                                    else:
                                        logits_reshaped[b * num_positions + p, 0] = 1.0
                            
                            labels_reshaped = position_labels.reshape(-1).long()
                        else:
                            # 形状不匹配，跳过这个批次
                            print(f"验证时遇到不支持的形状: {position_logits.shape}")
                            continue
                    
                    # 计算损失
                    position_loss = self.criterion(logits_reshaped, labels_reshaped)
                    loss = position_loss
                    
                    val_loss += loss.item()
                    batches_processed += 1
                    
                except Exception as e:
                    print(f"验证时发生错误: {e}")
                    continue
        
        avg_val_loss = val_loss / batches_processed if batches_processed > 0 else float('inf')
        
        # 早停逻辑
        if avg_val_loss < self.best_val_loss:
            self.best_val_loss = avg_val_loss
            self.patience_counter = 0
            # 保存最佳模型
            self.save_checkpoint('checkpoints/enhanced/best_model.pth')
        else:
            self.patience_counter += 1
            
        # 返回是否应该早停
        early_stop = self.patience_counter >= self.patience
        return avg_val_loss, early_stop

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
