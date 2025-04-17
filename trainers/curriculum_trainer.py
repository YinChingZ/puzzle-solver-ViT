import torch
import numpy as np
import os
from torch.utils.data import DataLoader
from trainers.base_trainer import BaseTrainer
import torch.nn.functional as F
import math
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import time
from torch.amp import autocast, GradScaler
from models.modules.contrastive_loss import ContrastivePuzzleLoss

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
        self.scaler = GradScaler('cuda')
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
            print(f"设置数据集网格大小为 {current_grid_size}...")
            self.train_dataset.set_grid_size(current_grid_size)
        else:
            print("警告：数据集不支持动态调整网格大小！")
        
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
                with autocast('cuda'):
                    position_logits, relation_logits, reconstructed, features = self.model(images, return_features=True)
                    
                    # 打印形状信息以便调试
                    # print(f"position_logits.shape: {position_logits.shape}, dtype: {position_logits.dtype}")
                    # print(f"positions.shape: {positions.shape}, dtype: {positions.dtype}")
                    
                    # 根据网格大小调整位置标签
                    grid_size = int(np.sqrt(positions.size(1))) if positions.dim() > 1 else current_grid_size
                    batch_size = position_logits.size(0)
                    
                    # 处理position_logits
                    if position_logits.dim() == 3:  # [batch, seq_len, num_classes]
                        # 移除class token (如果有)
                        if position_logits.size(1) > grid_size * grid_size:
                            # print(f"Detected class token, removing it. Original shape: {position_logits.shape}")
                            position_logits = position_logits[:, 1:grid_size*grid_size+1, :]
                            # print(f"New shape after removing class token: {position_logits.shape}")
                        
                        # 重塑为批次 x 块数 x 类别数
                        position_logits = position_logits.reshape(batch_size, -1, position_logits.size(-1))
                        # print(f"Reshaped position_logits: {position_logits.shape}")
                        
                        # 如果是课程学习的早期阶段，可能需要取前N个位置
                        if positions.size(1) < position_logits.size(1):
                            position_logits = position_logits[:, :positions.size(1), :]
                        
                        # 将logits打平以匹配cross_entropy期望的形状
                        position_logits_flat = position_logits.reshape(-1, position_logits.size(-1))
                        
                        # 根据positions的维度和形状正确处理
                        if positions.dim() == 3:
                            if positions.size(1) == 1:
                                # 当positions是[B,1,4]格式时，只取每个样本的第一个元素
                                positions_flat = positions[:, 0, 0].long()  # [32]
                            else:
                                # 其他3D形状
                                positions_flat = positions.reshape(positions.size(0), -1)[:, 0].long()
                        else:
                            # 2D或1D情况
                            positions_flat = positions.reshape(-1).long()
                        
                        # 检查批次大小是否匹配，如果不匹配则调整
                        if position_logits_flat.size(0) != positions_flat.size(0):
                            if len(position_logits_flat) > len(positions_flat):
                                # 如果logits比positions多，复制positions以匹配
                                repeat_factor = position_logits_flat.size(0) // positions_flat.size(0)
                                if repeat_factor > 1:
                                    positions_flat = positions_flat.repeat_interleave(repeat_factor)
                            else:
                                # 如果positions比logits多，截断positions
                                positions_flat = positions_flat[:position_logits_flat.size(0)]
                        
                        # print(f"Corrected shapes: position_logits_flat={position_logits_flat.shape}, positions_flat={positions_flat.shape}")
                        
                        # 计算损失
                        position_loss = self.criterion(position_logits_flat, positions_flat)
                    else:
                        # 处理其他维度情况...与您的原始代码类似
                        position_loss = self.criterion(position_logits, positions)
                    
                    # 计算对比损失
                    contrastive_loss = self.contrastive_criterion(features, positions)
                    
                    # 计算相邻块预测损失
                    adjacent_loss = self._compute_adjacent_loss(position_logits, positions)
                    
                    # 可选：重建损失
                    recon_loss = 0.0
                    if isinstance(reconstructed, torch.Tensor) and reconstructed.dim() > 3:
                        if images.dim() == 5:  # [B, 1, C, H, W]
                            recon_loss = F.mse_loss(reconstructed, images.squeeze(1))  # 移除多余的维度
                        else:
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
            self.writer.add_scalar('Loss/train/total', loss.item() if torch.is_tensor(loss) else float(loss), epoch)
            self.writer.add_scalar('Loss/train/position', position_loss.item() if torch.is_tensor(position_loss) else float(position_loss), epoch)
            self.writer.add_scalar('Loss/train/contrastive', contrastive_loss.item() if torch.is_tensor(contrastive_loss) else float(contrastive_loss), epoch)
            self.writer.add_scalar('Loss/train/adjacent', float(adjacent_loss), epoch)  # 直接使用float
            self.writer.add_scalar('Loss/train/reconstruction', float(recon_loss), epoch)  # 确保是float
            
            # 记录梯度统计
            if batch_idx % 50 == 0:
                for name, param in self.model.named_parameters():
                    if param.requires_grad and param.grad is not None:
                        try:
                            # 检查梯度是否为空或包含无效值
                            grad_data = param.grad.detach().cpu()
                            
                            # 检查是否为空张量
                            if grad_data.numel() == 0:
                                continue
                                
                            # 检查是否包含NaN或Inf
                            if torch.isnan(grad_data).any() or torch.isinf(grad_data).any():
                                print(f"警告：参数 {name} 的梯度包含NaN或Inf值，已跳过记录")
                                continue
                                
                            # 添加直方图，使用valid_range确保数值有效
                            valid_range = torch.clamp(grad_data, -1e6, 1e6)  # 限制数值范围
                            self.writer.add_histogram(f'Gradients/{name}', valid_range, epoch)
                        except Exception as e:
                            print(f"记录梯度 {name} 时出错: {e}")
            
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
            
            # 检查位置张量的维度并适当处理
            if pos_idx.dim() > 1:
                # 如果position是多维的 [1, 4]，只取第一个维度的第一个值作为位置
                if pos_idx.size(0) == 1:
                    # 形状为 [1, 4]
                    pos_idx = pos_idx[0, 0].reshape(1).long()
                else:
                    # 多个块的情况
                    pos_idx = pos_idx[:, 0].long()
            
            # 安全检查：确保pos_idx包含至少一个元素
            if len(pos_idx) == 0:
                continue
            
            # 获取真实位置的行列索引
            rows = pos_idx // grid_size
            cols = pos_idx % grid_size
            
            # 对于每个位置，寻找其真实的相邻块
            for p in range(len(pos_idx)):
                try:
                    # 安全地提取行列值
                    r = int(rows[p].item())
                    c = int(cols[p].item())
                except (RuntimeError, ValueError):
                    print(f"警告: 无法将张量转换为标量: rows[{p}]={rows[p]}, cols[{p}]={cols[p]}")
                    continue
                
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
                # 确保p是有效索引
                if p < position_probs.size(1):
                    p_probs = position_probs[i, p]
                else:
                    continue
                
                # 计算该块和其相邻块的预测概率相似度
                similarity_loss = 0.0
                valid_neighbors = 0
                
                for n in neighbors:
                    # 找到相邻块在当前批次中的索引
                    n_idx = (pos_idx == n).nonzero()
                    if len(n_idx) > 0:
                        n_idx = n_idx[0][0]
                        if n_idx < position_probs.size(1):
                            n_probs = position_probs[i, n_idx]
                            
                            # 计算相邻块位置预测概率的KL散度
                            try:
                                kl_div = F.kl_div(p_probs.log(), n_probs, reduction='sum')
                                similarity_loss += kl_div
                                valid_neighbors += 1
                            except:
                                continue
                
                if valid_neighbors > 0:
                    loss += similarity_loss / valid_neighbors
        
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
                    
                    # 统一张量形状
                    batch_size = position_labels.size(0)
                    num_positions = position_labels.size(1)
                    
                    if len(position_logits.shape) == 3:  # [batch, seq_len, num_classes]
                        # 重要修复: 只取有效位置的logits
                        position_logits = position_logits[:, 1:num_positions+1, :]  # 跳过class token
                        logits_reshaped = position_logits.reshape(batch_size*num_positions, -1)
                        labels_reshaped = position_labels.reshape(-1).long()
                    elif len(position_logits.shape) == 2:
                        # 添加这个缩进块
                        logits_reshaped = position_logits
                        labels_reshaped = position_labels.reshape(-1).long()
                        if logits_reshaped.size(0) != labels_reshaped.size(0):
                            # 处理批次大小不匹配的情况
                            if logits_reshaped.size(0) > labels_reshaped.size(0):
                                # 如果logits比labels多，截断logits
                                logits_reshaped = logits_reshaped[:labels_reshaped.size(0)]
                            else:
                                # 如果labels比logits多，截断labels
                                labels_reshaped = labels_reshaped[:logits_reshaped.size(0)]
                
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
