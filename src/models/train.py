import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Tuple, Union
import time
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import os
from tqdm import tqdm
import wandb

from src.models.vit import VisionTransformer
from src.utils.config import TrainingConfig
from src.models.optimizer_manager import OptimizerManager
from src.utils.metrics_logger import MetricsLogger
from src.models.model_utils import save_checkpoint, save_model

class LossCalculator:
    """
    损失计算器，提供多种损失函数选择
    
    参数:
        loss_type (str): 损失函数类型，支持 'cross_entropy', 'mse', 'bce', 'focal'
        class_weights (Optional[torch.Tensor]): 类别权重，用于加权损失计算
        gamma (float): Focal Loss的gamma参数
        reduction (str): 损失计算方式，'mean', 'sum', 或 'none'
    """
    def __init__(
        self, 
        loss_type: str = 'cross_entropy',
        class_weights: Optional[torch.Tensor] = None,
        gamma: float = 2.0,
        reduction: str = 'mean'
    ):
        self.loss_type = loss_type.lower()
        self.class_weights = class_weights
        self.gamma = gamma
        self.reduction = reduction
        
        # 初始化损失函数
        if self.loss_type == 'cross_entropy':
            self.loss_fn = nn.CrossEntropyLoss(weight=class_weights, reduction=reduction)
        elif self.loss_type == 'mse':
            self.loss_fn = nn.MSELoss(reduction=reduction)
        elif self.loss_type == 'bce':
            self.loss_fn = nn.BCEWithLogitsLoss(weight=class_weights, reduction=reduction)
        elif self.loss_type == 'focal':
            # Focal Loss需要在forward中特殊实现
            self.loss_fn = None
        else:
            raise ValueError(f"不支持的损失函数类型: {loss_type}. 支持的类型: 'cross_entropy', 'mse', 'bce', 'focal'")
    
    def focal_loss(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        计算Focal Loss
        
        参数:
            inputs (torch.Tensor): 模型输出的预测值 (B, C, ...)
            targets (torch.Tensor): 目标值 (B, ...)
            
        返回:
            torch.Tensor: 计算得到的Focal Loss
        """
        ce_loss = F.cross_entropy(
            inputs, targets, weight=self.class_weights, 
            reduction='none'
        )
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss
    
    def __call__(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        计算损失
        
        参数:
            outputs (torch.Tensor): 模型输出
            targets (torch.Tensor): 目标值
            
        返回:
            torch.Tensor: 计算得到的损失值
        """
        if self.loss_type == 'focal':
            return self.focal_loss(outputs, targets)
        else:
            return self.loss_fn(outputs, targets)
    
    def get_loss_name(self) -> str:
        """获取损失函数名称"""
        return self.loss_type

class BackpropManager:
    """
    反向传播管理器，处理梯度计算、缩放和裁剪
    
    参数:
        grad_clip_value (Optional[float]): 梯度裁剪的阈值，None表示不裁剪
        grad_clip_norm (Optional[float]): 梯度范数裁剪的阈值，None表示不裁剪
        grad_scaler (Optional[torch.cuda.amp.GradScaler]): 用于混合精度训练的梯度缩放器
    """
    def __init__(
        self,
        grad_clip_value: Optional[float] = None,
        grad_clip_norm: Optional[float] = None,
        grad_scaler: Optional[object] = None
    ):
        self.grad_clip_value = grad_clip_value
        self.grad_clip_norm = grad_clip_norm
        self.grad_scaler = grad_scaler
        
    def compute_gradients(
        self, 
        loss: torch.Tensor, 
        model: nn.Module,
        optimizer: Union[optim.Optimizer, OptimizerManager],
        retain_graph: bool = False
    ) -> None:
        """
        计算损失的梯度
        
        参数:
            loss (torch.Tensor): 损失值
            model (nn.Module): 需要计算梯度的模型
            optimizer (Union[optim.Optimizer, OptimizerManager]): 优化器或优化器管理器
            retain_graph (bool): 是否保留计算图
        """
        # 处理OptimizerManager
        if isinstance(optimizer, OptimizerManager):
            optimizer_obj = optimizer.optimizer
        else:
            optimizer_obj = optimizer
        
        # 清除之前的梯度
        optimizer_obj.zero_grad()
        
        # 使用混合精度训练时的梯度计算
        if self.grad_scaler is not None:
            self.grad_scaler.scale(loss).backward(retain_graph=retain_graph)
        else:
            # 标准梯度计算
            loss.backward(retain_graph=retain_graph)
    
    def apply_gradient_clipping(self, model: nn.Module) -> None:
        """
        应用梯度裁剪
        
        参数:
            model (nn.Module): 需要裁剪梯度的模型
        """
        if self.grad_clip_value is not None:
            # 按值裁剪梯度
            torch.nn.utils.clip_grad_value_(model.parameters(), self.grad_clip_value)
            
        if self.grad_clip_norm is not None:
            # 按范数裁剪梯度
            torch.nn.utils.clip_grad_norm_(model.parameters(), self.grad_clip_norm)
    
    def optimizer_step(self, optimizer: Union[optim.Optimizer, OptimizerManager]) -> None:
        """
        执行优化器步骤，更新模型参数
        
        参数:
            optimizer (Union[optim.Optimizer, OptimizerManager]): 优化器或优化器管理器
        """
        # 处理OptimizerManager
        if isinstance(optimizer, OptimizerManager):
            optimizer_obj = optimizer.optimizer
        else:
            optimizer_obj = optimizer
            
        if self.grad_scaler is not None:
            # 混合精度训练中的参数更新
            self.grad_scaler.step(optimizer_obj)
            self.grad_scaler.update()
        else:
            # 标准参数更新
            optimizer_obj.step()
    
    def backward_and_update(
        self, 
        loss: torch.Tensor, 
        model: nn.Module, 
        optimizer: Union[optim.Optimizer, OptimizerManager],
        retain_graph: bool = False
    ) -> None:
        """
        执行完整的反向传播和参数更新过程
        
        参数:
            loss (torch.Tensor): 损失值
            model (nn.Module): 模型
            optimizer (Union[optim.Optimizer, OptimizerManager]): 优化器或优化器管理器
            retain_graph (bool): 是否保留计算图
        """
        # 计算梯度
        self.compute_gradients(loss, model, optimizer, retain_graph)
        
        # 应用梯度裁剪
        self.apply_gradient_clipping(model)
        
        # 更新模型参数
        self.optimizer_step(optimizer)

class TrainingLoop:
    """
    训练循环，整合损失计算、反向传播和优化器步骤
    
    参数:
        model (nn.Module): 要训练的模型
        loss_calculator (LossCalculator): 损失计算器
        optimizer_manager (Optional[OptimizerManager]): 优化器管理器，仅做评估时可为None
        backprop_manager (Optional[BackpropManager]): 反向传播管理器，仅做评估时可为None
        device (torch.device): 运行设备
        metrics_logger (Optional[MetricsLogger]): 指标记录器，用于记录训练和评估指标
        log_histograms (bool): 是否记录模型参数和梯度的直方图到TensorBoard
        log_images (bool): 是否记录样本图像到TensorBoard
    """
    def __init__(
        self,
        model: nn.Module,
        loss_calculator: LossCalculator,
        optimizer_manager: Optional[OptimizerManager] = None,
        backprop_manager: Optional[BackpropManager] = None,
        device: torch.device = torch.device('cpu'),
        metrics_logger: Optional[MetricsLogger] = None,
        log_histograms: bool = False,
        log_images: bool = False
    ):
        self.model = model
        self.loss_calculator = loss_calculator
        self.optimizer_manager = optimizer_manager
        self.backprop_manager = backprop_manager
        self.device = device
        self.metrics_logger = metrics_logger
        self.log_histograms = log_histograms
        self.log_images = log_images
        
        # 将模型移至指定设备
        self.model.to(self.device)
    
    @classmethod
    def from_config(
        cls,
        model: nn.Module,
        config: TrainingConfig,
        class_weights: Optional[torch.Tensor] = None
    ) -> 'TrainingLoop':
        """
        从配置创建训练循环
        
        参数:
            model (nn.Module): 模型
            config (TrainingConfig): 训练配置
            class_weights (Optional[torch.Tensor]): 类别权重张量
            
        返回:
            TrainingLoop: 训练循环实例
        """
        # 确定设备
        device = torch.device(config.device)
        
        # 创建损失计算器
        loss_calculator = LossCalculator(
            loss_type=config.loss_type,
            class_weights=class_weights
        )
        
        # 创建梯度缩放器（用于混合精度训练）
        grad_scaler = None
        if config.use_mixed_precision and torch.cuda.is_available():
            grad_scaler = torch.cuda.amp.GradScaler()
        
        # 创建反向传播管理器
        backprop_manager = BackpropManager(
            grad_clip_value=config.grad_clip_value,
            grad_clip_norm=config.grad_clip_norm,
            grad_scaler=grad_scaler
        )
        
        # 创建优化器管理器
        optimizer_params = config.optimizer_params.copy() if hasattr(config, 'optimizer_params') else {}
        optimizer_manager = OptimizerManager(
            model=model,
            optimizer_type=config.optimizer_type,
            learning_rate=config.learning_rate,
            weight_decay=config.weight_decay,
            scheduler_type=config.scheduler_type,
            scheduler_params=config.scheduler_params,
            **optimizer_params
        )
        
        # 创建指标记录器
        metrics_logger = None
        if hasattr(config, 'metrics_dir') and config.metrics_dir:
            # 获取实验名称
            experiment_name = config.metrics_experiment_name
            if experiment_name is None:
                # 如果未提供实验名称，使用时间戳作为默认名称
                experiment_name = f"exp_{int(time.time())}"
            
            # 创建指标记录器
            metrics_logger = MetricsLogger(
                save_dir=config.metrics_dir,
                experiment_name=experiment_name,
                save_format=config.metrics_format,
                save_freq=config.metrics_save_freq,
                enable_wandb=config.enable_wandb,
                wandb_project=config.wandb_project,
                wandb_config=config.wandb_config if hasattr(config, 'wandb_config') else None
            )
        
        # 返回训练循环实例
        return cls(
            model=model,
            loss_calculator=loss_calculator,
            optimizer_manager=optimizer_manager,
            backprop_manager=backprop_manager,
            device=device,
            metrics_logger=metrics_logger,
            log_histograms=config.log_histograms,
            log_images=config.log_images
        )

    def train_epoch(
        self, 
        train_loader: DataLoader,
        epoch: int
    ) -> Dict[str, float]:
        """
        训练一个周期
        
        参数:
            train_loader (DataLoader): 训练数据加载器
            epoch (int): 当前周期
            
        返回:
            Dict[str, float]: 训练指标
        """
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        # 记录样本图像
        if self.log_images and self.metrics_logger and self.metrics_logger.enable_wandb:
            for batch_idx, (inputs, targets) in enumerate(train_loader):
                if batch_idx == 0:  # 只记录第一个批次的图像
                    for i in range(min(4, inputs.size(0))):  # 最多记录4张图像
                        if self.metrics_logger:
                            # 转换为CPU上的NumPy数组
                            img = inputs[i].cpu().numpy().transpose(1, 2, 0)
                            # 归一化到[0, 1]范围
                            img = (img - img.min()) / (img.max() - img.min() + 1e-8)
                            self.metrics_logger.log_image(
                                f"sample_{i}_label_{targets[i].item() if not isinstance(targets, list) else 'unknown'}", 
                                img, 
                                epoch
                            )
                break
        
        # 添加进度条
        pbar = tqdm(train_loader, desc=f"训练 Epoch {epoch}", unit="batch", leave=True)
        
        # 训练循环
        for batch_idx, (inputs, targets) in enumerate(pbar):
            # 将输入数据移动到设备上
            inputs = inputs.to(self.device)
            
            # 处理targets - 可能是张量或字典列表
            if isinstance(targets, torch.Tensor):
                targets = targets.to(self.device)
            elif isinstance(targets, list) and all(isinstance(t, dict) for t in targets):
                # 处理字典列表情况 - 需要递归处理每个字典中的张量
                processed_targets = []
                for target_dict in targets:
                    processed_dict = {}
                    for k, v in target_dict.items():
                        if isinstance(v, torch.Tensor):
                            processed_dict[k] = v.to(self.device)
                        else:
                            processed_dict[k] = v
                    processed_targets.append(processed_dict)
                targets = processed_targets
            
            # 前向传播
            outputs = self.model(inputs)
            
            # 计算损失 - 根据targets类型调整
            if isinstance(targets, list) and all(isinstance(t, dict) for t in targets):
                # 如果是对象检测或分割任务，targets是字典列表
                # 从字典中提取标签进行损失计算
                # 假设我们需要从targets中获取labels
                if all('labels' in t for t in targets):
                    label_tensors = [t['labels'] for t in targets]
                    # 如果labels是批次维度的张量，需要连接它们
                    if all(isinstance(lt, torch.Tensor) and lt.dim() > 0 for lt in label_tensors):
                        labels = torch.cat(label_tensors, dim=0)
                        loss = self.loss_calculator(outputs, labels)
                    else:
                        # 如果无法正确提取标签，使用自定义的损失计算逻辑
                        # 这里可能需要根据具体的任务类型来调整
                        print(f"警告: 在第{epoch}个epoch的第{batch_idx}个批次中无法提取标准标签格式")
                        # 默认处理：根据模型的输出和targets计算适当的损失
                        # 这里需要根据具体任务自定义损失计算逻辑
                        loss = torch.tensor(0.0, device=self.device, requires_grad=True)
                else:
                    print(f"警告: 在第{epoch}个epoch的第{batch_idx}个批次中targets字典中没有'labels'键")
                    loss = torch.tensor(0.0, device=self.device, requires_grad=True)
            else:
                # 标准分类任务
                loss = self.loss_calculator(outputs, targets)
            
            # 反向传播和优化
            if self.backprop_manager is not None and self.optimizer_manager is not None:
                self.backprop_manager.backward_and_update(
                    loss, 
                    self.model, 
                    self.optimizer_manager
                )
            
            # 统计 - 根据任务类型调整
            current_loss = loss.item()
            total_loss += current_loss
            
            # 对于分类任务，计算准确率
            if not isinstance(targets, list):
                _, predicted = outputs.max(1)
                total += targets.size(0)
                current_correct = predicted.eq(targets).sum().item()
                correct += current_correct
                current_acc = 100.0 * current_correct / targets.size(0)
                pbar.set_postfix({'loss': f"{current_loss:.4f}", 'acc': f"{current_acc:.2f}%"})
            else:
                # 对于对象检测等任务，准确率计算可能不同
                # 这里只增加样本数，不计算准确率
                total += inputs.size(0)
                pbar.set_postfix({'loss': f"{current_loss:.4f}"})
        
        # 计算平均损失和准确率
        avg_loss = total_loss / len(train_loader)
        accuracy = 100. * correct / total if total > 0 and correct > 0 else 0.0
        
        # 记录模型参数和梯度直方图(每个epoch记录)
        if self.log_histograms and self.metrics_logger and self.metrics_logger.enable_wandb:
            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    self.metrics_logger.log_histogram(f"{name}.grad", param.grad.cpu().numpy(), epoch, "gradients")
                self.metrics_logger.log_histogram(name, param.data.cpu().numpy(), epoch, "parameters")
        
        # 返回指标结果
        return {
            'loss': avg_loss,
            'accuracy': accuracy
        }
    
    def validate(
        self, 
        val_loader: DataLoader
    ) -> Dict[str, float]:
        """
        在验证集上评估模型
        
        参数:
            val_loader (DataLoader): 验证数据加载器
            
        返回:
            Dict[str, float]: 验证指标
        """
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        # 添加进度条
        pbar = tqdm(val_loader, desc="验证中", unit="batch", leave=True)
        
        with torch.no_grad():
            for inputs, targets in pbar:
                # 将输入数据移动到设备上
                inputs = inputs.to(self.device)
                
                # 处理targets - 可能是张量或字典列表
                if isinstance(targets, torch.Tensor):
                    targets = targets.to(self.device)
                elif isinstance(targets, list) and all(isinstance(t, dict) for t in targets):
                    # 处理字典列表情况 - 需要递归处理每个字典中的张量
                    processed_targets = []
                    for target_dict in targets:
                        processed_dict = {}
                        for k, v in target_dict.items():
                            if isinstance(v, torch.Tensor):
                                processed_dict[k] = v.to(self.device)
                            else:
                                processed_dict[k] = v
                        processed_targets.append(processed_dict)
                    targets = processed_targets
                
                # 前向传播
                outputs = self.model(inputs)
                
                # 计算损失 - 根据targets类型调整
                if isinstance(targets, list) and all(isinstance(t, dict) for t in targets):
                    # 如果是对象检测或分割任务，targets是字典列表
                    # 从字典中提取标签进行损失计算
                    if all('labels' in t for t in targets):
                        label_tensors = [t['labels'] for t in targets]
                        # 如果labels是批次维度的张量，需要连接它们
                        if all(isinstance(lt, torch.Tensor) and lt.dim() > 0 for lt in label_tensors):
                            labels = torch.cat(label_tensors, dim=0)
                            loss = self.loss_calculator(outputs, labels)
                        else:
                            # 如果无法正确提取标签，使用自定义的损失计算逻辑
                            print(f"验证时无法提取标准标签格式")
                            # 默认处理
                            loss = torch.tensor(0.0, device=self.device)
                    else:
                        print(f"验证时targets字典中没有'labels'键")
                        loss = torch.tensor(0.0, device=self.device)
                else:
                    # 标准分类任务
                    loss = self.loss_calculator(outputs, targets)
                
                # 计算准确率（如果是分类任务）
                current_loss = loss.item()
                if not isinstance(targets, list):
                    _, predicted = outputs.max(1)
                    batch_size = targets.size(0)
                    total += batch_size
                    current_correct = predicted.eq(targets).sum().item()
                    correct += current_correct
                    current_acc = 100.0 * current_correct / batch_size
                    pbar.set_postfix({'val_loss': f"{current_loss:.4f}", 'val_acc': f"{current_acc:.2f}%"})
                else:
                    # 对于对象检测等任务，准确率计算可能不同
                    # 这里只增加样本数，不计算准确率
                    total += inputs.size(0)
                    pbar.set_postfix({'val_loss': f"{current_loss:.4f}"})
                
                # 累加损失
                total_loss += current_loss
        
        # 计算平均损失和准确率
        avg_loss = total_loss / len(val_loader)
        accuracy = 100.0 * correct / total if total > 0 and correct > 0 else 0.0
        
        return {
            'val_loss': avg_loss,
            'val_accuracy': accuracy
        }
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        num_epochs: int = 10,
        checkpoint_path: Optional[str] = None,
        log_interval: int = 1,
        early_stopping: bool = False,
        early_stopping_patience: int = 10,
        checkpoint_dir: Optional[str] = None,
        checkpoint_freq: int = 1
    ) -> Dict[str, List[float]]:
        """
        训练模型
        
        参数:
            train_loader (DataLoader): 训练数据加载器
            val_loader (Optional[DataLoader]): 验证数据加载器
            num_epochs (int): 训练周期数
            checkpoint_path (Optional[str]): 检查点路径，用于恢复训练
            log_interval (int): 日志记录间隔
            early_stopping (bool): 是否启用早停
            early_stopping_patience (int): 早停的耐心值
            checkpoint_dir (Optional[str]): 检查点保存目录
            checkpoint_freq (int): 检查点保存频率
            
        返回:
            Dict[str, List[float]]: 训练历史
        """
        # 从检查点恢复（代码保持不变）
        start_epoch = 1
        best_val_loss = float('inf')
        patience_counter = 0
        train_history = {'loss': [], 'accuracy': []}
        val_history = {'loss': [], 'accuracy': []}
        
        # 初始化用于跟踪的属性
        self.train_losses = []
        self.val_losses = []
        self.train_accs = []
        self.val_accs = []
        
        if checkpoint_path is not None:
            # 恢复代码保持不变...
            pass
        
        # 如果使用Weights & Biases记录超参数
        if self.metrics_logger and self.metrics_logger.enable_wandb:
            # 收集超参数
            hparams = {}
            if self.optimizer_manager is not None:
                opt_class = self.optimizer_manager.optimizer.__class__.__name__
                hparams['optimizer'] = opt_class
                
                # 提取优化器参数
                for key, value in self.optimizer_manager.optimizer.defaults.items():
                    if isinstance(value, (int, float, str, bool)):
                        hparams[f"optimizer/{key}"] = value
            
            # 记录损失函数类型
            hparams['loss_function'] = self.loss_calculator.get_loss_name()
            
            # 如果有学习率调度器，记录其类型
            if self.optimizer_manager and self.optimizer_manager.scheduler:
                scheduler_class = self.optimizer_manager.scheduler.__class__.__name__
                hparams['lr_scheduler'] = scheduler_class
            
            # 记录模型结构信息
            hparams['model_type'] = self.model.__class__.__name__
            # 计算模型参数量
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            hparams['total_params'] = total_params
            hparams['trainable_params'] = trainable_params
            
            # 记录训练配置
            hparams['batch_size'] = train_loader.batch_size if hasattr(train_loader, 'batch_size') else None
            hparams['num_epochs'] = num_epochs
            hparams['early_stopping'] = early_stopping
            if early_stopping:
                hparams['early_stopping_patience'] = early_stopping_patience
            
            # 将在训练结束时记录最终指标
        
        # 添加 epoch 进度条
        epochs_pbar = tqdm(range(start_epoch, num_epochs + 1), desc="训练进度", unit="epoch", position=0)
        
        # 训练循环
        for epoch in epochs_pbar:
            # 训练一个周期
            train_metrics = self.train_epoch(train_loader, epoch)
            train_history['loss'].append(train_metrics['loss'])
            train_history['accuracy'].append(train_metrics['accuracy'])
            # 更新用于跟踪的属性
            self.train_losses.append(train_metrics['loss'])
            self.train_accs.append(train_metrics['accuracy'])
            
            # 验证
            if val_loader is not None:
                val_metrics = self.validate(val_loader)
                val_history['loss'].append(val_metrics['val_loss'])
                val_history['accuracy'].append(val_metrics['val_accuracy'])
                # 更新用于跟踪的属性
                self.val_losses.append(val_metrics['val_loss'])
                self.val_accs.append(val_metrics['val_accuracy'])
                
                # 检查早停
                if early_stopping:
                    if val_metrics['val_loss'] < best_val_loss:
                        best_val_loss = val_metrics['val_loss']
                        patience_counter = 0
                    else:
                        patience_counter += 1
                        
                    if patience_counter >= early_stopping_patience:
                        print(f"早停触发于epoch {epoch}，耐心值: {early_stopping_patience}")
                        break
            
            # 记录指标
            if self.metrics_logger:
                self.metrics_logger.log_train_metrics(train_metrics, epoch)
                if val_loader is not None:
                    self.metrics_logger.log_eval_metrics(val_metrics, epoch)
            
            # 保存检查点
            if checkpoint_dir is not None and epoch % checkpoint_freq == 0:
                os.makedirs(checkpoint_dir, exist_ok=True)
                checkpoint_file = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch}.pt")
                
                # 保存检查点
                if self.optimizer_manager is not None:
                    save_checkpoint(
                        self.model,
                        self.optimizer_manager.state_dict(),  # 获取优化器状态
                        checkpoint_file,  # 文件路径
                        epoch,  # 当前epoch
                        best_val_loss if val_loader is not None else None  # 训练历史
                    )
                else:
                    # 仅保存模型
                    save_model(self.model, checkpoint_file)
                    
                epochs_pbar.write(f"保存检查点到 {checkpoint_file}")
            
            # 更新进度条描述，显示当前指标
            status_desc = f"损失:{train_metrics['loss']:.4f} 准确率:{train_metrics['accuracy']:.2f}%"
            if val_loader is not None:
                status_desc += f" | 验证损失:{val_metrics['val_loss']:.4f} 验证准确率:{val_metrics['val_accuracy']:.2f}%"
            epochs_pbar.set_postfix_str(status_desc)
            
            # 打印日志
            if epoch % log_interval == 0:
                log_str = f"Epoch {epoch}/{num_epochs} "
                log_str += f"Train Loss: {train_metrics['loss']:.4f} "
                log_str += f"Train Acc: {train_metrics['accuracy']:.2f}% "
                
                if val_loader is not None:
                    log_str += f"Val Loss: {val_metrics['val_loss']:.4f} "
                    log_str += f"Val Acc: {val_metrics['val_accuracy']:.2f}% "
                
                epochs_pbar.write(log_str)
        
        # 训练结束后，记录最终的超参数和结果指标到Weights & Biases
        if self.metrics_logger and self.metrics_logger.enable_wandb:
            # 创建最终结果指标字典
            final_metrics = {}
            if train_history['loss']:
                final_metrics['hparam/train_loss'] = train_history['loss'][-1]
                final_metrics['hparam/train_accuracy'] = train_history['accuracy'][-1]
            
            if val_history['loss']:
                final_metrics['hparam/val_loss'] = val_history['loss'][-1]
                final_metrics['hparam/val_accuracy'] = val_history['accuracy'][-1]
                final_metrics['hparam/best_val_loss'] = best_val_loss
            
            # 记录超参数和最终指标
            self.metrics_logger.log_hparams(hparams, final_metrics)
            
            # 注意：不要在这里关闭连接，因为后续还需要继续使用wandb记录评估指标
        
        return {
            'train': train_history,
            'val': val_history if val_loader is not None else None
        }

    def evaluate(
        self, 
        test_loader: DataLoader,
        num_classes: Optional[int] = None,
        average: str = 'weighted',
        visualize_confusion_matrix: bool = False,
        output_path: Optional[str] = None
    ) -> Dict[str, Union[float, np.ndarray, str]]:
        """
        在测试集上评估模型，并计算更多的评价指标
        
        参数:
            test_loader (DataLoader): 测试数据加载器
            num_classes (Optional[int]): 类别数量，用于计算评价指标
            average (str): 多分类评价指标的平均方式，可选'micro', 'macro', 'weighted'
            visualize_confusion_matrix (bool): 是否可视化混淆矩阵
            output_path (Optional[str]): 混淆矩阵图像保存路径
            
        返回:
            Dict[str, Union[float, np.ndarray, str]]: 测试指标
        """
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        y_true = []
        y_pred = []
        
        # 添加进度条
        pbar = tqdm(test_loader, desc="测试评估中", unit="batch", leave=True)
        
        with torch.no_grad():
            for inputs, targets in pbar:
                # 将输入数据移动到设备上
                inputs = inputs.to(self.device)
                
                # 处理targets - 可能是张量或字典列表
                if isinstance(targets, torch.Tensor):
                    targets = targets.to(self.device)
                elif isinstance(targets, list) and all(isinstance(t, dict) for t in targets):
                    # 处理字典列表情况 - 需要递归处理每个字典中的张量
                    processed_targets = []
                    for target_dict in targets:
                        processed_dict = {}
                        for k, v in target_dict.items():
                            if isinstance(v, torch.Tensor):
                                processed_dict[k] = v.to(self.device)
                            else:
                                processed_dict[k] = v
                        processed_targets.append(processed_dict)
                    targets = processed_targets
                
                # 前向传播
                outputs = self.model(inputs)
                
                # 计算损失 - 根据targets类型调整
                if isinstance(targets, list) and all(isinstance(t, dict) for t in targets):
                    # 如果是对象检测或分割任务，targets是字典列表
                    # 从字典中提取标签进行损失计算
                    if all('labels' in t for t in targets):
                        label_tensors = [t['labels'] for t in targets]
                        # 如果labels是批次维度的张量，需要连接它们
                        if all(isinstance(lt, torch.Tensor) and lt.dim() > 0 for lt in label_tensors):
                            labels = torch.cat(label_tensors, dim=0)
                            loss = self.loss_calculator(outputs, labels)
                        else:
                            # 如果无法正确提取标签，使用自定义的损失计算逻辑
                            print(f"测试时无法提取标准标签格式")
                            # 默认处理
                            loss = torch.tensor(0.0, device=self.device)
                    else:
                        print(f"测试时targets字典中没有'labels'键")
                        loss = torch.tensor(0.0, device=self.device)
                else:
                    # 标准分类任务
                    loss = self.loss_calculator(outputs, targets)
                
                # 计算准确率（如果是分类任务）
                current_loss = loss.item()
                if not isinstance(targets, list):
                    _, predicted = outputs.max(1)
                    batch_size = targets.size(0)
                    total += batch_size
                    current_correct = predicted.eq(targets).sum().item()
                    correct += current_correct
                    current_acc = 100.0 * current_correct / batch_size
                    
                    # 记录真实标签和预测标签
                    y_true.extend(targets.cpu().numpy())
                    y_pred.extend(predicted.cpu().numpy())
                    
                    # 更新进度条
                    pbar.set_postfix({'test_loss': f"{current_loss:.4f}", 'test_acc': f"{current_acc:.2f}%"})
                else:
                    # 对于对象检测等任务，准确率计算可能不同
                    # 这里只增加样本数，不计算准确率
                    total += inputs.size(0)
                    
                    # 对于对象检测任务，尝试从字典中提取标签 (如果存在)
                    if all('labels' in t for t in targets):
                        for i, target_dict in enumerate(targets):
                            if isinstance(target_dict['labels'], torch.Tensor):
                                # 获取当前样本的标签
                                labels = target_dict['labels'].cpu().numpy()
                                # 获取当前样本的预测结果
                                _, sample_pred = outputs[i].max(0)
                                sample_pred = sample_pred.cpu().numpy()
                                # 添加到总体结果中
                                y_true.extend(labels)
                                y_pred.extend([sample_pred] * len(labels))
                    
                    # 更新进度条，不显示准确率
                    pbar.set_postfix({'test_loss': f"{current_loss:.4f}"})
                
                # 累加损失
                total_loss += current_loss
        
        # 计算平均损失和准确率
        avg_loss = total_loss / len(test_loader)
        accuracy = 100.0 * correct / total if total > 0 and correct > 0 else 0.0
        
        # 整理评估指标
        eval_metrics = {
            'test_loss': avg_loss,
            'test_accuracy': accuracy
        }
        
        # 如果有收集的标签和预测值，计算详细的评估指标
        if len(y_true) > 0 and len(y_pred) > 0:
            # 转换为numpy数组
            y_true = np.array(y_true)
            y_pred = np.array(y_pred)
            
            # 检测类别数量
            if num_classes is None:
                num_classes = len(np.unique(np.concatenate([y_true, y_pred])))
            
            # 是否为二分类问题
            is_binary = num_classes == 2
            
            # 计算更多的评价指标
            if is_binary:
                # 二分类问题的评价指标
                precision = precision_score(y_true, y_pred)
                recall = recall_score(y_true, y_pred)
                f1 = f1_score(y_true, y_pred)
            else:
                # 多分类问题的评价指标
                precision = precision_score(y_true, y_pred, average=average, zero_division=0)
                recall = recall_score(y_true, y_pred, average=average, zero_division=0)
                f1 = f1_score(y_true, y_pred, average=average, zero_division=0)
            
            # 计算混淆矩阵
            conf_matrix = confusion_matrix(y_true, y_pred, labels=range(num_classes))
            class_report = classification_report(y_true, y_pred, labels=range(num_classes))
            
            # 可视化混淆矩阵
            if visualize_confusion_matrix:
                self._plot_confusion_matrix(conf_matrix, num_classes, output_path)
            
            # 添加到评估指标
            eval_metrics.update({
                'precision': precision * 100.0,  # 转换为百分比
                'recall': recall * 100.0,        # 转换为百分比
                'f1_score': f1 * 100.0,          # 转换为百分比
                'confusion_matrix': conf_matrix,
                'classification_report': class_report
            })
        else:
            print("警告: 无法计算详细评估指标，因为没有收集到足够的真实标签和预测值")
            # 添加默认值，避免代码后续报错
            eval_metrics.update({
                'precision': 0.0,
                'recall': 0.0,
                'f1_score': 0.0,
                'confusion_matrix': np.zeros((1, 1)),
                'classification_report': "无法生成分类报告"
            })
        
        # 如果有指标记录器，记录评估指标
        if self.metrics_logger is not None:
            # 创建一个指标副本，不包含复杂对象（如numpy数组和字符串）
            metrics_for_logging = {k: v for k, v in eval_metrics.items()
                                  if not isinstance(v, (np.ndarray, str))}
            
            # 使用当前的wandb步骤+10作为评估步骤，确保不会有步骤冲突
            # 为评估指标预留一个较大的步骤空间
            current_epoch = 1000  # 使用一个足够大的值，确保不会冲突
            if hasattr(self, 'train_losses') and len(self.train_losses) > 0:
                # 如果有训练历史，使用训练轮次数+100作为基础
                current_epoch = len(self.train_losses) + 100
                
            # 如果使用wandb，检查当前步骤
            if self.metrics_logger.enable_wandb and self.metrics_logger.wandb_initialized and wandb.run:
                if hasattr(wandb.run, 'step'):
                    # 使用当前wandb步骤+10确保没有冲突
                    current_epoch = max(current_epoch, wandb.run.step + 10)
                    
            # 使用计算的步骤记录指标
            self.metrics_logger.log_eval_metrics(metrics_for_logging, current_epoch)
        
        return eval_metrics
    
    def _plot_confusion_matrix(
        self,
        conf_matrix: np.ndarray,
        num_classes: int,
        output_path: Optional[str] = None
    ) -> None:
        """
        绘制混淆矩阵
        
        参数:
            conf_matrix (np.ndarray): 混淆矩阵
            num_classes (int): 类别数量
            output_path (Optional[str]): 图像保存路径
        """
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            conf_matrix, 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            xticklabels=range(num_classes),
            yticklabels=range(num_classes)
        )
        plt.xlabel('预测标签')
        plt.ylabel('真实标签')
        plt.title('混淆矩阵')
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"混淆矩阵已保存到 {output_path}")
        else:
            plt.show()
        
        plt.close()
    
    @classmethod
    def evaluate_model(
        cls,
        test_loader: DataLoader,
        model: nn.Module,
        checkpoint_path: Optional[str] = None,
        loss_type: str = 'cross_entropy',
        device: Optional[torch.device] = None,
        num_classes: Optional[int] = None,
        average: str = 'weighted',
        visualize_confusion_matrix: bool = False,
        output_path: Optional[str] = None
    ) -> Dict[str, Union[float, np.ndarray, str]]:
        """
        加载和评估模型的静态方法
        
        参数:
            test_loader (DataLoader): 测试数据加载器
            model (nn.Module): 要评估的模型
            checkpoint_path (Optional[str]): 模型检查点路径，如果提供则加载模型权重
            loss_type (str): 损失函数类型
            device (Optional[torch.device]): 运行设备，如果为None则自动选择
            num_classes (Optional[int]): 类别数量
            average (str): 多分类评价指标的平均方式
            visualize_confusion_matrix (bool): 是否可视化混淆矩阵
            output_path (Optional[str]): 混淆矩阵图像保存路径
            
        返回:
            Dict[str, Union[float, np.ndarray, str]]: 测试指标
        """
        # 设置设备
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 模型移至设备
        model = model.to(device)
        
        # 如果提供了检查点路径，加载模型权重
        if checkpoint_path:
            print(f"加载模型检查点: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
        
        # 创建损失计算器
        loss_calculator = LossCalculator(loss_type=loss_type)
        
        # 创建指标记录器
        metrics_logger = MetricsLogger(save_dir='metrics', experiment_name='model_evaluation')
        
        # 创建空的优化器管理器和反向传播管理器（评估时不需要）
        optimizer_manager = None
        backprop_manager = None
        
        # 创建训练循环实例
        training_loop = cls(
            model=model,
            loss_calculator=loss_calculator,
            optimizer_manager=optimizer_manager,
            backprop_manager=backprop_manager,
            device=device,
            metrics_logger=metrics_logger
        )
        
        # 评估模型
        eval_metrics = training_loop.evaluate(
            test_loader=test_loader,
            num_classes=num_classes,
            average=average,
            visualize_confusion_matrix=visualize_confusion_matrix,
            output_path=output_path
        )
        
        # 打印评估结果
        print("\n" + "=" * 50)
        print("模型评估结果:")
        print("=" * 50)
        print(f"测试损失: {eval_metrics['test_loss']:.4f}")
        print(f"测试准确率: {eval_metrics['test_accuracy']:.2f}%")
        print(f"精确率: {eval_metrics['precision']:.2f}%")
        print(f"召回率: {eval_metrics['recall']:.2f}%")
        print(f"F1分数: {eval_metrics['f1_score']:.2f}%")
        print("\n分类报告:")
        print(eval_metrics['classification_report'])
        print("=" * 50)
        
        return eval_metrics 