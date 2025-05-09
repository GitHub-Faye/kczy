import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Optional, Tuple, Union
import re

class OptimizerManager:
    """
    优化器管理器，负责初始化和管理优化器及其行为
    
    参数:
        optimizer_type (str): 优化器类型，支持 'sgd', 'adam', 'adamw', 'rmsprop'
        model (nn.Module): 需要优化的模型
        lr (float): 学习率
        weight_decay (float): 权重衰减参数
        scheduler_type (Optional[str]): 学习率调度器类型，支持 'step', 'multistep', 'exponential', 'cosine', None
        scheduler_params (Dict): 学习率调度器参数
        **optimizer_params: 特定优化器的额外参数
    """
    def __init__(
        self,
        optimizer_type: str,
        model: nn.Module,
        lr: float = 0.001,
        weight_decay: float = 0.0,
        scheduler_type: Optional[str] = None,
        scheduler_params: Optional[Dict] = None,
        **optimizer_params
    ):
        self.optimizer_type = optimizer_type.lower()
        self.model = model
        self.lr = lr
        self.weight_decay = weight_decay
        self.optimizer_params = optimizer_params
        self.scheduler_type = scheduler_type
        self.scheduler_params = scheduler_params or {}
        
        # 初始化优化器
        self.optimizer = self._init_optimizer()
        
        # 初始化学习率调度器（如果指定）
        self.scheduler = self._init_scheduler() if scheduler_type else None
    
    def _init_optimizer(self) -> optim.Optimizer:
        """
        初始化优化器
        
        返回:
            optim.Optimizer: 配置好的优化器实例
        """
        if self.optimizer_type == 'sgd':
            return optim.SGD(
                self.model.parameters(),
                lr=self.lr,
                weight_decay=self.weight_decay,
                momentum=self.optimizer_params.get('momentum', 0.9),
                nesterov=self.optimizer_params.get('nesterov', False)
            )
        elif self.optimizer_type == 'adam':
            return optim.Adam(
                self.model.parameters(),
                lr=self.lr,
                weight_decay=self.weight_decay,
                betas=self.optimizer_params.get('betas', (0.9, 0.999)),
                eps=self.optimizer_params.get('eps', 1e-8)
            )
        elif self.optimizer_type == 'adamw':
            return optim.AdamW(
                self.model.parameters(),
                lr=self.lr,
                weight_decay=self.weight_decay,
                betas=self.optimizer_params.get('betas', (0.9, 0.999)),
                eps=self.optimizer_params.get('eps', 1e-8)
            )
        elif self.optimizer_type == 'rmsprop':
            return optim.RMSprop(
                self.model.parameters(),
                lr=self.lr,
                weight_decay=self.weight_decay,
                momentum=self.optimizer_params.get('momentum', 0.0),
                alpha=self.optimizer_params.get('alpha', 0.99),
                eps=self.optimizer_params.get('eps', 1e-8)
            )
        else:
            raise ValueError(f"不支持的优化器类型: {self.optimizer_type}")
    
    def _init_scheduler(self) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
        """
        初始化学习率调度器
        
        返回:
            Optional[torch.optim.lr_scheduler._LRScheduler]: 配置好的学习率调度器或None
        """
        if self.scheduler_type == 'step':
            return optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.scheduler_params.get('step_size', 10),
                gamma=self.scheduler_params.get('gamma', 0.1)
            )
        elif self.scheduler_type == 'multistep':
            return optim.lr_scheduler.MultiStepLR(
                self.optimizer,
                milestones=self.scheduler_params.get('milestones', [30, 60, 90]),
                gamma=self.scheduler_params.get('gamma', 0.1)
            )
        elif self.scheduler_type == 'exponential':
            return optim.lr_scheduler.ExponentialLR(
                self.optimizer,
                gamma=self.scheduler_params.get('gamma', 0.9)
            )
        elif self.scheduler_type == 'cosine':
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.scheduler_params.get('T_max', 10),
                eta_min=self.scheduler_params.get('eta_min', 0)
            )
        elif self.scheduler_type == 'plateau':
            return optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode=self.scheduler_params.get('mode', 'min'),
                factor=self.scheduler_params.get('factor', 0.1),
                patience=self.scheduler_params.get('patience', 10),
                threshold=self.scheduler_params.get('threshold', 1e-4),
                threshold_mode=self.scheduler_params.get('threshold_mode', 'rel'),
                cooldown=self.scheduler_params.get('cooldown', 0),
                min_lr=self.scheduler_params.get('min_lr', 0),
                eps=self.scheduler_params.get('eps', 1e-8),
                verbose=self.scheduler_params.get('verbose', False)
            )
        elif self.scheduler_type == 'warmup_cosine':
            # 实现带预热的余弦退火调度
            warmup_epochs = self.scheduler_params.get('warmup_epochs', 5)
            total_epochs = self.scheduler_params.get('total_epochs', 100)
            
            def lambda_fn(epoch):
                if epoch < warmup_epochs:
                    return float(epoch) / max(1, warmup_epochs)
                else:
                    progress = float(epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
                    return 0.5 * (1.0 + torch.cos(torch.tensor(progress * torch.pi)))
            
            return optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda_fn)
        else:
            raise ValueError(f"不支持的调度器类型: {self.scheduler_type}")
    
    def scheduler_step(self, epoch: Optional[int] = None, metric: Optional[float] = None) -> None:
        """
        执行学习率调度器步骤
        
        参数:
            epoch (Optional[int]): 当前epoch，某些调度器需要
            metric (Optional[float]): 性能指标，某些调度器需要
        """
        if self.scheduler:
            if self.scheduler_type == 'plateau':
                # ReduceLROnPlateau需要性能指标
                if metric is None:
                    raise ValueError("ReduceLROnPlateau scheduler需要提供metric参数")
                self.scheduler.step(metric)
            elif epoch is not None:
                # 有些调度器需要epoch参数
                self.scheduler.step(epoch)
            else:
                # 其他调度器
                self.scheduler.step()
    
    def get_lr(self) -> float:
        """
        获取当前学习率
        
        返回:
            float: 当前学习率
        """
        return self.optimizer.param_groups[0]['lr']
    
    def get_optimizer_name(self) -> str:
        """
        获取优化器名称
        
        返回:
            str: 优化器名称
        """
        return self.optimizer_type
    
    def state_dict(self) -> Dict:
        """
        获取优化器的状态字典（用于保存检查点）
        
        返回:
            Dict: 优化器状态字典
        """
        return {
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict() if self.scheduler else None
        }
    
    def load_state_dict(self, state_dict: Dict) -> None:
        """
        从状态字典加载优化器状态（用于恢复检查点）
        
        参数:
            state_dict (Dict): 包含optimizer和scheduler状态的字典
        """
        if 'optimizer' in state_dict:
            self.optimizer.load_state_dict(state_dict['optimizer'])
        if self.scheduler and 'scheduler' in state_dict and state_dict['scheduler']:
            self.scheduler.load_state_dict(state_dict['scheduler'])
    
    def configure_parameter_groups(self, config: List[Dict]) -> None:
        """
        根据配置为不同参数组配置不同的学习率和权重衰减
        
        参数:
            config (List[Dict]): 参数组配置列表
            
        示例:
            config = [
                {'params_regex': '.*bias', 'weight_decay': 0.0},
                {'module_names': ['encoder'], 'lr': 0.0001},
                {'module_names': ['decoder'], 'lr': 0.001}
            ]
        """
        param_groups = []
        named_params = list(self.model.named_parameters())
        params_set = set()
        
        # 处理每个配置组
        for group_config in config:
            params = []
            
            # 通过正则表达式匹配参数名
            if 'params_regex' in group_config:
                pattern = re.compile(group_config['params_regex'])
                for name, param in named_params:
                    if pattern.match(name) and param not in params_set:
                        params.append(param)
                        params_set.add(param)
            
            # 通过模块名匹配参数
            if 'module_names' in group_config:
                for module_name in group_config['module_names']:
                    for name, param in named_params:
                        if name.startswith(module_name) and param not in params_set:
                            params.append(param)
                            params_set.add(param)
            
            # 如果找到了参数，创建一个新的参数组
            if params:
                group = {'params': params}
                
                # 复制配置中除params_regex和module_names外的所有参数
                for key, value in group_config.items():
                    if key not in ['params_regex', 'module_names']:
                        group[key] = value
                
                param_groups.append(group)
        
        # 添加未分组的参数
        remaining_params = [p for name, p in named_params if p not in params_set]
        if remaining_params:
            param_groups.append({'params': remaining_params})
        
        # 重新初始化优化器
        if self.optimizer_type == 'sgd':
            self.optimizer = optim.SGD(
                param_groups,
                lr=self.lr,
                momentum=self.optimizer_params.get('momentum', 0.9),
                nesterov=self.optimizer_params.get('nesterov', False)
            )
        elif self.optimizer_type == 'adam':
            self.optimizer = optim.Adam(
                param_groups,
                lr=self.lr,
                betas=self.optimizer_params.get('betas', (0.9, 0.999)),
                eps=self.optimizer_params.get('eps', 1e-8)
            )
        elif self.optimizer_type == 'adamw':
            self.optimizer = optim.AdamW(
                param_groups,
                lr=self.lr,
                betas=self.optimizer_params.get('betas', (0.9, 0.999)),
                eps=self.optimizer_params.get('eps', 1e-8)
            )
        elif self.optimizer_type == 'rmsprop':
            self.optimizer = optim.RMSprop(
                param_groups,
                lr=self.lr,
                alpha=self.optimizer_params.get('alpha', 0.99),
                eps=self.optimizer_params.get('eps', 1e-8),
                momentum=self.optimizer_params.get('momentum', 0.0)
            )
        
        # 如果有调度器，也需要重新初始化
        if self.scheduler_type:
            self.scheduler = self._init_scheduler() 