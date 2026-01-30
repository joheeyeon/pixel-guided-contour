import torch
from torch.optim.lr_scheduler import MultiStepLR, _LRScheduler, CosineAnnealingLR
from collections import Counter
import math

_scheduler_factory = {
    'MultiStepLR': torch.optim.lr_scheduler.MultiStepLR,
    'ReduceLROnPlateau': torch.optim.lr_scheduler.ReduceLROnPlateau
}

class AdvancedTemperatureScheduler(_LRScheduler):
    """
    Advanced scheduling for temperature parameter: freeze → warmup → cosine/multistep
    """
    def __init__(self, optimizer, config, last_epoch=-1, verbose=False):
        # Extract and save only necessary values from config (prevent pickle issues)
        self.total_epochs = config.train.epoch
        self.temp_config = config.train.temperature_advanced_scheduling.copy()
        
        # Save basic optimizer configuration
        self.default_milestones = config.train.optimizer['milestones']
        self.default_gamma = config.train.optimizer['gamma']
        self.temp_milestones = config.train.optimizer.get('temperature_milestones', self.default_milestones)
        self.temp_gamma = config.train.optimizer.get('temperature_gamma', self.default_gamma)
        
        # Configure scheduling phases
        self.freeze_epochs = self.temp_config['freeze_epochs']
        self.warmup_epochs = int(self.total_epochs * self.temp_config['warmup_ratio'])
        self.main_start_epoch = self.freeze_epochs + self.warmup_epochs
        
        # Find temperature parameter group
        self.temp_lr = config.train.optimizer.get('temperature_lr', config.train.optimizer['lr'])
        self.is_temp_group = []
        for group in optimizer.param_groups:
            is_temp = abs(group['lr'] - self.temp_lr) < 1e-10
            self.is_temp_group.append(is_temp)
        
        # Cosine configuration (when final_scheduler is 'cosine')
        if self.temp_config['final_scheduler'] == 'cosine':
            self.cosine_eta_min = self.temp_lr * self.temp_config['cosine_eta_min_ratio']
            self.cosine_epochs = self.total_epochs - self.main_start_epoch
        
        print(f"[AdvancedTemperatureScheduler] Config:")
        print(f"  Total epochs: {self.total_epochs}")
        print(f"  Freeze epochs: {self.freeze_epochs}")
        print(f"  Warmup epochs: {self.warmup_epochs}")
        print(f"  Main scheduler start: {self.main_start_epoch}")
        print(f"  Final scheduler: {self.temp_config['final_scheduler']}")
        print(f"  Temperature groups: {sum(self.is_temp_group)}/{len(self.is_temp_group)}")
        
        super().__init__(optimizer, last_epoch, verbose)
    
    def get_lr(self):
        """PyTorch LRScheduler API implementation"""
        if not self._get_lr_called_within_step:
            import warnings
            warnings.warn("To get the last learning rate computed by the scheduler, "
                         "please use `get_last_lr()`.", UserWarning)

        lrs = []
        for i, (base_lr, is_temp) in enumerate(zip(self.base_lrs, self.is_temp_group)):
            if is_temp:
                # Apply advanced scheduling to temperature parameter group
                if self.last_epoch < self.freeze_epochs:
                    # Phase 1: Freeze
                    lr = 0.0
                elif self.last_epoch < self.main_start_epoch:
                    # Phase 2: Warmup
                    warmup_progress = (self.last_epoch - self.freeze_epochs) / self.warmup_epochs
                    lr = base_lr * warmup_progress
                else:
                    # Phase 3: Main scheduler
                    main_epoch = self.last_epoch - self.main_start_epoch
                    if self.temp_config['final_scheduler'] == 'cosine':
                        # Cosine annealing
                        lr = self.cosine_eta_min + (base_lr - self.cosine_eta_min) * \
                             (1 + math.cos(math.pi * main_epoch / self.cosine_epochs)) / 2
                    else:
                        # MultiStep (using saved settings)
                        # Adjust milestone based on main_start_epoch
                        adjusted_milestones = [m - self.main_start_epoch for m in self.temp_milestones if m >= self.main_start_epoch]
                        decay_count = sum(1 for m in adjusted_milestones if m <= main_epoch)
                        lr = base_lr * (self.temp_gamma ** decay_count)
            else:
                # Regular parameter group (existing MultiStepLR)
                decay_count = sum(1 for m in self.default_milestones if m <= self.last_epoch)
                lr = base_lr * (self.default_gamma ** decay_count)
            
            lrs.append(lr)
        
        return lrs

class CustomScheduler(_LRScheduler):
    """Custom scheduler supporting different schedulers per parameter group"""
    def __init__(self, optimizer, config, last_epoch=-1, verbose=False):
        # Extract and save only necessary values from config (prevent pickle issues)
        
        # Basic configuration (MultiStepLR)
        self.default_scheduler = config.train.optimizer['scheduler']
        self.default_milestones = set(config.train.optimizer.get('milestones', []))
        self.default_gamma = config.train.optimizer.get('gamma', 0.5)
        
        # Temperature parameter configuration
        self.temp_scheduler = config.train.optimizer.get('temperature_scheduler', self.default_scheduler)
        
        if self.temp_scheduler == 'MultiStepLR':
            self.temp_milestones = set(config.train.optimizer.get('temperature_milestones', config.train.optimizer.get('milestones', [])))
            self.temp_gamma = config.train.optimizer.get('temperature_gamma', self.default_gamma)
        elif self.temp_scheduler == 'CosineAnnealingLR':
            self.temp_T_max = config.train.optimizer.get('temperature_T_max', 1000)
            self.temp_eta_min = config.train.optimizer.get('temperature_eta_min', 0)
        
        # Extract temperature LR value
        self.temperature_lr = config.train.optimizer.get('temperature_lr', config.train.optimizer['lr'])
        
        # Check if each parameter group is a temperature group
        self.is_temp_group = []
        for group in optimizer.param_groups:
            # temperature parameter is a group with lr matching temperature_lr
            is_temp = abs(group['lr'] - self.temperature_lr) < 1e-10  # floating point comparison
            self.is_temp_group.append(is_temp)
        
        print(f"[SCHEDULER] Groups: {len(self.is_temp_group)}, Temperature groups: {sum(self.is_temp_group)}")
        print(f"[SCHEDULER] Default scheduler: {self.default_scheduler}")
        print(f"[SCHEDULER] Temperature scheduler: {self.temp_scheduler}")
        
        if self.temp_scheduler == 'MultiStepLR':
            print(f"[SCHEDULER] Temp milestones: {sorted(self.temp_milestones)}, gamma: {self.temp_gamma}")
        elif self.temp_scheduler == 'CosineAnnealingLR':
            print(f"[SCHEDULER] Temp cosine: T_max={self.temp_T_max}, eta_min={self.temp_eta_min}")
        
        super(CustomScheduler, self).__init__(optimizer, last_epoch, verbose)
    
    def get_lr(self):
        """PyTorch LRScheduler API 구현"""
        if not self._get_lr_called_within_step:
            import warnings
            warnings.warn("To get the last learning rate computed by the scheduler, "
                         "please use `get_last_lr()`.", UserWarning)

        lrs = []
        for i, (base_lr, is_temp) in enumerate(zip(self.base_lrs, self.is_temp_group)):
            if is_temp:
                # Temperature parameter group
                if self.temp_scheduler == 'MultiStepLR':
                    # MultiStep decay
                    decay_count = sum(1 for m in self.temp_milestones if m <= self.last_epoch)
                    lr = base_lr * (self.temp_gamma ** decay_count)
                elif self.temp_scheduler == 'CosineAnnealingLR':
                    # Cosine annealing
                    lr = self.temp_eta_min + (base_lr - self.temp_eta_min) * \
                         (1 + math.cos(math.pi * self.last_epoch / self.temp_T_max)) / 2
                else:
                    lr = base_lr
            else:
                # 일반 parameter group (MultiStepLR)
                decay_count = sum(1 for m in self.default_milestones if m <= self.last_epoch)
                lr = base_lr * (self.default_gamma ** decay_count)
            
            lrs.append(lr)
        
        return lrs

def make_lr_scheduler(optimizer, config):
    if config.train.optimizer['scheduler'] == 'MultiStepLR':
        # 1. 고급 Temperature 스케줄링 우선 확인
        has_advanced_temp_scheduling = (
            hasattr(config.model, 'ccp_deform_pixel_norm') and 
            config.model.ccp_deform_pixel_norm in ['trainable_softmax', 'trainable_softmax_softclamp'] and
            hasattr(config.train, 'temperature_advanced_scheduling') and
            config.train.temperature_advanced_scheduling['enabled']
        )
        
        if has_advanced_temp_scheduling:
            print("[SCHEDULER] Using AdvancedTemperatureScheduler (freeze → warmup → cosine/multistep)")
            scheduler = AdvancedTemperatureScheduler(optimizer, config)
            return scheduler
        
        # 2. 기본 Temperature 별도 스케줄링 확인
        has_temp_config = (
            hasattr(config.model, 'ccp_deform_pixel_norm') and 
            config.model.ccp_deform_pixel_norm in ['trainable_softmax', 'trainable_softmax_softclamp'] and
            ('temperature_scheduler' in config.train.optimizer or 
             'temperature_milestones' in config.train.optimizer or 
             'temperature_gamma' in config.train.optimizer)
        )
        
        if has_temp_config:
            print("[SCHEDULER] Using CustomScheduler for separate temperature scheduling")
            scheduler = CustomScheduler(optimizer, config)
        else:
            print("[SCHEDULER] Using standard MultiStepLR")
            scheduler = _scheduler_factory[config.train.optimizer['scheduler']](
                optimizer, 
                milestones=config.train.optimizer['milestones'],
                gamma=config.train.optimizer['gamma']
            )
    else:
        scheduler = _scheduler_factory[config.train.optimizer['scheduler']](optimizer,
                                                                            mode='max',
                                                                            patience=config.train.optimizer['patience'],
                                                                            factor=config.train.optimizer['gamma'])
    return scheduler


def set_lr_scheduler(scheduler, config):
    if hasattr(scheduler, 'milestones'):  # Standard MultiStepLR
        scheduler.milestones = Counter(config.train.optimizer['milestones'])
        scheduler.gamma = config.train.optimizer['gamma']
    elif isinstance(scheduler, CustomMultiStepLR):  # Custom MultiStepLR
        scheduler.default_milestones = set(config.train.optimizer['milestones'])
        scheduler.default_gamma = config.train.optimizer['gamma']
        scheduler.temp_milestones = set(config.train.optimizer.get('temperature_milestones', config.train.optimizer['milestones']))
        scheduler.temp_gamma = config.train.optimizer.get('temperature_gamma', config.train.optimizer['gamma'])

