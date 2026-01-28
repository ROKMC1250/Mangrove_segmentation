import torch as T
import torch.nn as nn
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import segmentation_models_pytorch as smp
from typing import Dict

from trainer.base_trainer import BaseTrainer


class CombinedLoss(nn.Module):
    def __init__(self, loss1, loss2, w1=1.0, w2=1.0):
        super().__init__()
        self.loss1 = loss1
        self.loss2 = loss2
        self.w1 = w1
        self.w2 = w2

    def forward(self, prediction, target):
        loss1 = self.loss1(prediction, target)
        loss2 = self.loss2(prediction, target)
        return loss1 + loss2


class SegmentationTrainer(BaseTrainer):
    """
    A specialized trainer for segmentation tasks, inheriting from BaseTrainer.
    """
    def __init__(self, model: nn.Module, gpu_id: int, args: Dict, log_enabled: bool = True):
        super().__init__(model, gpu_id, args, log_enabled=log_enabled)
        # The metric to track for saving the best model. We want to maximize Dice score.
        self.tracker.direction = 'max'

    def _get_optimizer(self) -> T.optim.Optimizer:
        """Overrides the base optimizer to support various types from config."""
        cfg = self.args['train']['optimizer']
        lr = self.args['train']['learning_rate']
        
        if cfg['name'].lower() == 'adam':
            return Adam(self.model.parameters(), lr=lr, **cfg.get('args', {}))
        elif cfg['name'].lower() == 'adamw':
            return AdamW(self.model.parameters(), lr=lr, **cfg.get('args', {}))
        else:
            self.logger.warning(f"Optimizer {cfg['name']} not in custom list, falling back to BaseTrainer default.")
            # The base implementation uses AdamW by default.
            return super()._get_optimizer()

    def _get_scheduler(self) -> T.optim.lr_scheduler.LRScheduler:
        """Implements the abstract method to create a learning rate scheduler from config."""
        cfg = self.args['train'].get('scheduler')
        if not cfg:
            # Create a scheduler that does nothing if not specified
            return T.optim.lr_scheduler.LambdaLR(self.optim, lr_lambda=lambda epoch: 1.0)
        
        name = cfg.get('name')
        if name == 'CosineAnnealingWarmRestarts':
            return CosineAnnealingWarmRestarts(self.optim, **cfg.get('args', {}))
        elif name == 'ReduceLROnPlateau':
            return T.optim.lr_scheduler.ReduceLROnPlateau(self.optim, **cfg.get('args', {}))
        else:
            raise ValueError(f"Scheduler '{name}' not supported.")

    def _get_loss_fn(self) -> nn.Module:
        """Implements the abstract method to create the loss function from config."""
        cfg = self.args['train'].get('loss')
        if not cfg or cfg.get('name') == 'BCE':
            return nn.BCEWithLogitsLoss()
        
        name = cfg.get('name')
        if name == 'DiceBCE':
            # A common and effective loss for segmentation
            bce_weight = self.args['model']['args'].get('bce_weight')
            dice_loss = smp.losses.DiceLoss(mode='binary', from_logits=True, eps=1e-7)
            bce_loss = smp.losses.SoftBCEWithLogitsLoss()
            return CombinedLoss(dice_loss, bce_loss, w1=1.0, w2=bce_weight)
        else:
            raise ValueError(f"Loss function '{name}' not supported.")

    def step(self, images: T.Tensor, labels: T.Tensor) -> T.Tensor:
        """
        Implements the abstract method for a single training/validation step.
        This includes forward pass, loss calculation, and metric calculation.
        """
        images, labels = images.to(self.gpu_id), labels.to(self.gpu_id)
        
        # Forward pass
        outputs = self.model(images)
        # Calculate loss
        loss = self.loss_fn(outputs, labels)
        
        if T.isnan(outputs).any():
            if T.isnan(images).any():
                print("nan is in images")
            print(loss)
        # During validation, calculate and log metrics
        if not self.model.training:
            preds = T.sigmoid(outputs) # Get probabilities for metric calculation
            
            tp, fp, fn, tn = smp.metrics.get_stats(
                preds, labels.long(), mode='binary', threshold=0.5
            )
            dice_score = smp.metrics.f1_score(tp, fp, fn, tn, reduction='micro-imagewise')
            
            # The BaseTrainer's validation loop will check this metric
            self.tracker.last_metric = dice_score.item()
            
            # Also log it to TensorBoard for real-time tracking
            self.write_summary('Validation/Dice Score', dice_score.item(), self.tracker.val_step_counter)

        return loss, outputs 