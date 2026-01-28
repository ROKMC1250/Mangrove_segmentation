import torch as T
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import segmentation_models_pytorch as smp
from typing import Dict, Tuple
import csv
import os

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


# separation_loss_with_stats function removed for performance - was causing 8x slowdown


def separation_loss_bounded(Z, target, tau=2.0, eps=1e-4, var_floor=1e-6):
    """
    0~1 범위, 작아질수록 좋음. (역수형)
    Z: BxDxhxw, target: BxHxW {0,1}
    
    Returns:
        loss: bounded loss value (0~1)
        sep_val: raw separation score for EMA update
    """
    B, D, H, W = Z.shape
    N = H * W
    Zf = Z.view(B, D, N)
    Yf = target.view(B, N)

    m1 = (Yf == 1).float()  # BxN
    m0 = (Yf == 0).float()
    n1 = m1.sum()
    n0 = m0.sum()
    if (n1 < 1) or (n0 < 1):
        # 클래스 한쪽이 비면 기여하지 않도록 0 리턴(= "이미 충분히 좋다"로 취급)
        zero_loss = Z.new_zeros((), dtype=Z.dtype, device=Z.device) + 0.0
        zero_sep = Z.new_zeros((), dtype=Z.dtype, device=Z.device) + float('nan')  # EMA 업데이트 안함
        return zero_loss, zero_sep

    sum1 = (Zf * m1.unsqueeze(1)).sum(dim=(0, 2))  # D
    sum0 = (Zf * m0.unsqueeze(1)).sum(dim=(0, 2))  # D
    mu1 = sum1 / (n1 + eps)
    mu0 = sum0 / (n0 + eps)

    var1 = ((Zf - mu1.view(1, D, 1))**2 * m1.unsqueeze(1)).sum(dim=(0, 2)) / (n1 + eps)
    var0 = ((Zf - mu0.view(1, D, 1))**2 * m0.unsqueeze(1)).sum(dim=(0, 2)) / (n0 + eps)
    denom = T.clamp(var1 + var0 + eps, min=var_floor)

    sep = ((mu1 - mu0) ** 2) / denom          # D, >=0
    sep_score = sep.mean()                    # scalar, >=0

    # 0~1 bounded, 작아질수록 좋음
    loss = 1.0 / (1.0 + sep_score / tau)
    
    return loss, sep_score

        

def orthogonality_loss(conv1x1_weight):
    """
    Enforce columns of W to be orthogonal.
    conv1x1_weight: (D, 64, 1, 1) for 64→D mapping
    L_ortho = || W^T W - I ||_F^2
    """
    if conv1x1_weight.numel() == 0:
        return conv1x1_weight.new_zeros((), dtype=conv1x1_weight.dtype, device=conv1x1_weight.device)
    
    w = conv1x1_weight.view(conv1x1_weight.shape[0], -1)  # D x 64
    W = w.t()  # 64 x D
    M = W.t() @ W  # D x D
    I = T.eye(M.shape[0], device=M.device, dtype=M.dtype)
    return F.mse_loss(M, I, reduction='mean')  # Use mean instead of sum for stability


def tv_loss(Z):
    """
    Anisotropic total variation on Z (BxDxhxw).
    """
    if Z.numel() == 0 or Z.shape[2] <= 1 or Z.shape[3] <= 1:
        return Z.new_zeros((), dtype=Z.dtype, device=Z.device)
    
    dh = (Z[:, :, 1:, :] - Z[:, :, :-1, :]).abs().mean()
    dw = (Z[:, :, :, 1:] - Z[:, :, :, :-1]).abs().mean()
    tv_value = dh + dw
    # Ensure it's a scalar tensor
    if tv_value.dim() > 0:
        tv_value = tv_value.mean()
    return tv_value


# compute_iou function removed - now using smp.metrics for consistency with SegmentationTrainer


class AETrainer(BaseTrainer):
    """
    A specialized trainer for AE-enhanced segmentation tasks, inheriting from BaseTrainer.
    """
    def __init__(self, model: nn.Module, gpu_id: int, args: Dict, log_enabled: bool = True):
        super().__init__(model, gpu_id, args, log_enabled=log_enabled)
        # The metric to track for saving the best model. We want to maximize IoU score.
        self.tracker.direction = 'max'
        
        # Initialize loss components (identical to SegmentationTrainer)
        # Use the same loss functions as SegmentationTrainer for consistency
        bce_weight = self.args['model']['args'].get('bce_weight', 2.0)
        self.dice_loss = smp.losses.DiceLoss(mode='binary', from_logits=True, eps=1e-7)
        self.bce_loss = smp.losses.SoftBCEWithLogitsLoss()
        self.combined_loss = CombinedLoss(self.dice_loss, self.bce_loss, w1=1.0, w2=bce_weight)
        
        # Get loss weights from config and ensure they are floats
        loss_config = self.args['train'].get('loss', {})
        self.lambda_sep = float(loss_config.get('lambda_sep'))
        self.lambda_ortho = float(loss_config.get('lambda_ortho'))
        self.lambda_tv = float(loss_config.get('lambda_tv'))
        self.lambda_mag = float(loss_config.get('lambda_mag'))
        
        # Note: lambda_dice is now handled by CombinedLoss, not separately
        
        # Initialize CSV logging for 9-channel statistics
        self.channel_stats_file = None
        self.channel_stats_writer = None
        self._setup_channel_stats_logging()
        
        # Initialize separation loss tau EMA parameters
        self.sep_tau_ema = None  # Will be initialized on first update
        self.sep_ema_momentum = 0.99  # EMA momentum for tau update
        
        # Simplified tracking - no complex statistics collection
        self.loss_history = {
            'separation': [],
            'orthogonality': [],
            'tv': [],
            'segmentation': []
        }
    
    def _get_model(self):
        """Get the actual model, handling DDP wrapping."""
        if hasattr(self.model, 'module'):
            return self.model.module
        return self.model

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
        # AE trainer uses complex multi-component loss, so we return a dummy here
        # The actual loss computation is done in the step method
        return nn.CrossEntropyLoss()

    def step(self, images: T.Tensor, ae_features: T.Tensor, labels: T.Tensor) -> T.Tensor:
        """
        Implements the abstract method for a single training/validation step.
        This includes forward pass, loss calculation, and metric calculation.
        """
        images, ae_features, labels = images.to(self.gpu_id), ae_features.to(self.gpu_id), labels.to(self.gpu_id)
        
        # Forward pass
        logits, Z = self.model(images, ae_features)
        
        # Log 9-channel statistics for mangrove vs non-mangrove (every step during training, first 5 epochs only)
        if self.model.training and self.tracker.epoch <= 5 and self.is_main_process:
            self._log_nine_channel_statistics(images, Z, labels)
        

        # --- segmentation loss (identical to SegmentationTrainer) ---
        # Use the exact same loss calculation as SegmentationTrainer
        loss_seg = self.combined_loss(logits, labels)  # Note: labels as float, not .long()

        # --- separability loss on Z with adaptive tau ---
        tau = 2.0 if (self.sep_tau_ema is None) else float(self.sep_tau_ema)
        loss_sep_bounded, sep_val = separation_loss_bounded(Z, labels, tau=tau)
        
        # sep EMA 업데이트
        if not T.isnan(sep_val):
            s = float(sep_val.detach().cpu())
            if self.sep_tau_ema is None:
                self.sep_tau_ema = s
            else:
                m = self.sep_ema_momentum
                self.sep_tau_ema = m * self.sep_tau_ema + (1 - m) * s
            tau = float(self.sep_tau_ema)
        
        loss_sep = loss_sep_bounded
        
        # Log simple separation loss value and tau to tensorboard (much faster)
        if self.tracker.step_counter % 100 == 0 and self.is_main_process:
            self.write_summary('Loss_Components/Separation', loss_sep.item(), self.tracker.step_counter)
            if self.sep_tau_ema is not None:
                self.write_summary('Loss_Components/Separation_Tau', tau, self.tracker.step_counter)
                self.write_summary('Loss_Components/Separation_Raw', sep_val.item(), self.tracker.step_counter)

        # --- orthogonality on projection W (conditional) ---
        if self.lambda_ortho > 0:
            actual_model = self._get_model()
            loss_ortho = orthogonality_loss(actual_model.proj.weight)
        else:
            loss_ortho = T.tensor(0.0, device=Z.device, dtype=Z.dtype)

        # --- total variation on Z (conditional) ---
        if self.lambda_tv > 0:
            loss_tv = tv_loss(Z)
        else:
            loss_tv = T.tensor(0.0, device=Z.device, dtype=Z.dtype)



        # --- total ---

        # Ensure all loss components are scalar tensors
        loss_sep = loss_sep if loss_sep.dim() == 0 else loss_sep.mean()
        loss_ortho = loss_ortho if loss_ortho.dim() == 0 else loss_ortho.mean()
        loss_tv = loss_tv if loss_tv.dim() == 0 else loss_tv.mean()

        
        # Use lambdas directly as float values - no need to convert to tensors
        # PyTorch automatically handles scalar * tensor operations
        
        loss = (
            loss_seg
            + self.lambda_sep   * loss_sep
            # + self.lambda_ortho * loss_ortho
            # + self.lambda_tv    * loss_tv
        )
        
        if T.isnan(logits).any():
            if T.isnan(images).any():
                print("nan is in images")
            if T.isnan(ae_features).any():
                print("nan is in ae_features")
            print(f"Loss: {loss}")
            
        # During validation, calculate and log metrics (identical to SegmentationTrainer)
        if not self.model.training:
            preds = T.sigmoid(logits)  # Use sigmoid for binary segmentation (same as SegmentationTrainer)
            
            # Calculate dice score using smp metrics (same as SegmentationTrainer)
            tp, fp, fn, tn = smp.metrics.get_stats(
                preds, labels.long(), mode='binary', threshold=0.5
            )
            dice_score = smp.metrics.f1_score(tp, fp, fn, tn, reduction='micro-imagewise')
            
            # The BaseTrainer's validation loop will check this metric
            self.tracker.last_metric = dice_score.item()
            
            # Also log individual loss components and metrics to TensorBoard
            self.write_summary('Validation/Dice Score', dice_score.item(), self.tracker.val_step_counter)
            self.write_summary('Validation/Loss_Seg', loss_seg.item(), self.tracker.val_step_counter)
            self.write_summary('Validation/Loss_Sep', loss_sep.item(), self.tracker.val_step_counter)
            self.write_summary('Validation/Loss_Ortho', loss_ortho.item(), self.tracker.val_step_counter)
            self.write_summary('Validation/Loss_TV', loss_tv.item(), self.tracker.val_step_counter)
            
            # Store loss values for simple tracking (only during validation, limit memory usage)
            if len(self.loss_history['separation']) < 1000:  # Limit to prevent memory bloat
                self.loss_history['separation'].append(loss_sep.item())
                self.loss_history['orthogonality'].append(loss_ortho.item())
                self.loss_history['tv'].append(loss_tv.item())
                self.loss_history['segmentation'].append(loss_seg.item())

        return loss, logits
    
    def on_epoch_end(self, epoch: int):
        """
        Called at the end of each epoch to log simple loss statistics.
        """
        if not self.is_main_process or not self.log_enabled:
            return
            
        # Close CSV file after 5 epochs to save resources
        if epoch >= 5 and hasattr(self, 'channel_stats_file') and self.channel_stats_file:
            self.channel_stats_file.close()
            self.channel_stats_file = None
            self.channel_stats_writer = None
            if self.logger:
                self.logger.info("Channel statistics logging completed (first 5 epochs).")
            
        # Log epoch-level loss summaries (much simpler and faster)
        if len(self.loss_history['separation']) > 0:
            avg_sep = sum(self.loss_history['separation']) / len(self.loss_history['separation'])
            avg_ortho = sum(self.loss_history['orthogonality']) / len(self.loss_history['orthogonality'])
            avg_tv = sum(self.loss_history['tv']) / len(self.loss_history['tv'])
            avg_seg = sum(self.loss_history['segmentation']) / len(self.loss_history['segmentation'])
            
            self.write_summary(f'Epoch_Summary/Avg_Separation_Loss', avg_sep, epoch)
            self.write_summary(f'Epoch_Summary/Avg_Orthogonality_Loss', avg_ortho, epoch)
            self.write_summary(f'Epoch_Summary/Avg_TV_Loss', avg_tv, epoch)
            self.write_summary(f'Epoch_Summary/Avg_Segmentation_Loss', avg_seg, epoch)
            
            if self.logger:
                self.logger.info(f"Epoch {epoch} - Avg losses: Sep={avg_sep:.4f}, Ortho={avg_ortho:.4f}, TV={avg_tv:.4f}, Seg={avg_seg:.4f}")
            
            # Clear history for next epoch
            for key in self.loss_history:
                self.loss_history[key] = []
    
# _average_epoch_stats method removed - no longer needed after simplification
    
# _save_band_separation_plot method removed - complex visualization no longer needed
    
    def _setup_channel_stats_logging(self):
        """
        Setup CSV file for logging 9-channel statistics.
        """
        if not self.is_main_process:
            return
            
        # Create CSV file for channel statistics
        log_dir = os.path.join(self.log_dir, 'channel_stats')
        os.makedirs(log_dir, exist_ok=True)
        
        csv_path = os.path.join(log_dir, 'nine_channel_stats.csv')
        self.channel_stats_file = open(csv_path, 'w', newline='')
        
        # Setup CSV writer with headers
        fieldnames = ['epoch', 'step']
        # Add headers for 6 satellite channels
        for i in range(6):
            fieldnames.extend([f'sat_ch{i}_mangrove_mean', f'sat_ch{i}_non_mangrove_mean'])
        # Add headers for 3 projected AE channels
        for i in range(3):
            fieldnames.extend([f'ae_ch{i}_mangrove_mean', f'ae_ch{i}_non_mangrove_mean'])
            
        self.channel_stats_writer = csv.DictWriter(self.channel_stats_file, fieldnames=fieldnames)
        self.channel_stats_writer.writeheader()
    
    def _log_nine_channel_statistics(self, images: T.Tensor, projected_features: T.Tensor, labels: T.Tensor):
        """
        Log 9-channel (6 satellite + 3 projected AE) statistics for mangrove vs non-mangrove regions.
        """
        if not self.is_main_process or self.channel_stats_writer is None:
            return
            
        # Combine 6 satellite channels + 3 projected AE channels
        nine_channels = T.cat([images, projected_features], dim=1)  # (B, 9, H, W)
        
        # Flatten spatial dimensions: (B, 9, H*W)
        B, C, H, W = nine_channels.shape
        channels_flat = nine_channels.view(B, C, -1)  # (B, 9, H*W)
        labels_flat = labels.view(B, -1)  # (B, H*W)
        
        # Create masks for mangrove (1) and non-mangrove (0) pixels
        mangrove_mask = (labels_flat == 1)  # (B, H*W)
        non_mangrove_mask = (labels_flat == 0)  # (B, H*W)
        
        # Calculate channel-wise means for each class across all pixels in the batch
        channel_stats = {}
        channel_stats['epoch'] = self.tracker.epoch
        channel_stats['step'] = self.tracker.step_counter
        
        for ch in range(9):
            # Get channel data: (B, H*W)
            ch_data = channels_flat[:, ch, :]  # (B, H*W)
            
            # Calculate mangrove pixel mean
            mangrove_pixels = ch_data[mangrove_mask]  # All mangrove pixels across batch
            if len(mangrove_pixels) > 0:
                mangrove_mean = mangrove_pixels.mean().item()
            else:
                mangrove_mean = 0.0
                
            # Calculate non-mangrove pixel mean 
            non_mangrove_pixels = ch_data[non_mangrove_mask]  # All non-mangrove pixels across batch
            if len(non_mangrove_pixels) > 0:
                non_mangrove_mean = non_mangrove_pixels.mean().item()
            else:
                non_mangrove_mean = 0.0
            
            # Store in stats dict
            if ch < 6:  # Satellite channels
                channel_stats[f'sat_ch{ch}_mangrove_mean'] = mangrove_mean
                channel_stats[f'sat_ch{ch}_non_mangrove_mean'] = non_mangrove_mean
            else:  # Projected AE channels
                ae_ch = ch - 6
                channel_stats[f'ae_ch{ae_ch}_mangrove_mean'] = mangrove_mean
                channel_stats[f'ae_ch{ae_ch}_non_mangrove_mean'] = non_mangrove_mean
        
        # Write to CSV
        self.channel_stats_writer.writerow(channel_stats)
        self.channel_stats_file.flush()  # Ensure data is written immediately
    
    def __del__(self):
        """
        Clean up CSV file when trainer is destroyed.
        """
        if hasattr(self, 'channel_stats_file') and self.channel_stats_file:
            self.channel_stats_file.close()
