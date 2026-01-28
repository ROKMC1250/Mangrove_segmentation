import os
import argparse
import yaml
import random
import torch as T
import numpy as np
import torch.multiprocessing as mp
import atexit
import warnings
import logging
from typing import Dict

# Suppress rasterio TIFF warnings globally
warnings.filterwarnings('ignore', message='.*TIFFReadDirectory.*')
warnings.filterwarnings('ignore', message='.*SamplesPerPixel.*')
logging.getLogger('rasterio').setLevel(logging.ERROR)
logging.getLogger('rasterio._env').setLevel(logging.ERROR)

from utils.logger import *
from torch.distributed import init_process_group, destroy_process_group
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from data.mangrove_dataset import MangroveDataset
from model.unet_plus_plus import UnetPlusPlus
from model.smp_models import create_model
from trainer.segmentation_trainer import SegmentationTrainer

# Import AE components conditionally
try:
    from data.ae_dataset import AEDataset
    from model.ae_enhanced_smp import create_ae_enhanced_model
    from trainer.ae_trainer import AETrainer
    AE_AVAILABLE = True
except ImportError as e:
    AE_AVAILABLE = False
    print(f"AE components not available: {e}")

def needs_224_input(model_name: str) -> bool:
    """
    Check if a model requires 224x224 input size instead of 256x256.
    
    Args:
        model_name: Name of the model (with or without AE prefix)
    
    Returns:
        True if model needs 224x224 input, False otherwise
    """
    # Remove AE prefix if present
    base_name = model_name[2:] if model_name.startswith('AE') else model_name
    
    # Models that require 224x224 input
    models_224 = ['DPT', 'UperNet']
    
    return base_name in models_224


def ddp_setup(rank: int, world_size: int, port):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = port
    T.cuda.set_device(rank)
    T.cuda.empty_cache()
    init_process_group('nccl', rank=rank, world_size=world_size)
    
    # Register cleanup function to avoid ProcessGroupNCCL warnings
    def cleanup_ddp():
        try:
            if T.distributed.is_initialized():
                T.cuda.empty_cache()
                destroy_process_group()
                T.cuda.empty_cache()
        except:
            pass  # Ignore cleanup errors
    
    atexit.register(cleanup_ddp)

def main(rank: int, world_size: int, train_args: Dict, port: int):
    setup_logging(train_args['train']['uid'])
    seed_everything(train_args['train']['seed'])
    if not train_args['train']['no_ddp']:
        ddp_setup(rank, world_size, port)
    
    logger = get_logger(__name__, rank)

    logger.info('Instantiating model and trainer agent')
    
    # Dynamic model loading based on config
    model_name = train_args['model']['name']
    
    # Check if it's an AE-enhanced model
    if model_name.startswith('AE'):
        # AE-enhanced versions of all models (including UNet++)
        if not AE_AVAILABLE:
            raise RuntimeError("AE components are not available. Please check your installation.")
        
        # Extract base model name (e.g., "AEMAnet" -> "MAnet", "AEUnetPlusPlus" -> "UnetPlusPlus")
        base_model_name = model_name[2:]  # Remove "AE" prefix
        
        if base_model_name in ['UnetPlusPlus', 'MAnet', 'PAN', 'DeepLabV3Plus', 'Segformer', 'FPN', 'DPT', 'UperNet']:
            # Create AE-enhanced version using unified SMP approach
            model = create_ae_enhanced_model(base_model_name, **train_args['model']['args'])
            trainer_class = AETrainer
            dataset_class = AEDataset
        else:
            raise ValueError(f"Unknown AE model base: {base_model_name}")
    elif model_name in ['UnetPlusPlus', 'MAnet', 'PAN', 'DeepLabV3Plus', 'Segformer', 'FPN', 'DPT', 'UperNet']:
        # Standard SMP models
        if model_name == 'UnetPlusPlus':
            # Use original implementation for backward compatibility
            model = UnetPlusPlus(**train_args['model']['args'])
        else:
            # Use SMP factory for new models
            model = create_model(model_name, **train_args['model']['args'])
        trainer_class = SegmentationTrainer
        dataset_class = MangroveDataset
    else:
        raise ValueError(f"Unknown model name: {model_name}. Supported models: UnetPlusPlus, MAnet, PAN, DeepLabV3Plus, Segformer, FPN, DPT, UperNet, and their AE variants (prefix with 'AE').")
    
    trainer = trainer_class(
        model=model, 
        gpu_id=rank, 
        args=train_args, 
        log_enabled=not train_args['train']['no_save']
    )

    logger.info('Preparing dataset')
    # Determine target size based on model
    target_size = 224 if needs_224_input(model_name) else 256
    logger.info(f'Using target size: {target_size}x{target_size} for model: {model_name}')
    
    train_dataset = dataset_class(root_dir=train_args['data']['root_dir'], split='train', target_size=target_size)
    val_dataset = dataset_class(root_dir=train_args['data']['root_dir'], split='validation', target_size=target_size)
    logger.info(f'Train dataset size: {len(train_dataset)}')
    logger.info(f'Val dataset size: {len(val_dataset)}')

    logger.info(f'Using {world_size} GPU(s)')
    if train_args['train'].get('model_path') is not None:
        trainer.load_checkpoint(train_args['train']['model_path'])

    if not train_args['train']['batch_size'] % world_size == 0:
        logger.error(f"Batch size {train_args['train']['batch_size']} must be divisible by the number of GPUs {world_size}")
        if not train_args['train']['no_ddp']:
            destroy_process_group()
        return

    logger.info('Instantiating dataloader')
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=train_args['train']['batch_size'] // world_size,
        shuffle=False, # Shuffle is handled by DistributedSampler
        num_workers=train_args['train']['n_workers'],
        pin_memory=True,
        sampler=DistributedSampler(train_dataset) if not train_args['train']['no_ddp'] else None,
        persistent_workers=True,
        prefetch_factor=2,  # Reduce prefetch to save memory
        drop_last=True,  # Drop incomplete batches to avoid memory accumulation
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=train_args['train']['batch_size'] // world_size,
        shuffle=False,
        num_workers=train_args['train']['n_workers'],
        pin_memory=True,
        sampler=DistributedSampler(val_dataset) if not train_args['train']['no_ddp'] else None,
        persistent_workers=True,
        prefetch_factor=2,  # Reduce prefetch to save memory
        drop_last=True,  # Drop incomplete batches to avoid memory accumulation
    )

    trainer.do_training(train_dataloader, val_dataloader)

    if not train_args['train']['no_ddp']:
        # More explicit cleanup for DDP
        T.cuda.empty_cache()
        destroy_process_group()
        T.cuda.empty_cache()

def seed_everything(seed: int):    
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    T.manual_seed(seed)
    T.cuda.manual_seed(seed)
    T.backends.cudnn.deterministic = True
    T.backends.cudnn.benchmark = False # Set to False for reproducibility

def get_args_parser():
    parser = argparse.ArgumentParser('Mangrove Segmentation Train', add_help=False)
    parser.add_argument('--uid', type=str, help='unique id for the run', default=None)
    parser.add_argument('--config', type=str, help='path to yaml config', default='config/mangrove_config.yaml')
    parser.add_argument('--model-path', type=str, help='ckpt path to continue', default=None)
    parser.add_argument('--patience', type=int, help='patience for early stopping', default=-1)
    parser.add_argument('--seed', type=int, help='random seed', default=None)
    parser.add_argument('--port', type=int, help='DDP port', default=None)
    parser.add_argument('--no-ddp', action='store_true', help='disable DDP')
    parser.add_argument('--no-save', action='store_true', help='disable logging and checkpoint saving (for debugging)')
    return parser.parse_args()

if __name__ == '__main__':
    args = get_args_parser()
    with open(args.config, 'r') as f:
        train_args = yaml.safe_load(f)
    
    # Override config with command-line arguments if provided
    if args.seed is not None:
        train_args['train']['seed'] = args.seed
    elif train_args['train'].get('seed', None) is None:
        train_args['train']['seed'] = random.randint(0, 1000000)
    
    if args.uid is not None:
        train_args['train']['uid'] = args.uid
    elif train_args['train'].get('uid', None) is None:
        train_args['train']['uid'] = ''.join(random.choices('0123456789abcdef', k=8))
    
    if args.model_path is not None:
        train_args['train']['model_path'] = args.model_path
    if args.patience != -1:
        train_args['train']['patience'] = args.patience 
        
    train_args['train']['no_ddp'] = args.no_ddp
    train_args['train']['no_save'] = args.no_save

    if not train_args['train']['no_ddp']:
        world_size = T.cuda.device_count()
        port = str(random.randint(10000, 60000)) if args.port is None else str(args.port)
        mp.spawn(main, nprocs=world_size, args=(world_size, train_args, port))
    else:
        main(0, 1, train_args, 0)
