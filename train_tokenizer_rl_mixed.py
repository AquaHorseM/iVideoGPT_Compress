import argparse
import json
import sys
import os
import cv2
import time
from pathlib import Path
import psutil
from typing import Dict, Callable, Optional
from collections import deque

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
import numpy as np

from tqdm import tqdm
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version, is_wandb_available

from ivideogpt.vq_model import CompressiveVQModel, LPIPS
from ivideogpt.data import *

if is_wandb_available():
    import wandb

check_min_version("0.22.0.dev0")
logger = get_logger(__name__, log_level="INFO")


from rl.value import SimpleValueNetwork
from rl.reward import reward_registry
from rl.utils import AverageMeter, save_checkpoint, compute_combined_reward
from rl.reward_bp import train_with_reward_backprop
from rl.ppo import train_with_ppo, RolloutBuffer

# ============================================================================
# Main Training Functions
# ============================================================================


def parse_args():
    parser = argparse.ArgumentParser(description="RL finetuning for VQGAN")
    
    # Original args
    parser.add_argument("--pretrained_model_name_or_path", type=str, required=True,
                        help="Path to pretrained VQGAN model")
    parser.add_argument("--output_dir", type=str, default="vqgan-rl-output")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--train_batch_size", type=int, default=8)
    parser.add_argument("--max_train_steps", type=int, default=100000)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--value_lr", type=float, default=1e-4)
    
    # RL-specific args
    parser.add_argument("--rl_algorithm", type=str, default="ppo", choices=["ppo", "reinforce", "reward_backprop"])
    parser.add_argument("--reward_functions", type=str, nargs="+", 
                        default=["perceptual", "temporal_consistency"],
                        help="List of reward functions to use")
    parser.add_argument("--reward_weights", type=float, nargs="+", default=None,
                        help="Weights for each reward function")
    
    # PPO hyperparameters
    parser.add_argument("--ppo_epochs", type=int, default=4)
    parser.add_argument("--clip_epsilon", type=float, default=0.2)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae_lambda", type=float, default=0.95)
    parser.add_argument("--value_loss_coef", type=float, default=0.5)
    parser.add_argument("--entropy_coef", type=float, default=0.01)
    
    # Commitment loss weights (keep from original training)
    parser.add_argument("--commit_loss_weight", type=float, default=1.0)
    parser.add_argument("--dyna_commit_loss_weight", type=float, default=1.0)
    
    # Training args
    parser.add_argument("--context_length", type=int, default=1)
    parser.add_argument("--segment_length", type=int, default=5)
    parser.add_argument("--segment_horizon", type=int, default=16)
    parser.add_argument("--video_stepsize", type=int, default=1)
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--dataset_name", type=str, default="robotic")
    parser.add_argument("--oxe_data_mixes_type", type=str, default="frac")
    parser.add_argument("--dataloader_num_workers", type=int, default=4)
    parser.add_argument("--mixed_precision", type=str, default="no", choices=["no", "fp16", "bf16"])
    
    # Logging
    parser.add_argument("--log_steps", type=int, default=50)
    parser.add_argument("--validation_steps", type=int, default=1000)
    parser.add_argument("--checkpointing_steps", type=int, default=5000)
    parser.add_argument("--log_image_steps", type=int, default=500)
    parser.add_argument("--report_to", type=str, default="tensorboard")
    parser.add_argument("--tracker_project_name", type=str, default="vqgan-rl-training")
    
    # Additional
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--sthsth_root_path", type=str, default=None)
    parser.add_argument("--no_aug", action="store_true")
    parser.add_argument("--exp_name", type=str, default=None)
    
    args = parser.parse_args()
    
    # Set default reward weights if not provided
    if args.reward_weights is None:
        args.reward_weights = [1.0] * len(args.reward_functions)
    elif len(args.reward_weights) != len(args.reward_functions):
        raise ValueError("Number of reward_weights must match number of reward_functions")
    
    return args


def main():
    args = parse_args()
    args.output_dir = os.path.join(
        args.output_dir, 
        time.strftime("%Y-%m-%d-%X", time.localtime()) + 
        ("" if args.exp_name is None else f"-{args.exp_name}")
    )
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Setup accelerator
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
    )
    
    # Initialize tracking
    if accelerator.is_main_process:
        tracker_config = dict(vars(args))
        accelerator.init_trackers(args.tracker_project_name, tracker_config)
        
        with open(os.path.join(args.output_dir, "cmd.sh"), "w") as f:
            f.write("python " + " ".join(sys.argv))
    
    # Set seed
    if args.seed is not None:
        set_seed(args.seed, device_specific=True)
    
    # Load model
    logger.info(f"Loading model from {args.pretrained_model_name_or_path}")
    model = CompressiveVQModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder=None,
        use_safetensor=True,
        low_cpu_mem_usage=False,
        ignore_mismatched_sizes=True
    )
    
    if args.context_length != model.context_length:
        logger.warning(f"Changing context length from {model.context_length} to {args.context_length}")
        model.set_context_length(args.context_length)
    
    # Initialize value network for PPO
    value_net = None
    value_optimizer = None
    if args.rl_algorithm == "ppo":
        value_net = SimpleValueNetwork(latent_dim=model.config.latent_channels)
        value_optimizer = torch.optim.AdamW(value_net.parameters(), lr=args.value_lr)
    
    # LPIPS model for perceptual reward
    lpips_model = LPIPS().to(accelerator.device).eval()
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(0.9, 0.999),
        weight_decay=0.0,
        eps=1e-8
    )
    
    # Learning rate scheduler
    lr_scheduler = get_scheduler(
        "constant_with_warmup",
        optimizer=optimizer,
        num_training_steps=args.max_train_steps,
        num_warmup_steps=500
    )
    
    # Data loader
    augmentation_args = {
        'brightness': [0.9, 1.1],
        'contrast': [0.9, 1.1],
        'saturation': [0.9, 1.1],
        'hue': [-0.05, 0.05],
        'random_resized_crop_scale': (0.8, 1.0),
        'random_resized_crop_ratio': (0.9, 1.1),
        'no_aug': args.no_aug,
    }
    segment_args = {
        'random_selection': False,
        'random_shuffle': False,
        'goal_conditioned': False,
        'segment_length': args.segment_length,
        'context_length': args.context_length,
        'stepsize': args.video_stepsize,
        'segment_horizon': args.segment_horizon,
    }
    
    train_dataloader = SimpleRoboticDataLoaderv2(
        parent_dir=args.dataset_path,
        datasets=DATASET_NAMED_MIXES[args.oxe_data_mixes_type],
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
        train=True,
        image_size=args.resolution,
        sthsth_root_path=args.sthsth_root_path,
        **augmentation_args,
        **segment_args,
    )
    
    # Prepare with accelerator
    if value_net is not None and value_optimizer is not None:
        model, value_net, optimizer, value_optimizer, lr_scheduler = accelerator.prepare(
            model, value_net, optimizer, value_optimizer, lr_scheduler
        )
    else:
        model, optimizer, lr_scheduler = accelerator.prepare(
            model, optimizer, lr_scheduler
        )
    
    # Training info
    logger.info("***** Running RL Finetuning *****")
    logger.info(f"  Algorithm = {args.rl_algorithm}")
    logger.info(f"  Reward functions = {args.reward_functions}")
    logger.info(f"  Reward weights = {args.reward_weights}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    
    # Rollout buffer for PPO
    rollout_buffer = RolloutBuffer() if args.rl_algorithm == "ppo" else None
    
    # Training loop
    global_step = 0
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    
    # Metrics
    batch_time_m = AverageMeter()
    reward_m = AverageMeter()
    loss_m = AverageMeter()
    
    model.train()
    end = time.time()
    
    for epoch in range(1000):  # Arbitrary large number
        for i, batch in enumerate(train_dataloader):
            with accelerator.accumulate(model):
                optimizer.zero_grad()
                if value_optimizer is not None:
                    value_optimizer.zero_grad()
                
                # Choose training algorithm
                if args.rl_algorithm == "reward_backprop":
                    loss, reward_dict, extra_metrics = train_with_reward_backprop(
                        model, batch, args, lpips_model, accelerator
                    )
                elif args.rl_algorithm == "ppo":
                    result = train_with_ppo(
                        model, value_net, batch, rollout_buffer, args, lpips_model, accelerator, global_step
                    )
                    if result[0] is None:
                        # Not enough samples yet
                        continue
                    loss, reward_dict, extra_metrics = result
                else:
                    raise ValueError(f"Unknown RL algorithm: {args.rl_algorithm}")
                
                # Backward pass
                accelerator.backward(loss)
                
                # Gradient clipping
                if args.max_grad_norm is not None:
                    accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    if value_net is not None:
                        accelerator.clip_grad_norm_(value_net.parameters(), args.max_grad_norm)
                
                # Optimizer step
                optimizer.step()
                lr_scheduler.step()
                if value_optimizer is not None:
                    value_optimizer.step()
            
            # Update metrics
            if accelerator.sync_gradients:
                global_step += 1
                progress_bar.update(1)
                
                batch_time_m.update(time.time() - end)
                loss_m.update(loss.item())
                reward_m.update(reward_dict["reward/total"])
                end = time.time()
                
                # Logging
                if global_step % args.log_steps == 0:
                    logs = {
                        "loss": loss_m.avg,
                        "lr": lr_scheduler.get_last_lr()[0],
                        "batch_time": batch_time_m.avg,
                        **reward_dict,
                        **extra_metrics
                    }
                    accelerator.log(logs, step=global_step)
                    
                    progress_bar.set_postfix({
                        "loss": f"{loss_m.avg:.4f}",
                        "reward": f"{reward_m.avg:.4f}"
                    })
                    
                    # Reset meters
                    batch_time_m.reset()
                    loss_m.reset()
                    reward_m.reset()
                
                # Save images
                if global_step % args.log_image_steps == 0 and accelerator.is_main_process:
                    with torch.no_grad():
                        save_path = os.path.join(args.output_dir, "images", f"step-{global_step}")
                        os.makedirs(save_path, exist_ok=True)
                        
                        # Generate sample images
                        # (Implementation similar to your original code)
                
                # Checkpointing
                if global_step % args.checkpointing_steps == 0:
                    save_checkpoint(model, value_net, args, accelerator, global_step)
            
            if global_step >= args.max_train_steps:
                break
        
        if global_step >= args.max_train_steps:
            break
    
    # Final checkpoint
    accelerator.wait_for_everyone()
    save_checkpoint(model, value_net, args, accelerator, global_step)
    
    if accelerator.is_main_process:
        model = accelerator.unwrap_model(model)
        model.save_pretrained(args.output_dir)
    
    accelerator.end_training()


if __name__ == "__main__":
    main()