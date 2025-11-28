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

# ============================================================================
# Reward Functions Registry
# ============================================================================

class RewardRegistry:
    """Registry for different reward functions"""
    
    def __init__(self):
        self.rewards = {}
    
    def register(self, name: str):
        """Decorator to register a reward function"""
        def decorator(func: Callable):
            self.rewards[name] = func
            return func
        return decorator
    
    def get(self, name: str) -> Callable:
        """Get a reward function by name"""
        if name not in self.rewards:
            raise ValueError(f"Unknown reward: {name}. Available: {list(self.rewards.keys())}")
        return self.rewards[name]
    
    def list_rewards(self):
        """List all available rewards"""
        return list(self.rewards.keys())


reward_registry = RewardRegistry()


# ============================================================================
# Reward Function Implementations
# ============================================================================

@reward_registry.register("perceptual")
def perceptual_reward(pixel_values, fmap, lpips_model, **kwargs):
    """Negative LPIPS loss as reward (higher is better)"""
    with torch.no_grad():
        perceptual_loss = lpips_model(
            pixel_values.contiguous() * 2 - 1.0,
            fmap.contiguous() * 2 - 1.0
        ).mean()
    # Return negative (we want to minimize LPIPS, so maximize negative LPIPS)
    return -perceptual_loss


@reward_registry.register("reconstruction")
def reconstruction_reward(pixel_values, fmap, loss_type="l1", **kwargs):
    """Negative reconstruction loss as reward"""
    with torch.no_grad():
        if loss_type == "l2":
            recon_loss = F.mse_loss(pixel_values, fmap)
        else:
            recon_loss = F.l1_loss(pixel_values, fmap)
    return -recon_loss


@reward_registry.register("sharpness")
def sharpness_reward(fmap, **kwargs):
    """Reward for sharp images using Laplacian variance"""
    with torch.no_grad():
        # Convert to grayscale
        gray = 0.299 * fmap[:, 0] + 0.587 * fmap[:, 1] + 0.114 * fmap[:, 2]
        # Laplacian kernel
        laplacian_kernel = torch.tensor([
            [0, 1, 0],
            [1, -4, 1],
            [0, 1, 0]
        ], dtype=gray.dtype, device=gray.device).view(1, 1, 3, 3)
        
        # Compute Laplacian
        laplacian = F.conv2d(gray.unsqueeze(1), laplacian_kernel, padding=1)
        variance = laplacian.var()
    return variance


@reward_registry.register("temporal_consistency")
def temporal_consistency_reward(fmap, **kwargs):
    """Reward for temporally consistent predictions"""
    with torch.no_grad():
        if fmap.shape[0] < 2:
            return torch.tensor(0.0, device=fmap.device)
        
        # Compute frame-to-frame differences
        diffs = []
        for i in range(1, fmap.shape[0]):
            diff = F.mse_loss(fmap[i], fmap[i-1])
            diffs.append(diff)
        
        # Lower difference = higher consistency = higher reward
        avg_diff = torch.stack(diffs).mean()
        return -avg_diff


@reward_registry.register("aesthetic")
def aesthetic_reward(fmap, aesthetic_model=None, **kwargs):
    """Aesthetic quality reward (requires an aesthetic predictor model)"""
    if aesthetic_model is None:
        raise ValueError("aesthetic_model must be provided for aesthetic reward")
    
    with torch.no_grad():
        # Assume aesthetic_model takes images and returns scores
        scores = aesthetic_model(fmap)
    return scores.mean()


@reward_registry.register("custom")
def custom_reward(custom_reward_fn=None, **kwargs):
    """Wrapper for custom user-defined reward functions"""
    if custom_reward_fn is None:
        raise ValueError("custom_reward_fn must be provided for custom reward")
    return custom_reward_fn(**kwargs)