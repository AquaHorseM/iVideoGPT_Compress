from rl.reward_bp import compute_combined_reward
import torch
import torch.nn.functional as F

class RolloutBuffer:
    """Buffer for storing rollout data for PPO"""
    
    def __init__(self):
        self.states = []
        self.actions = []  # Quantized indices
        self.rewards = []
        self.log_probs = []
        self.values = []
        self.advantages = []
        self.returns = []
    
    def clear(self):
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.log_probs.clear()
        self.values.clear()
        self.advantages.clear()
        self.returns.clear()
    
    def add(self, state, action, reward, log_prob, value):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.log_probs.append(log_prob)
        self.values.append(value)
    
    def compute_advantages(self, gamma=0.99, gae_lambda=0.95, last_value=0):
        """Compute GAE advantages"""
        advantages = []
        gae = 0
        
        # Add last_value for bootstrap
        values = self.values + [last_value]
        
        for t in reversed(range(len(self.rewards))):
            delta = self.rewards[t] + gamma * values[t + 1] - values[t]
            gae = delta + gamma * gae_lambda * gae
            advantages.insert(0, gae)
        
        self.advantages = advantages
        self.returns = [adv + val for adv, val in zip(advantages, self.values)]

def train_with_ppo(model, value_net, batch, rollout_buffer, args, lpips_model, accelerator, global_step):
    """
    PPO training with advantage estimation
    More stable but more complex
    """
    pixel_values = batch.to(accelerator.device, non_blocking=True)
    pixel_values = pixel_values.reshape(-1, *pixel_values.shape[-3:])
    
    # Prepare frames
    BT, C, H, W = pixel_values.shape
    B, T = (BT // args.segment_length), args.segment_length
    frame_pixel_values = pixel_values.reshape(B, T, C, H, W)
    target = frame_pixel_values[:, args.context_length:].reshape(B * (T - args.context_length), C, H, W)
    reference_single = frame_pixel_values[:, args.context_length - 1]
    
    # Forward pass (with gradient tracking for policy)
    fmap, fmap_ref, commit_loss, dyna_commit_loss = model(
        sample=reference_single,
        dyn_sample=target,
        return_dict=False,
        return_loss=True,
        segment_len=args.segment_length - args.context_length
    )
    
    # Get latent representation for value network
    # We need to access the latent codes - this depends on your model structure
    # For now, assume we can get it from the encoder output
    with torch.no_grad():
        latent = model.encode(reference_single).latents if hasattr(model, 'encode') else fmap.detach()
    
    # Value prediction
    value = value_net(latent)
    
    # Compute reward
    reward, reward_dict = compute_combined_reward(
        pixel_values=target,
        fmap=fmap,
        reward_functions=args.reward_functions,
        reward_weights=args.reward_weights,
        lpips_model=lpips_model
    )
    
    # For PPO, we need log probabilities of actions
    # In VQVAE, "actions" are the discrete codebook selections
    # This requires accessing the quantization layer's distribution
    # Simplified: use reconstruction likelihood as pseudo log_prob
    log_prob = -F.mse_loss(target, fmap, reduction='none').mean(dim=[1, 2, 3])
    
    # Store in rollout buffer
    rollout_buffer.add(
        state=latent,
        action=None,  # Discrete codes - depends on model implementation
        reward=reward,
        log_prob=log_prob.mean(),
        value=value.mean()
    )
    
    # Compute advantages every N steps
    if len(rollout_buffer.rewards) >= args.train_batch_size:
        rollout_buffer.compute_advantages(gamma=args.gamma, gae_lambda=args.gae_lambda)
        
        # PPO update
        policy_loss, value_loss, entropy = compute_ppo_loss(
            model, value_net, rollout_buffer, args, target, reference_single
        )
        
        # Total loss
        loss = policy_loss + args.value_loss_coef * value_loss - args.entropy_coef * entropy
        loss += args.commit_loss_weight * commit_loss + args.dyna_commit_loss_weight * dyna_commit_loss
        
        rollout_buffer.clear()
        
        return loss, reward_dict, {
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "entropy": entropy.item(),
            "commit_loss": commit_loss.item(),
            "dyna_commit_loss": dyna_commit_loss.item()
        }
    else:
        # Not enough samples yet
        return None, reward_dict, {}


def compute_ppo_loss(model, value_net, rollout_buffer, args, target, reference_single):
    """Compute PPO loss with clipped objective"""
    
    # Convert buffer to tensors
    old_log_probs = torch.stack([lp.detach() for lp in rollout_buffer.log_probs])
    advantages = torch.tensor(rollout_buffer.advantages, device=old_log_probs.device)
    returns = torch.tensor(rollout_buffer.returns, device=old_log_probs.device)
    
    # Normalize advantages
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    
    # Recompute values and log probs
    fmap, _, _, _ = model(
        sample=reference_single,
        dyn_sample=target,
        return_dict=False,
        return_loss=True,
        segment_len=args.segment_length - args.context_length
    )
    
    new_log_probs = -F.mse_loss(target, fmap, reduction='none').mean(dim=[1, 2, 3])
    latent = fmap.detach()
    new_values = value_net(latent)
    
    # Compute ratio and clipped objective
    ratio = torch.exp(new_log_probs - old_log_probs)
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1.0 - args.clip_epsilon, 1.0 + args.clip_epsilon) * advantages
    policy_loss = -torch.min(surr1, surr2).mean()
    
    # Value loss
    value_loss = F.mse_loss(new_values, returns)
    
    # Entropy bonus (encourage exploration)
    entropy = -(new_log_probs * torch.exp(new_log_probs)).mean()
    
    return policy_loss, value_loss, entropy