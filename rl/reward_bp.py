from rl.utils import compute_combined_reward

def train_with_reward_backprop(model, batch, args, lpips_model, accelerator):
    """
    Direct reward backpropagation (like VADER)
    Simplest approach: backprop reward directly through the model
    """
    pixel_values = batch.to(accelerator.device, non_blocking=True)
    pixel_values = pixel_values.reshape(-1, *pixel_values.shape[-3:])
    
    # Prepare frames
    BT, C, H, W = pixel_values.shape
    B, T = (BT // args.segment_length), args.segment_length
    frame_pixel_values = pixel_values.reshape(B, T, C, H, W)
    target = frame_pixel_values[:, args.context_length:].reshape(B * (T - args.context_length), C, H, W)
    reference_single = frame_pixel_values[:, args.context_length - 1]
    
    # Forward pass
    fmap, fmap_ref, commit_loss, dyna_commit_loss = model(
        sample=reference_single,
        dyn_sample=target,
        return_dict=False,
        return_loss=True,
        segment_len=args.segment_length - args.context_length
    )
    
    # Compute reward
    reward, reward_dict = compute_combined_reward(
        pixel_values=target,
        fmap=fmap,
        reward_functions=args.reward_functions,
        reward_weights=args.reward_weights,
        lpips_model=lpips_model
    )
    
    # Total loss = -reward + commitment losses
    loss = -reward + args.commit_loss_weight * commit_loss + args.dyna_commit_loss_weight * dyna_commit_loss
    
    return loss, reward_dict, {"commit_loss": commit_loss.item(), "dyna_commit_loss": dyna_commit_loss.item()}