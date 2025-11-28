class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        
def compute_combined_reward(pixel_values, fmap, reward_functions, reward_weights, lpips_model=None, **kwargs):
    """Compute weighted combination of multiple rewards"""
    total_reward = 0.0
    reward_dict = {}
    
    for reward_name, weight in zip(reward_functions, reward_weights):
        reward_fn = reward_registry.get(reward_name)
        reward = reward_fn(
            pixel_values=pixel_values,
            fmap=fmap,
            lpips_model=lpips_model,
            **kwargs
        )
        total_reward += weight * reward
        reward_dict[f"reward/{reward_name}"] = reward.item()
    
    reward_dict["reward/total"] = total_reward.item()
    return total_reward, reward_dict

def save_checkpoint(model, value_net, args, accelerator, global_step):
    """Save model checkpoint"""
    save_path = Path(args.output_dir) / f"checkpoint-{global_step}"
    
    state_dict = accelerator.get_state_dict(model)
    value_state_dict = accelerator.get_state_dict(value_net) if value_net is not None else None
    
    if accelerator.is_main_process:
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(
            save_path / "model",
            save_function=accelerator.save,
            state_dict=state_dict,
        )
        if value_state_dict is not None:
            torch.save(value_state_dict, save_path / "value_net.pt")
        json.dump({"global_step": global_step}, (save_path / "metadata.json").open("w+"))
        logger.info(f"Saved checkpoint to {save_path}")
    
    accelerator.save_state(save_path)