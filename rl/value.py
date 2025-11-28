import torch.nn as nn

class SimpleValueNetwork(nn.Module):
    """Simple value network for PPO critic"""
    
    def __init__(self, latent_dim=256):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(latent_dim, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 1)
        )
    
    def forward(self, latent):
        # Pool spatial dimensions
        latent = latent.mean(dim=[2, 3])  # B, C, H, W -> B, C
        return self.net(latent).squeeze(-1)