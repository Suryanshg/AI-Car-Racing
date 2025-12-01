import torch
import numpy as np
from environment_framestacking import CarRacingV3Wrapper
from stable_baselines3 import A2C
from torch.utils.data import Dataset, DataLoader
from stable_baselines3.common.policies import ActorCriticCnnPolicy
from torch.distributions import Normal
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
import argparse
import gc
import os


class ActorCriticCNN(nn.Module):
    def __init__(self, action_dim=3):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        # cnn output size
        n_flat = 32 * 2 * 7 * 7

        # shared MLP
        self.fc = nn.Sequential(nn.Linear(n_flat, 512), nn.ReLU())

        # actor
        self.mu = nn.Linear(512, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))

        # critic
        self.value = nn.Linear(512, 1)

    def forward(self, x):
        x = x / 255.0  # normalize input
        features = self.fc(self.cnn(x))

        mu = self.mu(features)
        # Constrain outputs
        steer = torch.tanh(mu[:, 0])  # Steering between -1 and 1
        gas = torch.sigmoid(mu[:, 1])  # Gas between 0 and 1
        brake = torch.sigmoid(mu[:, 2])  # Brake between 0 and 1
        mu = torch.stack([steer, gas, brake], dim=1)
        # std
        std = torch.exp(self.log_std)
        # value
        value = self.value(features)

        return mu, std, value


class PretrainBCDataset(Dataset):

    def __init__(self, data_file, indices=None):
        """
        Args:
            data_file: Path to .npz file
            indices: Optional indices to use (for train/val split)
        """
        # Load data with memory mapping
        self.data = np.load(data_file, mmap_mode="r")
        self.obs = self.data["obs"]
        self.actions = self.data["actions"]
        self.returns = self.data["returns"]

        if indices is not None:
            self.indices = indices
        else:
            self.indices = np.arange(len(self.obs))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        actual_idx = self.indices[idx]
        # Load single sample from disk
        obs = torch.FloatTensor(self.obs[actual_idx].copy())
        action = torch.FloatTensor(self.actions[actual_idx].copy())
        ret = torch.FloatTensor([self.returns[actual_idx].copy()])
        return obs, action, ret


def get_dataloaders(data_file, val_split, batch_size):
    # Get dataset size
    with np.load(data_file, mmap_mode="r") as data:
        n_samples = len(data["obs"])

    # Split into train and validation
    n_val = int(n_samples * val_split)
    n_train = n_samples - n_val
    print(f"Training samples: {n_train}, Validation samples: {n_val}")

    indices = np.random.permutation(n_samples)
    train_indices = indices[:n_train]
    val_indices = indices[n_train:]

    # Create memory-mapped datasets
    train_dataset = PretrainBCDataset(data_file, train_indices)
    val_dataset = PretrainBCDataset(data_file, val_indices)
    # Use smaller number of workers and prefetch
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        prefetch_factor=2,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        prefetch_factor=2,
    )
    return train_loader, val_loader


def train_bc(
    data_file,
    epochs=100,
    batch_size=64,
    learning_rate=1e-4,
    val_split=0.1,
    device="auto",
    save_path="bc_policy",
    weight_decay=1e-5,
    dropout=0.4,
    lambda_val=0.5,
):
    # check/create save path
    save_folder = os.path.dirname(save_path)
    if save_folder != "":
        os.makedirs(save_folder, exist_ok=True)
    # Handle device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Using device: {device}")
    train_loader, val_loader = get_dataloaders(data_file, val_split, batch_size)

    # Create custom CNN policy
    print(f"\nInitializing custom CNN policy...")
    env = CarRacingV3Wrapper()
    # Just a dummy policy to get the architecture and train loop right
    policy = ActorCriticCNN()
    policy.to(device)

    # Print model architecture
    print(f"\nModel Architecture:")
    print(policy)
    total_params = sum(p.numel() for p in policy.parameters())
    trainable_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    optimizer = optim.Adam(
        policy.parameters(), lr=learning_rate, weight_decay=weight_decay
    )

    # Learning rate scheduler
    # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20, eta_min=1e-6)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min")

    # Training history
    train_losses = []
    val_losses = []
    best_val_loss = float("inf")

    # Train
    print(f"\nTraining BC for {epochs} epochs...")
    print("=" * 60)

    for epoch in range(epochs):
        # Training
        policy.train()
        train_loss = 0.0
        for batch_obs, batch_actions, batch_returns in train_loader:
            batch_obs = batch_obs.to(device, non_blocking=True)
            batch_actions = batch_actions.to(device, non_blocking=True)
            batch_returns = batch_returns.to(device, non_blocking=True)
            optimizer.zero_grad()

            # Forward pass
            mu, std, value = policy(batch_obs)
            # # actor loss
            # dist = Normal(mu, std)
            # actor_loss = -dist.log_prob(batch_actions).sum(-1).mean()
            actor_loss = nn.functional.mse_loss(mu, batch_actions)

            # critic loss
            critic_loss = nn.functional.mse_loss(
                value.squeeze(), batch_returns.squeeze()
            )
            loss = actor_loss + lambda_val * critic_loss
            # print(actor_loss.item(), critic_loss.item())

            # Backward pass
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)

            optimizer.step()
            # just for logging
            train_loss += loss.item()

            # Clear cache periodically
            if device == "cuda":
                torch.cuda.empty_cache()

        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        # Validation
        policy.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_obs, batch_actions, batch_returns in val_loader:
                batch_obs = batch_obs.to(device, non_blocking=True)
                batch_actions = batch_actions.to(device, non_blocking=True)
                batch_returns = batch_returns.to(device, non_blocking=True)
                mu, std, value = policy(batch_obs)
                # actor loss
                dist = Normal(mu, std)
                actor_loss = -dist.log_prob(batch_actions).sum(-1).mean()
                # critic loss
                critic_loss = nn.functional.mse_loss(
                    value.squeeze(), batch_returns.squeeze()
                )
                loss = actor_loss + lambda_val * critic_loss
                val_loss += loss.item()

        # Clear cache after validation
        if device == "cuda":
            torch.cuda.empty_cache()

        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        # Update learning rate
        scheduler.step(val_loss)

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": policy.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "scheduler_state_dict": scheduler.state_dict(),
                },
                f"{save_path}_best.pth",
            )

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(
                f"Epoch [{epoch+1}/{epochs}] - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}"
            )

        # Force garbage collection every 10 epochs
        if (epoch + 1) % 10 == 0:
            gc.collect()

    print("=" * 60)
    print("Training complete!")
    print(f"Best validation loss: {best_val_loss:.6f}")

    # Save final model
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": policy.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_loss": val_loss,
            "scheduler_state_dict": scheduler.state_dict(),
        },
        f"{save_path}.pth",
    )
    print(f"Model saved to {save_path}.pth")
    print(f"Best model saved to {save_path}_best.pth")

    # Plot training curve
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("Behavior Cloning Training Curve")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{save_path}_training_curve.png")
    print(f"Training curve saved to {save_path}_training_curve.png")

    return policy


def evaluate_bc(
    model_path,
    num_episodes=10,
    render=False,
    device="auto",
    use_best=False,
):
    # Handle device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Create environment
    env = CarRacingV3Wrapper(
        env_name="CarRacing-v3",
        render_mode="human" if render else "rgb_array",
        lap_complete_percent=0.95,
        continuous=True,
        framestack=4,
    )

    # Load policy
    model_file = f"{model_path}_best.pth" if use_best else f"{model_path}.pth"
    print(f"Loading policy from {model_file}...")

    # Just a dummy policy to get the architecture and train loop right
    policy = ActorCriticCNN()
    checkpoint = torch.load(model_file, map_location=device)
    policy.load_state_dict(checkpoint["model_state_dict"])
    policy.to(device)
    policy.eval()

    print(f"Loaded model from epoch {checkpoint['epoch']}")

    episode_rewards = []
    all_actions = []

    print(f"\nEvaluating for {num_episodes} episodes...")
    print("=" * 60)

    for episode in range(num_episodes):
        obs, info = env.reset()
        done = False
        truncated = False
        episode_reward = 0
        steps = 0
        episode_actions = []

        while not done and not truncated:
            # Predict action using the policy
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)

            with torch.no_grad():
                mu, std, value = policy(obs_tensor)
                action = mu.cpu().numpy()[0]  # use mean action predictions
                action = np.clip(
                    action,
                    np.array([-1.0, 0.0, 0.0]),
                    np.array([1.0, 1.0, 1.0]),
                    dtype=np.float32,
                )

            episode_actions.append(action)

            # Step environment
            obs, reward, done, truncated, info = env.step(action)
            episode_reward += reward
            steps += 1

        episode_rewards.append(episode_reward)
        all_actions.extend(episode_actions)

        # Print episode stats
        episode_actions_np = np.array(episode_actions)
        print(f"Episode {episode+1}: Reward = {episode_reward:.2f}, Steps = {steps}")
        print(
            f"  Steering - Mean: {episode_actions_np[:, 0].mean():.3f}, Std: {episode_actions_np[:, 0].std():.3f}\n",
            f"  Gas - Mean: {episode_actions_np[:, 1].mean():.3f}, Std: {episode_actions_np[:, 1].std():.3f}\n",
            f"  Brake - Mean: {episode_actions_np[:, 2].mean():.3f}, Std: {episode_actions_np[:, 2].std():.3f}",
        )

    print("=" * 60)
    print(
        f"Average Reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}"
    )
    print(f"Max Reward: {np.max(episode_rewards):.2f}")
    print(f"Min Reward: {np.min(episode_rewards):.2f}")

    # Print overall action statistics
    all_actions_np = np.array(all_actions)
    print("\nOverall Action Statistics During Evaluation:")
    print(
        f"Steering - Mean: {all_actions_np[:, 0].mean():.3f}, Std: {all_actions_np[:, 0].std():.3f}"
    )
    print(
        f"Gas - Mean: {all_actions_np[:, 1].mean():.3f}, Std: {all_actions_np[:, 1].std():.3f}"
    )
    print(
        f"Brake - Mean: {all_actions_np[:, 2].mean():.3f}, Std: {all_actions_np[:, 2].std():.3f}"
    )

    env.close()
    return episode_rewards


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Behavior Cloning for CarRacing-v3")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["train", "eval", "both"],
        default="both",
        help="Mode: train, eval, or both",
    )
    parser.add_argument(
        "--data-file",
        type=str,
        default="./bc-logs/combined_bc_data.npz",
        help="Path to combined BC data file",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="a2c_bc_pretrain",
        help="Path to save/load model",
    )
    parser.add_argument(
        "--epochs", type=int, default=100, help="Number of training epochs"
    )
    parser.add_argument(
        "--batch-size", type=int, default=64, help="Batch size for training"
    )
    parser.add_argument(
        "--learning-rate", type=float, default=1e-5, help="Learning rate"
    )
    parser.add_argument(
        "--lambda-val", type=float, default=0.5, help="Weight for critic loss"
    )
    parser.add_argument(
        "--val-split", type=float, default=0.1, help="Validation split ratio"
    )
    parser.add_argument(
        "--eval-episodes", type=int, default=10, help="Number of evaluation episodes"
    )
    parser.add_argument(
        "--render", action="store_true", help="Enable rendering during evaluation"
    )

    parser.add_argument(
        "--device", type=str, default="auto", help="Device to use (auto/cuda/cpu)"
    )
    parser.add_argument(
        "--weight-decay", type=float, default=1e-5, help="L2 regularization"
    )
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
    parser.add_argument(
        "--use-best",
        action="store_true",
        help="Use best model checkpoint for evaluation",
    )

    args = parser.parse_args()

    if args.mode in ["train", "both"]:
        print("Starting Behavior Cloning Training...")
        policy = train_bc(
            data_file=args.data_file,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            val_split=args.val_split,
            device=args.device,
            save_path=args.model_path,
            weight_decay=args.weight_decay,
            dropout=args.dropout,
            lambda_val=args.lambda_val,
        )

    if args.mode in ["eval", "both"]:
        print("\nStarting Evaluation...")
        evaluate_bc(
            model_path=args.model_path,
            num_episodes=args.eval_episodes,
            render=args.render,
            device=args.device,
            use_best=args.use_best,
        )
