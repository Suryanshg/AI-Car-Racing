import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from environment_framestacking import CarRacingV3Wrapper
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import gc


class MemoryMappedBCDataset(Dataset):

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
        return obs, action


class CNNPolicy(nn.Module):
    """
    Custom CNN policy for behavior cloning.

    Architecture:
    - 3 Convolutional layers for feature extraction
    - 2 Fully connected layers
    - Constrained output layer (tanh for steering, sigmoid for gas/brake)
    """

    def __init__(self, input_channels=4, dropout=0.4, rnn=True):
        super(CNNPolicy, self).__init__()

        # CNN Feature Extractor
        # Input: (batch, 4, 84, 84) or (batch*4, 84, 84)
        n_kernels = 8 if rnn else 32
        input_channels = 1 if rnn else input_channels
        self.conv1 = nn.Conv2d(
            input_channels, n_kernels, kernel_size=8, stride=4
        )  # -> (32, 20, 20)
        self.conv2 = nn.Conv2d(
            n_kernels, n_kernels * 2, kernel_size=4, stride=2
        )  # -> (64, 9, 9)
        self.conv3 = nn.Conv2d(
            n_kernels * 2, n_kernels * 2, kernel_size=3, stride=1
        )  # -> (64, 7, 7)
        cnn_out_dim = n_kernels * 2 * 7 * 7
        # RNN
        if rnn:
            self.rnn = nn.LSTM(
                input_size=cnn_out_dim, hidden_size=512, batch_first=True
            )
        else:
            self.fc1 = nn.Linear(cnn_out_dim, 512)
        # Fully connected layers
        self.fc2 = nn.Linear(512, 256)
        self.out = nn.Linear(256, 3)

        # Batch normalization
        self.bn1 = nn.BatchNorm2d(n_kernels)
        self.bn2 = nn.BatchNorm2d(n_kernels * 2)
        self.bn3 = nn.BatchNorm2d(n_kernels * 2)

        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)

        # Activation
        self.relu = nn.ReLU()
        self.with_rnn = rnn

    def forward(self, x):
        x = x / 255.0  # Normalize input
        if self.with_rnn:
            batch_size, seq_len, H, W = x.size()
            x = x.view(batch_size * seq_len, H, W).unsqueeze(1)  # (B*S, 1, H, W)
        # Convolutional layers with batch norm and ReLU
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))

        # Flatten
        x = x.view(x.size(0), -1)

        if self.with_rnn:
            x = x.view(batch_size, seq_len, -1)
            x, _ = self.rnn(x)
            x = x[:, -1]  # Take last output
        else:
            x = self.relu(self.fc1(x))
            x = self.dropout(x)
        # Fully connected layers
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.out(x)
        # Constrain outputs
        steer = torch.tanh(x[:, 0])  # Steering between -1 and 1
        gas = torch.sigmoid(x[:, 1])  # Gas between 0 and 1
        brake = torch.sigmoid(x[:, 2])  # Brake between 0 and 1
        x = torch.stack([steer, gas, brake], dim=1)

        return x


def load_expert_data(data_file):
    print(f"Loading data from {data_file} (memory-mapped)...")
    data = np.load(data_file, mmap_mode="r")
    obs = data["obs"]
    actions = data["actions"]

    print(f"Dataset size: {len(obs)} samples")
    print(f"Observation shape: {obs.shape}")
    print(f"Actions shape: {actions.shape}")

    # Compute statistics without loading all data
    print("\nComputing action statistics (sampled)...")
    sample_size = min(10000, len(actions))
    sample_indices = np.random.choice(len(actions), sample_size, replace=False)
    actions_sample = actions[sample_indices]

    print(
        f"Steering - Mean: {actions_sample[:, 0].mean():.3f}, Std: {actions_sample[:, 0].std():.3f}"
    )
    print(
        f"Gas - Mean: {actions_sample[:, 1].mean():.3f}, Std: {actions_sample[:, 1].std():.3f}"
    )
    print(
        f"Brake - Mean: {actions_sample[:, 2].mean():.3f}, Std: {actions_sample[:, 2].std():.3f}"
    )

    return data_file  # Return file path instead of loaded data


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
    with_rnn=True,
):

    # Handle device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Using device: {device}")

    # Load data info (not the actual data)
    data_file_path = load_expert_data(data_file)

    # Get dataset size
    with np.load(data_file_path, mmap_mode="r") as data:
        n_samples = len(data["obs"])

    # Split into train and validation
    n_val = int(n_samples * val_split)
    n_train = n_samples - n_val

    indices = np.random.permutation(n_samples)
    train_indices = indices[:n_train]
    val_indices = indices[n_train:]

    # Create memory-mapped datasets
    train_dataset = MemoryMappedBCDataset(data_file_path, train_indices)
    val_dataset = MemoryMappedBCDataset(data_file_path, val_indices)

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

    print(f"Training samples: {n_train}, Validation samples: {n_val}")

    # Create custom CNN policy
    print(f"\nInitializing custom CNN policy...")
    policy = CNNPolicy(dropout=dropout, rnn=with_rnn)
    policy.to(device)

    # Print model architecture
    print(f"\nModel Architecture:")
    print(policy)
    total_params = sum(p.numel() for p in policy.parameters())
    trainable_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Use optimizer with weight decay
    optimizer = optim.Adam(
        policy.parameters(), lr=learning_rate, weight_decay=weight_decay
    )

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20, eta_min=1e-6)

    # Use MSE loss
    criterion = nn.MSELoss()

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
        for batch_obs, batch_actions in train_loader:
            batch_obs = batch_obs.to(device, non_blocking=True)
            batch_actions = batch_actions.to(device, non_blocking=True)

            optimizer.zero_grad()

            # Forward pass
            predicted_actions = policy(batch_obs)
            loss = criterion(predicted_actions, batch_actions)

            # Backward pass
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)

            optimizer.step()

            train_loss += loss.item()

            # Clear cache periodically
            if device == "cuda":
                torch.cuda.empty_cache()

        scheduler.step()

        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        # Validation
        policy.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_obs, batch_actions in val_loader:
                batch_obs = batch_obs.to(device, non_blocking=True)
                batch_actions = batch_actions.to(device, non_blocking=True)

                predicted_actions = policy(batch_obs)
                loss = criterion(predicted_actions, batch_actions)
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
            "epoch": epochs,
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
    with_rnn=True,
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

    policy = CNNPolicy(rnn=with_rnn)
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
                action = policy(obs_tensor).cpu().numpy()[0]
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
        "--model-path", type=str, default="bc_policy", help="Path to save/load model"
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
        "--val-split", type=float, default=0.1, help="Validation split ratio"
    )
    parser.add_argument(
        "--eval-episodes", type=int, default=10, help="Number of evaluation episodes"
    )
    parser.add_argument(
        "--render", action="store_true", help="Enable rendering during evaluation"
    )
    parser.add_argument(
        "--with-rnn", action="store_true", help="Enable RNN in the model"
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
            with_rnn=args.with_rnn,
        )

    if args.mode in ["eval", "both"]:
        print("\nStarting Evaluation...")
        evaluate_bc(
            model_path=args.model_path,
            num_episodes=args.eval_episodes,
            render=args.render,
            device=args.device,
            use_best=args.use_best,
            with_rnn=args.with_rnn,
        )
