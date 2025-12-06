import numpy as np
import os
from pathlib import Path
import argparse
import matplotlib.pyplot as plt


def is_noop(action, steering_threshold=0.05, gas_threshold=0.05, brake_threshold=0.05):
    return (
        abs(action[0]) < steering_threshold
        and action[1] < gas_threshold
        and action[2] < brake_threshold
    )


def classify_action(action):
    # Check for no-op first
    if is_noop(action):
        return "noop"
    else:
        return "action"


def compute_discounted_returns(rewards, returns_gamma=0.99):
    returns = np.zeros_like(rewards, dtype=np.float32)
    running_return = 0.0

    # Compute backwards
    for t in reversed(range(len(rewards))):
        running_return = rewards[t] + returns_gamma * running_return
        returns[t] = running_return

    return returns


def combine_bc_data(
    bc_logs_dir="bc-logs",
    output_file="bc-logs/combined_bc_data.npz",
    noop_keep_ratio=0.4,
    returns_gamma=0.99,
):
    # Find all .npz files recursively
    bc_logs_path = Path(bc_logs_dir)
    npz_files = list(bc_logs_path.rglob("*.npz"))

    if not npz_files:
        print(f"No .npz files found in {bc_logs_dir}")
        return

    print(f"Found {len(npz_files)} .npz files")

    all_obs = []
    all_actions = []
    all_dones = []
    all_returns = []

    # Load and combine all files
    for npz_file in npz_files:
        print(f"Loading {npz_file}...")
        data = np.load(npz_file)

        # Check if the file contains the expected keys
        if (
            "obs" in data
            and "actions" in data
            and "rewards" in data
            and "dones" in data
        ):
            all_obs.append(data["obs"])
            all_actions.append(data["actions"])
            all_dones.append(data["dones"])
            all_returns.append(
                compute_discounted_returns(data["rewards"], returns_gamma)
            )
        else:
            print(f"  Skipping {npz_file}, missing required keys.")

    # Concatenate all data
    combined_obs = np.concatenate(all_obs, axis=0)
    combined_actions = np.concatenate(all_actions, axis=0)
    combined_dones = np.concatenate(all_dones, axis=0)
    combined_returns = np.concatenate(all_returns, axis=0)
    print(f"Combined observations shape: {combined_obs.shape}")
    print(f"Combined actions shape: {combined_actions.shape}")
    print(f"Combined dones shape: {combined_dones.shape}")
    print(f"Combined returns shape: {combined_returns.shape}")
    # Classify actions
    print("\nClassifying actions...")
    action_categories = {}
    for i, action in enumerate(combined_actions):
        category = classify_action(action)
        if category not in action_categories:
            action_categories[category] = []
        action_categories[category].append(i)

    # Print action distribution
    print("\nAction Distribution:")
    print("-" * 50)
    for category, indices in sorted(action_categories.items()):
        print(
            f"{category:10s}: {len(indices):6d} ({len(indices)/len(combined_actions)*100:.2f}%)"
        )
    print("-" * 50)

    # Balance or filter actions
    selected_indices = []

    # Keep all actions but reduce no-ops
    print(f"\nReducing no-ops to {noop_keep_ratio*100:.1f}%...")
    for category, indices in action_categories.items():
        if category == "noop":
            n_samples = int(len(indices) * noop_keep_ratio)
            sampled_indices = np.random.choice(indices, size=n_samples, replace=False)
            selected_indices.extend(sampled_indices)
            print(f"  no-ops: sampled {n_samples} / {len(indices)}")
        else:
            # Keep all action samples
            selected_indices.extend(indices)
            print(f"  {category}: kept all {len(indices)} samples")

    # Convert to numpy array and shuffle
    selected_indices = np.array(selected_indices)
    np.random.shuffle(selected_indices)

    # Select the balanced/filtered data
    final_obs = combined_obs[selected_indices]
    final_actions = combined_actions[selected_indices]
    final_dones = combined_dones[selected_indices]
    final_returns = combined_returns[selected_indices]

    print(f"\nFinal dataset:")
    print(f"  Observations: {final_obs.shape}")
    print(f"  Actions: {final_actions.shape}")
    print(
        f"  Reduction: {len(combined_obs)} -> {len(final_obs)} ({len(final_obs)/len(combined_obs)*100:.1f}%)"
    )

    # Horizontal flip the images and invert steering
    print("\nApplying data augmentation (horizontal flip)...")
    final_obs = np.append(
        final_obs, np.flip(final_obs, axis=3), axis=0
    )  # horizontal flip
    flipped_actions = np.copy(final_actions)
    flipped_actions[:, 0] = -1 * flipped_actions[:, 0]  # invert steering
    final_actions = np.append(final_actions, flipped_actions, axis=0)
    final_dones = np.append(final_dones, final_dones, axis=0)
    final_returns = np.append(final_returns, final_returns, axis=0)

    # Print final action distribution
    print("\nFinal Action Distribution:")
    print("-" * 50)
    final_categories = {}
    for action in final_actions:
        category = classify_action(action)
        final_categories[category] = final_categories.get(category, 0) + 1

    for category, count in sorted(final_categories.items()):
        print(f"{category:10s}: {count:6d} ({count/len(final_actions)*100:.2f}%)")
    print("-" * 50)

    # Save
    print(f"\nSaving to {output_file}...")
    np.savez(
        output_file,
        obs=final_obs,
        actions=final_actions,
        dones=final_dones,
        returns=final_returns,
    )

    print("Done!")
    print(f"Total samples: {final_obs.shape[0]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Combine BC data with no-op reduction")
    parser.add_argument(
        "--bc-logs-dir",
        type=str,
        default="./bc-logs",
        help="Directory containing .npz files",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default="./bc-logs/combined_bc_data.npz",
        help="Output file path",
    )

    parser.add_argument(
        "--noop-keep-ratio",
        type=float,
        default=0.3,
        help="Fraction of no-ops to keep (default: 0.3)",
    )

    args = parser.parse_args()

    combine_bc_data(
        bc_logs_dir=args.bc_logs_dir,
        output_file=args.output_file,
        noop_keep_ratio=args.noop_keep_ratio,
    )
