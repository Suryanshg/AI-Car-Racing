# from car_racing_env import CarRacingV3Wrapper
from car_racing_env_v2 import CarRacingV3Wrapper
import torch
import numpy as np
from ppo_network import PPO_Network
from collections import deque
from torch.distributions import Beta
from collections import namedtuple
import random
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter


# Define the structure of a single transition
Transition = namedtuple('Transition', 
    ['state', 'action', 'log_prob', 'reward', 'next_state', 'done', 'value']
)

class Agent_PPO():
    def __init__(self, env: CarRacingV3Wrapper, args):
        """
        TODO:
        """
        # Store the passed env as member variables
        self.env: CarRacingV3Wrapper = env

        # Determine device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Init Deep Learning Hyperparams
        self.lr = args.lr
        self.batch_size = args.batch_size
        self.max_grad_norm = args.max_grad_norm

        # Init PPO Hyperparams
        self.gamma = args.gamma
        self.gae_lambda = args.gae_lambda
        self.ppo_epochs = args.ppo_epochs
        self.clip_epsilon = args.clip_epsilon
        self.value_loss_coef = args.value_loss_coef
        self.entropy_coef = args.entropy_coef
        self.training_iterations = args.training_iterations
        self.buffer_capacity = args.buffer_capacity
        self.num_episodes_to_collect = args.num_episodes_to_collect
        self.max_episode_steps = args.max_episode_steps

        # Init Env Wrapper Args
        self.action_repetition = args.action_repetition
        self.frame_stack_size = args.frame_stack_size

        # Init Misc Args
        self.seed = args.seed
        self.save_freq = args.save_freq
        self.log_freq = args.log_freq

        # Set Random Seeds
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)

        # Init Deep Learning Model
        self.ppo_network = PPO_Network().to(self.device)

        # Init Loss Function, Optimizer
        self.loss_fn = torch.nn.SmoothL1Loss()
        self.optim = torch.optim.Adam(self.ppo_network.parameters(), lr = self.lr)

        # Init Replay Buffer
        self.buffer = []

        # Init Stats Tracking variables
        self.episode_rewards = deque(maxlen=self.num_episodes_to_collect)

        # Init SummaryWriter for TensorBoard
        self.tb_writer = SummaryWriter(log_dir="runs/ppo")

        # If we init AgentPPO in test (inference) mode
        # TODO: parameterize the path of the model here
        if args.test_ppo:
            print('loading trained model: ')
            self.load_model('checkpoints/ppo_model_final_config2.pth')
            self.ppo_network.eval()
            print('Loaded trained PPO Network successfully!')


    def train(self):
        """
        Main training loop for PPO.
        
        Training Process:
        1. Collect trajectories using current policy
        2. Compute advantages from collected data
        3. Update policy using PPO objective
        4. Repeat for many iterations
        """
        print("Starting PPO Training...")
        print(f"Device: {self.device}")
        print(f"Training for {self.training_iterations} iterations")
        
        # For each training iteration
        for iteration in tqdm(range(self.training_iterations)):
            print(f"\n{'='*60}")
            print(f"Iteration {iteration + 1}/{self.training_iterations}")
            print(f"{'='*60}")
            
            # Step 1: Collect trajectories
            print("Collecting episodes...")
            self.collect_episodes()
            
            # Step 2 & 3: Compute advantages and update policy
            print("Updating policy...")
            self.update_policy()
            
            # Log progress
            if (iteration + 1) % self.log_freq == 0:
                avg_reward = np.mean(self.episode_rewards)
                self.tb_writer.add_scalar('train/avg_reward', avg_reward, iteration + 1)
                print(f"\nAverage Reward (last 100 episodes): {avg_reward:.2f}")
            
            # Save model checkpoint
            if (iteration + 1) % self.save_freq == 0:
                self.save_model(f"checkpoints/ppo_model_iter_{iteration + 1}.pth")
        
        print("\n" + "="*60)
        print("Training completed!")
        print("="*60)
        self.save_model("checkpoints/ppo_model_final.pth")
        


    def reset_buffer(self):
        """
        Reset the replay buffer.
        """
        # Set the buffer to empty list
        self.buffer = []


    def select_action(self, state: np.ndarray):
        """
        Select an action using the current policy.
        
        How it works:
        1. Pass state through network to get alpha, beta (for Beta distribution) and value
        2. Create Beta distribution using alpha and beta
        3. Sample action from the distribution
        4. Compute log probability of that action
        5. Scale action to appropriate ranges: steering [-1, 1], gas/brake [0, 1]
        
        Args:
            state: Current state (numpy array). Shape: (4, 96, 96)
            
        Returns:
            action: Selected action (numpy array). Shape: (3,)
            log_prob: Log probability of the action (float)
            value: Value estimate of the state (float)
        """
        # TODO: Put Network in Eval Mode

        # Convert state to float32 tensor
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device).div_(255.0)    # (1, 4, 96, 96)
        
        # Freeze the model weights
        with torch.no_grad():

            # Predict the policy dist params and value using PPO Network
            (alpha, beta), value = self.ppo_network(state_tensor)               # alpha: (1, 3), beta: (1, 3), value: (1, 1)
        
        # Create Beta distribution for each action dimension
        # Beta distribution outputs values in [0, 1]
        beta_dist = Beta(alpha, beta)
        
        # Sample actions from Beta distributions
        action = beta_dist.sample()       # (1, 3)
        
        # Compute log probability of the sampled action
        log_prob = beta_dist.log_prob(action).sum(dim = -1) # (1,)

        # TODO: Check if action scaling is actually needed or not
        # Scale actions to appropriate ranges:
        # - Steering: [0, 1] -> [-1, 1]
        # - Gas: [0, 1] (already correct)
        # - Brake: [0, 1] (already correct)
        action_scaled = action.clone()
        action_scaled[0, 0] = action_scaled[0, 0] * 2 - 1  # Steering to [-1, 1]
        
        # Convert to numpy and remove batch dimension
        action_np = action_scaled.squeeze(0).cpu().numpy()  # (3,)
        
        # Return the sampled action vector, log prob of action, and value predicted by network
        return action_np, log_prob.item(), value.item()
    

    # TODO: Check if this implementation for collecting episodes is good or not
    # TODO: Look into parallelizing this 
    def collect_episodes(self):
        """
        Collect trajectories by running current policy in the environment.
        
        It works the following way:
        1. Run the agent in the environment using current policy
        2. Store states, actions, rewards, etc. in the buffer
        3. This data will be used to update the policy in the main training method
        """
        # Reset the buffer
        self.reset_buffer()
        
        # Collect episodes
        for episode in range(self.num_episodes_to_collect):

            # Begin an episode and get its first state
            state, _ = self.env.reset()

            # Keep track of rewards in this episode
            episode_reward = 0
            
            # Keep collecting data for this episode for max_episode_steps
            for step in range(self.max_episode_steps):
                # Select an action, its log_prob and the value of the current state
                action, log_prob, value = self.select_action(state)

                # Take the selected action in env
                next_state, reward, done, truncated, _ = self.env.step(action)
                
                # Create the transition tuple
                transition = Transition(
                    state=state, 
                    action=action, 
                    log_prob=log_prob, 
                    reward=reward, 
                    next_state=next_state, 
                    done=done,
                    value=value
                )
                
                # Push to the buffer
                self.buffer.append(transition)
                
                # Accumulate Rewards for this episode
                episode_reward += reward

                # Move to the next state
                state = next_state

                # # Stop if buffer is full
                # if len(self.buffer) >= self.buffer_capacity:
                #     break
                
                # Stop collecting data if episode is done or truncated
                if done or truncated:
                    break

            # Accumulate reward for each episode in the global reward queue (for stats)        
            self.episode_rewards.append(episode_reward)

            # Log the episode's reward
            episode += 1
            

            # TODO: For some reason, the buffer size always grows in multiples of 119 per each iteration (5 iterations)
            print(f"Episode {episode} - Reward: {episode_reward:.2f}, Buffer Size: {len(self.buffer)}/{self.buffer_capacity}, Done: {done}, Truncated: {truncated}")
            
            # # Stop if buffer is full
            # if len(self.buffer) >= self.buffer_capacity:
            #     break


    def compute_gae(self):
        """
        TODO:

        Returns:
            _type_: _description_
        """
        # Extract Transition Elements into their each numpy arrays
        rewards = np.array([t.reward for t in self.buffer])
        values = np.array([t.value for t in self.buffer])
        dones = np.array([t.done for t in self.buffer])
        next_states = np.array([t.next_state for t in self.buffer])

        # Create a tensor for next_states
        next_states_tensor = torch.FloatTensor(next_states).to(self.device).div_(255.0)
        
        # Predict values for the whole set of next_states using the PPO Network in one-fell-swoop
        with torch.no_grad():
            _, next_state_values = self.ppo_network(next_states_tensor) # (T, 1) where T = num_episodes * max_episode_steps

            # Convert the next_state_values into numpy arrays
            next_state_values = next_state_values.cpu().numpy().flatten() # (T,)
            
        # Init Variables to compute GAE
        advantages = []
        gae = 0
        
        # Loop backwards
        for t in reversed(range(len(self.buffer))):
            
            # Calculate the multiplier
            # If done=1 (died), non_terminal=0. Connection to future cut.
            # If done=0 (alive), non_terminal=1. Connection kept.

            # Figure out if state is terminal or not
            non_terminal = 1.0 - dones[t]
            
            # Compute TD Error: r + gamma * V(next_s) * (1-d) - V(s)
            delta = rewards[t] + self.gamma * next_state_values[t] * non_terminal - values[t]
            
            # Compute GAE: delta + gamma * lambda * (1-d) * gae
            gae = delta + self.gamma * self.gae_lambda * non_terminal * gae
            
            # Accumulate Advantage values
            # NOTE: They will be stored in reverse order here
            advantages.append(gae)
            
        # Reverse stored advantage values and convert to numpy array
        advantages = advantages[::-1] # Reverse
        advantages = np.array(advantages)
        
        # Compute Return using advantages and values at each timestep
        returns = advantages + values
        
        # Normalize Advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Return computed advantages and returns
        return advantages, returns
    

    def update_policy(self):
        """
        Update policy using PPO algorithm.
        
        Steps:
        1. Compute advantages using GAE
        2. For multiple epochs:
           a. Shuffle data and create mini-batches
           b. Compute new policy predictions
           c. Calculate PPO clipped loss
           d. Calculate value loss (using SmoothL1Loss)
           e. Add entropy bonus (encourages exploration)
           f. Update network
        """
        # TODO: Set Network to training mode

        # Compute advantages and returns from collected data
        advantages, returns = self.compute_gae()

        # Extract Data from Buffer for PPO Update as numpy arrays
        states_np = np.array([t.state for t in self.buffer])
        actions_np = np.array([t.action for t in self.buffer])
        log_probs_np = np.array([t.log_prob for t in self.buffer])

        # Convert all data to tensors
        states_tensor = torch.FloatTensor(states_np).to(self.device).div_(255.0)
        actions_tensor = torch.FloatTensor(actions_np).to(self.device)
        old_log_probs_tensor = torch.FloatTensor(log_probs_np).to(self.device)
        returns_tensor = torch.FloatTensor(returns).to(self.device)
        advantages_tensor = torch.FloatTensor(advantages).to(self.device)

        # Get Dataset (Buffer) Length
        # TODO: Later replace this line with buffer capacity
        dataset_size = len(states_np)

        # Update policy for multiple epochs
        for epoch in range(self.ppo_epochs):

            # Generate random indices for mini-batching
            indices = np.arange(dataset_size)
            np.random.shuffle(indices)

            # Iterate over mini-batches
            for start in range(0, dataset_size, self.batch_size):
                
                # Compute the indices for this mini-batch
                end = start + self.batch_size
                mini_batch_idx = indices[start: end]

                # Extract the mini-batch
                batch_states = states_tensor[mini_batch_idx]
                batch_actions = actions_tensor[mini_batch_idx]
                batch_old_log_probs = old_log_probs_tensor[mini_batch_idx]
                batch_returns = returns_tensor[mini_batch_idx]
                batch_advantages = advantages_tensor[mini_batch_idx]

                # Forward pass thru PPO Network to get new policy and values for this batch
                (alpha, beta), batch_values = self.ppo_network(batch_states)

                # Create Beta Distribution
                beta_dist = Beta(alpha, beta)

                # TODO: Check if scaling is actually needed or not
                # Unscale steering actions back to [0, 1] for log_prob calculations
                # This is necessary as log of negative values is not a real number
                current_batch_actions = batch_actions.clone()
                current_batch_actions[:, 0] = (current_batch_actions[:, 0] + 1) / 2

                # TODO: Remove this if we are not doing any scaling at all
                # Clamp actions to avoid log(0) = -inf
                # Beta distribution is defined on (0, 1). 
                # 0.0 and 1.0 result in -inf log_prob if alpha/beta > 1
                current_batch_actions = torch.clamp(current_batch_actions, 1e-6, 1.0 - 1e-6)

                # Compute new_log_probs and entropy
                batch_new_log_probs = beta_dist.log_prob(current_batch_actions).sum(dim = -1)
                entropy = beta_dist.entropy().sum(dim = -1)

                # TODO: Check if this is needed or not
                # Squeeze values for returning later
                batch_values = batch_values.squeeze() # (B,)

                # Compute Ratio: π_new(a|s) / π_old(a|s)
                # However, computing ratio as exp(log_new - log_old) is numerically more stable
                ratios = torch.exp(batch_new_log_probs - batch_old_log_probs)

                # Compute Surrogate Loss
                surr1 = ratios * batch_advantages
                surr2 = torch.clamp(ratios, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * batch_advantages

                # Compute Policy Loss (Actor Loss)
                # We take negative because we want to Maximize objective (& Minimize Loss)
                policy_loss = -torch.min(surr1, surr2).mean()

                # Compute Value Loss (Critic Loss)
                value_loss = self.loss_fn(batch_values, batch_returns)

                # Compute Entropy Loss
                entropy_loss = -torch.mean(entropy)

                # Compute Total Loss
                # Minimize Policy Loss + Value Loss - Entropy (to encourage exploration)
                loss = policy_loss + (self.value_loss_coef * value_loss) + (self.entropy_coef * entropy_loss)

                # Perform Backpropagation
                self.optim.zero_grad()
                loss.backward()

                # Perform Gradient Clipping (prevents exploding gradients)
                torch.nn.utils.clip_grad_norm_(self.ppo_network.parameters(), self.max_grad_norm)

                # Update the weights
                self.optim.step()


            # Logging Stats per epoch
            print(f"  Epoch {epoch + 1}/{self.ppo_epochs} - Actor Loss: {policy_loss.item():.4f}, "
                    f"Value Loss: {value_loss.item():.4f}, Entropy: {entropy.mean().item():.4f}")

        

    def save_model(self, filename):
        """
        Save model checkpoint.
        """
        torch.save({
            'model_state_dict': self.ppo_network.state_dict(),
            'optimizer_state_dict': self.optim.state_dict(),
            'episode_rewards': list(self.episode_rewards),
        }, filename)
        print(f"Model saved to {filename}")



    def load_model(self, filename):
        """
        Load model checkpoint.
        """
        checkpoint = torch.load(filename, map_location=self.device)
        self.ppo_network.load_state_dict(checkpoint['model_state_dict'])
        self.optim.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"Model loaded from {filename}")



    def test_drive(self):
        """
        TODO:
        """

        # Reset the env and get the first state
        state, _ = self.env.reset()
        self.env.save_state_img(
            state,
            "Image after resetting env",
            "img_reset.png",
            cmap="gray"
        )

        total_reward = 0.0

        # Do Full Gas Action for 10 times
        for i in range(11):
            next_state, reward, done, truncated, _ = self.env.step(np.array([0.0, 2.0, 0.0]))
            total_reward += reward
            print(f"Step {i+1} - Reward: {reward}, Total Reward: {total_reward}")

        self.env.save_state_img(next_state[-1], 
                                "Image after accelerating for 10 times", 
                                "img_accelerate_10.png", 
                                cmap = "gray")