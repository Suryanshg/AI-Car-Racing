import argparse

def get_args(parser):
    '''
    TODO:
    '''
    # Deep Learning Hyperparameters (TODO: Double check if all are used here)
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size')
    parser.add_argument('--grad_clip_value', type=float, default=0.5, help='max grad value')
    parser.add_argument('--train_freq', type=int, default=4, help='Train every N steps')

    # PPO Hyperparameters (TODO: Double check if all are used here)
    parser.add_argument('--gamma', type=float, default=0.99, help='discount factor')
    parser.add_argument('--gae_lambda', type=float, default=0.95, help='GAE lambda parameter')
    parser.add_argument('--ppo_epochs', type=int, default=10, help='number of PPO epochs')
    parser.add_argument('--clip_epsilon', type=float, default=0.2, help='PPO clip epsilon')
    parser.add_argument('--value_loss_coef', type=float, default=0.5, help='value loss coefficient')
    parser.add_argument('--entropy_coef', type=float, default=0.01, help='entropy coefficient')
    parser.add_argument('--max_grad_norm', type=float, default=0.5, help='max gradient norm')
    parser.add_argument('--training_iterations', type=int, default=1000, help='number of training iterations')
    parser.add_argument('--buffer_capacity', type=int, default=2048, help='replay buffer capacity')
    parser.add_argument('--num_episodes', type=int, default=5, help='number of episodes per iteration')
    parser.add_argument('--max_episode_steps', type=int, default=1000, help='max steps per episode')

    # Environment Wrapper Specific Args
    parser.add_argument('--action_repetition', type=int, default=8, help='number of times to repeat the action')
    parser.add_argument('--frame_stack_size', type=int, default=4, help='number of images to stack in a frame')

    # Misc Args
    parser.add_argument('--seed', type=int, default=0, help='seed value for randomizing')
    parser.add_argument('--save_freq', type=int, default=50, help='save model every N iterations')
    parser.add_argument('--log_freq', type=int, default=1, help='log progress every N iterations')

    # Return the parser with added arguments
    return parser
