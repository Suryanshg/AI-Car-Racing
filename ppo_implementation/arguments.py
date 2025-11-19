import argparse

def get_args(parser):
    '''
    TODO:
    '''
    # Deep Learning Hyperparameters
    parser.add_argument('--lr', type=float, default=1e-5, help='learning rate')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--grad_clip_value', type=float, default=1.0, help='max grad value')
    parser.add_argument('--train_freq', type=int, default=4, help='Train every N steps')

    # PPO Hyperparameters
    parser.add_argument('--gamma', type=float, default=0.99, help='discount factor')

    # Environment Wrapper Specific Args
    parser.add_argument('--action_repetition', type=int, default=8, help='number of times to repeat the action')
    parser.add_argument('--frame_stack_size', type=int, default=4, help='number of images to stack in a frame')

    # Misc Args
    parser.add_argument('--seed', type=int, default=0, help='seed value for randomizing')

    # Return the parser with added arguments
    return parser
