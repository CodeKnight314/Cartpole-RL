from environment import Environment
import argparse
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Deep Q Network')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to the config file')
    parser.add_argument('--output', type=str, default='output', help='Path to the output directory')
    args = parser.parse_args()

    if not os.path.exists(args.output):
        os.makedirs(args.output)

    env = Environment(args.config)
    env.train_dqn(args.output)
    env.test_dqn(args.output)
    env.close()