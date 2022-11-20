import os
import argparse
from src.models.solver import Solver
from mtat_loader import get_audio_loader


def main(config):
    # path for models
    if not os.path.exists(config.model_save_path):
        os.makedirs(config.model_save_path)

    train_loader = get_audio_loader(config.data_path,
                                    config.batch_size,
                                    split='TRAIN',
                                    input_length=3 * 16000,
                                    num_workers=config.num_workers)
    solver = Solver(train_loader, config)
    solver.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--dataset', type=str, default='mtat', choices=['mtat', 'msd', 'jamendo'])
    parser.add_argument('--model_type', type=str, default='musicnn',
                        choices=['fcn', 'musicnn', 'crnn', 'sample', 'se', 'short', 'short_res', 'attention', 'hcnn'])
    parser.add_argument('--n_epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--use_tensorboard', type=int, default=1)
    parser.add_argument('--model_save_path', type=str, default='../../models')
    parser.add_argument('--model_load_path', type=str, default='.')
    parser.add_argument('--data_path', type=str, default='./data')
    parser.add_argument('--log_step', type=int, default=20)

    config = parser.parse_args()

    print(config)
    main(config)
