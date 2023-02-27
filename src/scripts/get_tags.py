import argparse

import torch

from models.provider import Provider

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Script to tag music using selected model')
    parser.add_argument('filename', type=str, help='Path to music file (MP3)')
    parser.add_argument('model', type=str, help='Path to model file (PTH)')

    args = parser.parse_args()
    provider = Provider(args.model)
    data = provider.get_tags(args.filename)
    print(f"File: {args.filename}")
    print(f"Best tag: {provider.tags[max((torch.argmax(data, dim=1)))]}")
