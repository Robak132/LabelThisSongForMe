import argparse

from matplotlib import pyplot as plt

from models.common import create_tagogram, plot_probability_graph
from models.tester import Tester

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Script to tag music using selected model')
    parser.add_argument('filename', type=str, help='Path to music file (MP3)')
    parser.add_argument('model', type=str, help='Path to model file (PTH)')
    args = parser.parse_args()

    tester = Tester()
    raw_data, raw_tags, prediction = tester.predict_tags(mp3_file=args.filename)
    print(f"File: {args.filename}")
    print(f"Tags: {prediction}")
    create_tagogram(raw_data, tester.tags)
    plot_probability_graph(prediction)
    plt.show()
