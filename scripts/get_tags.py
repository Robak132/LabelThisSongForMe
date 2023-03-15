import argparse

from models.common import create_tagogram, plot_probability_graph, Config
from models.tester import Tester

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Script to tag music using selected model')
    parser.add_argument('filename', type=str, help='Path to music file (MP3)')
    parser.add_argument('model', type=str, help='Path to model file (PTH)')
    args = parser.parse_args()

    tester = Tester(config=Config(model_save_path=args.model), cuda=False)
    prediction = tester.predict_tags(mp3_file=args.filename)
    print(f"File: {args.filename}")
    print(f"Tags: {prediction}")
    fig1 = plot_probability_graph(prediction)
    fig1.show()
    fig1.write_image("fig1.png")

    fig2 = create_tagogram(prediction)
    fig2.show()
    fig2.write_image("fig2.png")
