from external.model import Musicnn
from models.common import Config
from models.preprocessor import PreProcessor
from models.tester import Tester
from models.trainer import Trainer

if __name__ == '__main__':
    config = Config(model=Musicnn(n_class=10),
                    # preprocessor=PreProcessor(input_path="data/mtat/mp3", output_path="data/mtat/npy", suffix="npy", sr=16000),
                    # preprocessor=OpenL3PreProcessor(input_path="data/mtat/mp3", output_path="data/mtat/emb", suffix="npy", sr=16000),
                    n_epochs=5,
                    batch_size=16,
                    lr=1e-4,
                    model_filename_path="models",
                    data_path='data',
                    log_step=100,
                    sr=16000,
                    input_length=3 * 16000,
                    tags_path="split/mtat-10/tags.npy",
                    train_path="split/mtat-10/train.npy",
                    valid_path="split/mtat-10/valid.npy",
                    test_path="split/mtat-10/test.npy",
                    binary_path="split/mtat-10/binary.npy")

    tester = Tester(config, "2023-03-18-11-03-47.pth")
    tester.test()