from external.musicnn import Musicnn
from components.common import Config
from components.preprocessor import PreProcessor
from components.tester import Tester
from components.trainer import Trainer

if __name__ == '__main__':
    config = Config(model=Musicnn(n_class=20),
                    # preprocessor=PreProcessor(input_path="../data/mtat/mp3", output_path="../data/mtat/npy", suffix="npy", sr=16000),
                    # preprocessor=OpenL3PreProcessor(input_path="../data/mtat/mp3", output_path="../data/mtat/emb", suffix="npy", sr=16000),
                    n_epochs=20,
                    batch_size=16,
                    lr=1e-4,
                    dataset_split_path="../split",
                    model_filename_path="../models",
                    dataset_name="mtat-20",
                    data_path='../data',
                    log_step=100,
                    sr=16000,
                    input_length=3 * 16000)

    trainer = Trainer(config)
    trainer.train()
    tester = Tester(config)
    tester.test(trainer.model_file_name)