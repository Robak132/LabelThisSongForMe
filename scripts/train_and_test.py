from components.common import Config
from components.tester import Tester
from components.trainer import Trainer
from external.musicnn import Musicnn

if __name__ == '__main__':
    config = Config(model=Musicnn(n_class=10),
                    # preprocessor=PreProcessor(input_path="../data/mtat/mp3", output_path="../data/mtat/npy", suffix="npy", sr=16000),
                    # preprocessor=OpenL3PreProcessor(input_path="../data/mtat/mp3", output_path="../data/mtat/emb", suffix="npy", sr=16000),
                    n_epochs=10,
                    batch_size=16,
                    lr=1e-3,
                    weight_decay=1e-5,
                    dataset_split_path="../split",
                    model_filename_path="../models",
                    logs_path="../runs",
                    dataset_name="mtat-10",
                    data_path='../data',
                    log_step=100,
                    sr=16000,
                    input_length=3 * 16000)

    # Trainer
    trainer = Trainer(config)
    trainer.train()

    # Tester
    tester = Tester(config, model_filename=trainer.model_file_name)
    # tester = Tester(config, model_filename="mtat-10/2023-03-18-11-03-47.pth")
    tester.test()