import numpy as np

from src.models.loader import get_audio_loader
from src.models.trainer import Trainer


class Config:
    def __init__(self):
        self.num_workers = 1
        self.dataset = 'mtat'
        self.n_epochs = 5
        self.batch_size = 16
        self.lr = 1e-4
        self.use_tensorboard = 1
        self.model_save_path = "models"
        self.model_load_path = ""
        self.data_path = 'data'
        self.log_step = 20
        self.input_length = 3 * 16000


if __name__ == '__main__':
    config = Config()

    train = np.load("split/mtat-mini/train.npy", allow_pickle=True)
    train_list = list(train)
    for row in range(len(train_list)):
        if list(train_list[row])[1] == "f/jackalopes-jacksploitation-15-drivein_saturday_nite-0-29.mp3":
            train = np.delete(train, row, axis=0)
    np.save(open('split/mtat-mini/train.npy', 'wb'), train)

    train_loader = get_audio_loader(config.data_path,
                                    config.batch_size,
                                    filename='split/mtat-mini/train.npy',
                                    input_length=config.input_length,
                                    num_workers=config.num_workers)

    trainer = Trainer(train_loader, config)
    trainer.train()
