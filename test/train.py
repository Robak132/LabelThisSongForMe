from src.models.trainer import Trainer
from src.models.loader import get_audio_loader


class Config:
    def __init__(self):
        self.num_workers = 0
        self.dataset = 'mtat'
        self.n_epochs = 5
        self.batch_size = 3
        self.lr = 1e-4
        self.use_tensorboard = 1
        self.model_save_path = "models"
        self.model_load_path = ""
        self.data_path = 'data'
        self.log_step = 20
        self.input_length = 3 * 16000


if __name__ == '__main__':
    config = Config()

    train_loader = get_audio_loader(config.data_path,
                                    config.batch_size,
                                    filename='split/mtat-mini/train.npy',
                                    input_length=config.input_length,
                                    num_workers=config.num_workers)

    trainer = Trainer(train_loader, config)
    trainer.train()
