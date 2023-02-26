from models.preprocessor import PreProcessor
from external.model import Musicnn
from models.tester import Tester
from models.trainer import Trainer


class Config:
    def __init__(self, num_workers, model, n_epochs, batch_size, lr, model_save_path, data_path,
                 log_step, input_length, train_path, valid_path, test_path, binary_path):
        self.num_workers = num_workers
        self.model = model
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.model_save_path = model_save_path
        self.data_path = data_path
        self.log_step = log_step
        self.input_length = input_length

        self.train_path = train_path
        self.valid_path = valid_path
        self.test_path = test_path
        self.binary_path = binary_path


if __name__ == '__main__':
    config = Config(num_workers=0,
                    model=Musicnn(dataset='mtat'),
                    n_epochs=5,
                    batch_size=16,
                    lr=1e-4,
                    model_save_path="models/musicnn.pth",
                    data_path='data',
                    log_step=100,
                    input_length=3 * 16000,
                    train_path="split/mtat/train.npy",
                    valid_path="split/mtat/valid.npy",
                    test_path="split/mtat/test.npy",
                    binary_path="split/mtat/binary.npy")

    p = PreProcessor(config)
    p.run()

    trainer = Trainer(config)
    trainer.train()

    tester = Tester(config)
    tester.test()
