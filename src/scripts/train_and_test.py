from models.common import Config
from models.preprocessor import PreProcessor, OpenL3PreProcessor
from external.model import Musicnn
from models.tester import Tester
from models.trainer import Trainer

if __name__ == '__main__':
    config = Config(num_workers=0,
                    model=Musicnn(),
                    n_epochs=5,
                    batch_size=16,
                    lr=1e-4,
                    model_save_path="models/musicnn",
                    data_path='data',
                    log_step=100,
                    sr=16000,
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
