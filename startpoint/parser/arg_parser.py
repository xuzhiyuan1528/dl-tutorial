import argparse


class MyArgParser(argparse.ArgumentParser):

    def __init__(self):
        super(MyArgParser, self).__init__()

        self.add_argument(
            '--batch_size',
            type=int,
            default=64,
            help='batch size for train')

        self.add_argument(
            '--mod_dir',
            type=str,
            default='./mod',
            help='The directory to store model')

        self.add_argument(
            '--multi_gpu',
            action='store_true',
            help='Set to use GPUs')

        self.add_argument(
            '--data_dir',
            type=str,
            default='./dat',
            help='The directory to raw data'
        )

        self.add_argument(
            '--train_epochs',
            type=int,
            default=100,
            help='The epochs for training'
        )


