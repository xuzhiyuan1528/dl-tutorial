import argparse

class MyArgParser(argparse.ArgumentParser):

    def __init__(self):
        super(MyArgParser, self).__init__()

        self.add_argument(
            '--batch_size',
            type=int,
            default=64,
            help='batch size for train'
        )
