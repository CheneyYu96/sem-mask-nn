import torch

import logging
import random as rd
import argparse
import resource
import click
import numpy as np

from utils import *
from train import *
from nets import models

rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (1000, rlimit[1]))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s'
)

@click.command()
@click.option('--cmd', type=int, default=0, help='Command 0: inference; 1:train')
@click.option('--net', type=str, default='fcn32', help='Assigned network')
@click.option('--gpu', type=int, default=-1, help='ID of GPU device; -1 if not use GPU')
def main(cmd, net, gpu):
    if gpu >= 0 and torch.cuda.is_available():
        use_gpu = 1
        logging.info('Use GPU. device: {}'.format(gpu))
        torch.cuda.set_device(gpu)

    model = models.all_models[net](n_class)
    if cmd:
        fine_tune(model, net)
    else:
        pass

if __name__ == '__main__':
    main()
