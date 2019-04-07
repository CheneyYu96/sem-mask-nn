import torch
import torchvision

import logging
import random as rd
import argparse
import resource
import click
import numpy as np

from utils import *
from train import *
from nets import models
from dataset import get_test_img

rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (1000, rlimit[1]))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s'
)

@click.command()
@click.option('--cmd', type=int, default=0, help='Command 0: inference; 1:train')
@click.option('--net', type=str, default='fcn32', help='Assigned network')
@click.option('--gpu', type=str, default='-1', help='ID of GPU device; -1 if not use GPU')
@click.option('--batch', type=int, default=1)
@click.option('--epochs', type=int, default=25)
@click.option('--lr', type=float, default=1e-4)
@click.option('--momentum', type=float, default=0)
@click.option('--w-decay', type=float, default=1e-5)
@click.option('--step-size', type=int, default=10)
@click.option('--gamma', type=float, default=0.5)
@click.option('--path', type=click.Path(exists=True, resolve_path=True))
def main(cmd, net, gpu, batch, epochs, lr, momentum, w_decay, step_size, gamma, path):
    vals['batch_size'] = batch
    vals['epochs'] = epochs
    vals['lr'] = lr
    vals['momentum'] = momentum
    vals['w_decay'] = w_decay
    vals['step_size'] = step_size
    vals['gamma'] = gamma

    gpu_ids = []
    if gpu != '-1' and torch.cuda.is_available():
        vals['use_gpu'] = 1
        gpu_ids = [ int(i) for i in gpu.split(',')]
        logging.info('Use GPU. device: {}'.format(gpu_ids))
        # torch.cuda.set_device(gpu)

    if cmd:
        model = models.all_models[net](n_class)
        if vals['use_gpu']:
            model = nn.DataParallel(model,device_ids=gpu_ids)
            model = model.cuda()
        fine_tune(model, net)
    else:
        logging.info('Model path: {}'.format(path))
        if path != None:
            load_model = torch.load(path)
            if vals['use_gpu']:
                load_model = nn.DataParallel(load_model,device_ids=gpu_ids)
                load_model = load_model.cuda()
            infer(load_model)

def infer(model):
    test_dir = '{}/test/images'.format(DATA_DIR)
    img_names = [ f for f in os.listdir(test_dir)]
    for name in img_names:
        image = get_test_img(test_dir, name)
        outputs = model(image)
        torchvision.utils.save_image(outputs, get_pred_name(name))
            
if __name__ == '__main__':
    main()
