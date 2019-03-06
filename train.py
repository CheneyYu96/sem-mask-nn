import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s'
)

from utils import *

def fine_tune(model, name):
    logging.info("Fine tuning model: {}".format(name))
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.RMSprop(model.parameters(), lr=lr, momentum=momentum, weight_decay=w_decay)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)  # decay LR by a factor of 0.5 every 30 epochs

    # TODO data loader
    train_loader, val_loader = None, None

    train(model, name, criterion, optimizer, scheduler, train_loader, val_loader, epochs)

'''
    Reference: https://github.com/pochih/FCN-pytorch/blob/master/python/train.py
'''

def train(model, name, criterion, optimizer, scheduler, train_loader, val_loader, epochs):
    
    for epoch in range(epochs):
        scheduler.step()

        epoch_losses = []
        since = now()
        for iter, batch in enumerate(train_loader):
            optimizer.zero_grad()
            
            # inputs, labels = None, None
            if use_gpu:
                inputs = Variable(batch['X'].cuda())
                labels = Variable(batch['Y'].cuda())
            else:
                inputs, labels = Variable(batch['X']), Variable(batch['Y'])

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            epoch_losses.append(loss.data[0])
            if iter % 10 == 0:
                logging.info("epoch: {}, iter: {}, loss: {:0.5f}, avg: {:0.5f}"
                    .format(epoch, iter, loss.data[0], np.mean(epoch_losses)))
        
        logging.info("Finish epoch: {}, time: {}, avg_loss: {:0.5f}"
            .format(epoch, gap_time(since), np.mean(epoch_losses)))
        torch.save(model, get_model_path(name, epoch))

        val(model, epoch)

def val(model, val_loader, epoch):
    model.eval()
    total_ious = []
    pixel_accs = []
    for iter, batch in enumerate(val_loader):
        if use_gpu:
            inputs = Variable(batch['X'].cuda())
        else:
            inputs = Variable(batch['X'])

        output = model(inputs)
        output = output.data.cpu().numpy()

        N, _, h, w = output.shape
        pred = output.transpose(0, 2, 3, 1).reshape(-1, n_class).argmax(axis=1).reshape(N, h, w)

        target = batch['l'].cpu().numpy().reshape(N, h, w)
        for p, t in zip(pred, target):
            total_ious.append(iou(p, t))
            pixel_accs.append(pixel_acc(p, t))

    # Calculate average IoU
    total_ious = np.array(total_ious).T  # n_class * val_len
    ious = np.nanmean(total_ious, axis=1)
    pixel_accs = np.array(pixel_accs).mean()
    logging.info("epoch: {}, pix_acc: {}, meanIoU: {}, IoUs: {}".format(epoch, pixel_accs, np.nanmean(ious), ious))

    # IU_scores[epoch] = ious
    # np.save(os.path.join(score_dir, "meanIU"), IU_scores)
    # pixel_scores[epoch] = pixel_accs
    # np.save(os.path.join(score_dir, "meanPixel"), pixel_scores)

# Calculates class intersections over unions
def iou(pred, target):
    ious = []
    for cls in range(n_class):
        pred_inds = pred == cls
        target_inds = target == cls
        intersection = pred_inds[target_inds].sum()
        union = pred_inds.sum() + target_inds.sum() - intersection
        if union == 0:
            ious.append(float('nan'))  # if there is no ground truth, do not include in evaluation
        else:
            ious.append(float(intersection) / max(union, 1))
        # print("cls", cls, pred_inds.sum(), target_inds.sum(), intersection, float(intersection) / max(union, 1))
    return ious


def pixel_acc(pred, target):
    correct = (pred == target).sum()
    total   = (target == target).sum()
    return correct / total