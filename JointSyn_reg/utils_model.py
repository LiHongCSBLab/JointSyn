import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
import math

import os
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from scipy.stats import pearsonr


def train(model, device, train_loader, optimizer, criterion, epoch):
    model.train()
    running_loss = 0.0
    total_loss = 0.0
    count = 0
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        labels = labels.reshape(labels.shape[0], 1).to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        labels = labels.float()
        outputs = outputs.float()
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        total_loss += loss.item()
        count += 1
        running_loss = 0.0
    return total_loss/count


def metric(compare):
    y_true = compare['true']
    y_pred = compare['pred']
    mse = mean_squared_error(y_true, y_pred)
    rmse = math.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    pear = pearsonr(y_true, y_pred)[0]

    return mse, rmse, r2, pear
    
    
def valid(model, device, valid_loader, criterion):
    model.eval()
    compare = pd.DataFrame(columns=('pred','true'))
    total_loss = 0.0
    count = 0
    with torch.no_grad():
        for data in valid_loader:
            inputs, labels = data
            labels = labels.reshape(labels.shape[0], 1).to(device)

            outputs = model(inputs)
            labels = labels.float()
            outputs = outputs.float()
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            count += 1

            labels = labels.cpu()
            outputs = outputs.cpu()
            labels_list = np.array(labels)[:,0].tolist()
            outputs_list = np.array(outputs)[:,0].tolist()
            compare_temp = pd.DataFrame(columns=('pred','true'))
            compare_temp['true'] = labels_list
            compare_temp['pred'] = outputs_list
            compare = pd.concat([compare,compare_temp])
    compare_copy = compare.copy()
    mse, rmse, r2, pear = metric(compare_copy)
    return total_loss/count, mse, rmse, r2, pear


def save_model(current_pear, best_pear, epoch, model, optimizer, log_dir_best):
    is_best = current_pear > best_pear
    best_pear = max(current_pear, best_pear)
    checkpoint = {
        'best_pear': best_pear,
        'epoch': epoch,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }
    if is_best:
        torch.save(checkpoint, log_dir_best)
    return best_pear


def test(model, device, test_loader):
    model.eval()
    compare = pd.DataFrame(columns=('pred','true'))
    with torch.no_grad():
        for data in test_loader:
            inputs, labels = data
            labels = labels.reshape(labels.shape[0], 1).to(device)
            outputs = model(inputs)

            labels = labels.cpu()
            outputs = outputs.cpu()
            labels_list = np.array(labels)[:,0].tolist()
            outputs_list = np.array(outputs)[:,0].tolist()
            compare_temp = pd.DataFrame(columns=('pred','true'))
            compare_temp['true'] = labels_list
            compare_temp['pred'] = outputs_list
            compare = pd.concat([compare,compare_temp])
    return compare

