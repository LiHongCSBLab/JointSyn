import torch
import torch.nn.functional as F
from torchvision import transforms
import pandas as pd
import numpy as np
import os

from sklearn.metrics import auc
from sklearn.metrics import roc_curve
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import cohen_kappa_score


def train(model, device, train_loader, optimizer, criterion, epoch):
    model.train()
    running_loss = 0.0
    total_loss = 0.0
    count = 0
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        labels = labels.reshape(labels.shape[0], 1).to(device)
        labels = labels.squeeze(1)

        optimizer.zero_grad()
        outputs = model(inputs)
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
    y_true = y_true.astype('int64')
    y_pred = y_pred.astype('int64')
    bacc = balanced_accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    kappa = cohen_kappa_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    
    y_prob = compare['prob']
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    prec, tpr, thresholds = precision_recall_curve(y_true, y_prob)
    prc_auc = auc(tpr, prec)
    return roc_auc, prc_auc, bacc, precision, recall, kappa, f1
    
    
def valid(model, device, valid_loader, criterion):
    model.eval()
    compare = pd.DataFrame(columns=('pred','true'))
    total_loss = 0.0
    count = 0
    with torch.no_grad():
        for data in valid_loader:
            inputs, labels = data
            labels = labels.reshape(labels.shape[0], 1).to(device)
            labels = labels.squeeze(1)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            count += 1

            probs = F.softmax(outputs, dim=1)
            probs = probs[:,1]
            _, predicteds = torch.max(outputs.data, 1)
            
            labels = labels.cpu()
            predicteds = predicteds.cpu()
            probs = probs.cpu()
            labels_list = np.array(labels).tolist()
            predicteds_list = np.array(predicteds).tolist()
            probs_list = np.array(probs).tolist()
            compare_temp = pd.DataFrame(columns=('pred','true'))
            compare_temp['true'] = labels_list
            compare_temp['pred'] = predicteds_list
            compare_temp['prob'] = probs_list
            compare = pd.concat([compare,compare_temp])
    compare_copy = compare.copy()
    roc_auc, prc_auc, bacc, precision, recall, kappa, f1 = metric(compare_copy)
    return total_loss/count, roc_auc, prc_auc, bacc, precision, recall, kappa, f1


def save_model(current_f1, best_f1, epoch, model, optimizer, log_dir_best):
    is_best = current_f1 > best_f1
    best_f1 = max(current_f1, best_f1)
    checkpoint = {
        'best_f1': best_f1,
        'epoch': epoch,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }
    if is_best:
        torch.save(checkpoint, log_dir_best)
    return best_f1


def test(model, device, test_loader):
    model.eval()
    compare = pd.DataFrame(columns=('pred','true'))
    with torch.no_grad():
        for data in test_loader:
            inputs, labels = data
            labels = labels.reshape(labels.shape[0], 1).to(device)
            labels = labels.squeeze(1)
            outputs = model(inputs)

            probs = F.softmax(outputs, dim=1)
            probs = probs[:,1]
            _, predicteds = torch.max(outputs.data, 1)
            
            labels = labels.cpu()
            predicteds = predicteds.cpu()
            probs = probs.cpu()
            labels_list = np.array(labels).tolist()
            predicteds_list = np.array(predicteds).tolist()
            probs_list = np.array(probs).tolist()
            compare_temp = pd.DataFrame(columns=('pred','true'))
            compare_temp['true'] = labels_list
            compare_temp['pred'] = predicteds_list
            compare_temp['prob'] = probs_list
            compare = pd.concat([compare,compare_temp])
    return compare

