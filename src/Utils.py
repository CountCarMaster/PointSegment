import torch
import os
import torch.optim as optim
import torch.nn as nn
from models.PointNet import PointNet
from models.PointNet2 import PointNet2
from models.DGCNN import DGCNN
from models.RepSurf import RepSurf
from models.PointMLP import PointMLP
from models.PointNeXT import PointNeXT
from models.GANN import GANN

import torch
import torch.nn.functional as F

from models.RepSurf import RepSurf
from models.TempModel import TempModel


class FocalLoss(torch.nn.Module):
    def __init__(self, alpha=0.25, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        inputs = F.softmax(inputs, dim=1)
        p_t = inputs.gather(dim=1, index=targets.view(-1, 1))
        loss = - (self.alpha * (1 - p_t) ** self.gamma * torch.log(p_t))
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

def config_checker(config: dict):
    # completeness checking
    assert 'datasetRootDir' in config.keys(), 'datasetRootDir is required.'
    assert 'batchSize' in config.keys(), 'batchSize is required.'
    assert 'fullPointNum' in config.keys(), 'fullPointNum is required.'
    assert 'downSample' in config.keys(), 'downSample is required.'
    assert 'downSamplePointNum' in config.keys(), 'downSamplePointNum is required.'
    assert 'channelNum' in config.keys(), 'channelNum is required.'
    assert 'trainDataNum' in config.keys(), 'trainDataNum is required.'
    assert 'testDataNum' in config.keys(), 'testDataNum is required.'
    assert 'model' in config.keys(), 'model is required.'
    assert 'device' in config.keys(), 'device is required.'
    assert 'outputNum' in config.keys(), 'outputNum is required.'
    assert 'mode' in config.keys(), 'mode is required.'
    assert 'learningRate' in config.keys(), 'learningRate is required.'
    assert 'epochs' in config.keys(), 'epochs is required.'
    assert 'optimizer' in config.keys(), 'optimizer is required.'
    assert 'weightDecay' in config.keys(), 'weightDecay is required.'
    assert 'seed' in config.keys(), 'seed is required.'
    assert 'val' in config.keys(), 'val is required.'
    assert 'weightSavingPath' in config.keys(), 'weightSavingPath is required.'
    assert 'checkpoint' in config.keys(), 'checkpoint is required.'
    assert 'weightPath' in config.keys(), 'weightPath is required.'
    assert 'enableTensorboard' in config.keys(), 'enableTensorboard is required.'
    assert 'logPath' in config.keys(), 'logPath is required.'

    assert os.path.exists(config['datasetRootDir']), 'datasetRootDir is not a valid path.'
    assert isinstance(config['batchSize'], int), 'batchSize is not an integer.'
    assert config['batchSize'] > 0, 'batchSize is not positive.'
    assert isinstance(config['fullPointNum'], int), 'fullPointNum is not an integer.'
    assert config['fullPointNum'] > 0, 'fullPointNum is not positive.'
    assert config['downSample'] == 1 or config['downSample'] == 0, 'downSample must be 0 or 1.'
    if config['downSample'] == 1:
        assert isinstance(config['downSamplePointNum'], int), 'downSamplePointNum is not an integer.'
        assert config['downSamplePointNum'] > 0, 'downSamplePointNum is not positive.'
    assert config['downSamplePointNum'] <= config['fullPointNum'], 'downSamplePointNum must be less than fullPointNum.'
    assert isinstance(config['channelNum'], int), 'channelNum is not an integer.'
    assert config['channelNum'] > 0, 'channelNum is not positive.'
    assert isinstance(config['trainDataNum'], int), 'trainDataNum is not an integer.'
    assert config['trainDataNum'] > 0, 'trainDataNum is not positive.'
    assert isinstance(config['testDataNum'], int), 'testDataNum is not an integer.'
    assert config['testDataNum'] > 0, 'testDataNum is not positive.'
    assert config['device'] == 'cpu' or config['device'] == 'cuda', 'device is not supported.'
    assert isinstance(config['outputNum'], int), 'outputNum is not an integer.'
    assert config['outputNum'] > 0, 'outputNum is not positive.'
    assert config['mode'] == 'train' or config['mode'] == 'test', 'mode is not supported.'

    if config['mode'] == 'train':
        assert isinstance(config['learningRate'], float), 'learningRate is not a float.'
        assert config['learningRate'] > 0, 'learningRate is not positive.'
        assert isinstance(config['epochs'], int), 'epochs is not a integer.'
        assert config['epochs'] > 0, 'epochs is not positive.'
        assert isinstance(config['weightDecay'], float), 'weightDecay is not a float.'
        assert config['weightDecay'] > 0, 'weightDecay is not positive.'
        assert isinstance(config['seed'], int), 'seed is not an integer.'
        assert config['seed'] > 0, 'seed is not positive.'
        assert config['val'] == 0 or config['val'] == 1, 'val must be 0 or 1.'
        assert os.path.exists(config['weightSavingPath']), 'weightSavingPath is not a valid path.'
        assert os.path.exists(config['checkpoint']) or config['checkpoint'] == 'None', 'checkpoint is not a valid path.'

    if config['mode'] == 'test':
        assert os.path.exists(config['weightPath']), 'weightPath is not a valid path.'

    assert config['enableTensorboard'] == 0 or config['enableTensorboard'] == 1, 'enableTensorBoard is not 0 or 1.'
    if config['enableTensorboard'] == 1:
        assert os.path.exists(config['logPath']), 'logPath is not a valid path.'

    # check model settings
    if config['model'] == 'DGCNN':
        assert isinstance(config['k'], int), 'k is not an integer.'
        assert config['k'] > 0, 'k is not positive.'

def loss_function_chooser(function_name, device, weight=None, alpha=0.25, gamma=2, reduction='mean'):
    if function_name == 'Cross Entropy Loss':
        nn.CrossEntropyLoss(weight).to(device)
    elif function_name == 'Focal Loss':
        return FocalLoss(alpha=alpha, gamma=gamma, reduction=reduction).to(device)
    else:
        raise KeyError('Unknown loss function.')

def model_chooser(model_name, device, input_channel, output_channel, k):
    if model_name == 'PointNet':
        return PointNet(input_channel, output_channel).to(device)
    elif model_name == 'PointNet++':
        return PointNet2(output_channel).to(device)
    elif model_name == 'DGCNN':
        return DGCNN(input_channel, output_channel, k=k).to(device)
    elif model_name == 'RepSurf':
        return RepSurf(input_channel, output_channel, k=k).to(device)
    elif model_name == 'PointMLP' :
        return PointMLP(output_channel).to(device)
    elif model_name == 'PointNeXT':
        return PointNeXT(output_channel).to(device)
    elif model_name == 'GANN':
        return GANN(input_channel, output_channel, k).to(device)
    elif model_name == 'TempModel':
        return TempModel(output_channel).to(device)
    else :
        raise KeyError('Unknown model name.')

def optimizer_chooser(model, optimizer_name: str, learning_rate: float, weight_decay=None):
    if optimizer_name == 'Adam':
        return optim.Adam(model.parameters(), lr=learning_rate)
    elif optimizer_name == 'AdamW':
        return optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_name == 'SGD':
        return optim.SGD(model.parameters(), lr=learning_rate)
    elif optimizer_name == 'RMSprop':
        return optim.RMSprop(model.parameters(), lr=learning_rate)
    elif optimizer_name == 'Nadam':
        return optim.NAdam(model.parameters(), lr=learning_rate)
    else:
        raise KeyError('Unknown optimizer name. Please choose the optimizer in '
                       'Adam, AdamW, SGD, RMSprop and Nadam')