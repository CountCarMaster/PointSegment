import torch
import torch.nn as nn
import numpy as np
import random
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

def train(dataset: DataLoader, model: torch.nn.Module,
          optimizer: torch.optim.Optimizer, seed: int,
          epochs: int, weight_saving_path: str, tensorboard: int,
          device: torch.device, val=1, test_dataset=None,
          tensorboard_log=None, sampled=1):
    writer = None
    if tensorboard == 1:
        writer = SummaryWriter(log_dir=tensorboard_log)

    model.train()
    criterion = nn.CrossEntropyLoss().to(device)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    loss = None
    min_loss = 10000000.0
    max_acc = -1.0
    for epoch in range(epochs):
        model.train()
        if sampled == 1:
            for x, y in dataset:
                x = x.to(device).transpose(1, 2)
                x = x.float()
                y = y.to(device)
                output = model(x)
                loss = criterion(output.type(torch.float32), y.long())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        else :
            for x, y in dataset:
                x = x.to(device).transpose(1, 2)
                x = x.float()
                batch_size, channel_num, now_point_num = x.shape
                indices = torch.randint(0, now_point_num, (batch_size, sampled)).to(device)
                y = y.to(device)
                x = torch.gather(x, 2, indices.unsqueeze(1).expand(-1, channel_num, -1))
                y = torch.gather(y, 1, indices)
                output = model(x)
                loss = criterion(output.type(torch.float32), y.long())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        if tensorboard == 1:
            writer.add_scalar('loss', loss.item(), epoch)

        if loss.item() < min_loss:
            min_loss = loss.item()
            torch.save(model.state_dict(), weight_saving_path + '_loss_best.pth')

        if val != 1:
            print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')

        else:
            model.eval()
            correct_points_all = 0
            total_points_all = 0
            for x, y in test_dataset:
                x = x.to(device).transpose(1, 2)
                x = x.float()
                if sampled != 1:
                    batch_size, channel_num, now_point_num = x.shape
                    indices = torch.randint(0, now_point_num, (batch_size, sampled)).to(device)
                    y = y.to(device)
                    x = torch.gather(x, 2, indices.unsqueeze(1).expand(-1, channel_num, -1))
                    y = torch.gather(y, 1, indices)
                else :
                    y = y.to(device)
                output = model(x)
                predicted_classes = torch.argmax(output, dim=-2)
                correct_points = (predicted_classes == y).sum().item()
                total_points = y.shape[0] * y.shape[1]
                correct_points_all += correct_points
                total_points_all += total_points

            acc = correct_points_all / total_points_all
            if acc > max_acc:
                max_acc = acc
                torch.save(model.state_dict(), weight_saving_path + '_acc_best.pth')

            if tensorboard == 1:
                writer.add_scalar('acc', acc, epoch)

            print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}, Acc: {acc:.4f}')

        torch.save(model.state_dict(), weight_saving_path + '_last.pth')

def test(dataset: DataLoader, model: torch.nn.Module, device: torch.device, sampled=1):
    model.eval()

    # Global for accuracy
    correct_points_all = 0
    total_points_all = 0

    # Global for TP, FP, TN, FN
    true_positive = 0
    false_positive = 0
    true_negative = 0
    false_negative = 0

    for x, y in dataset:
        x = x.to(device).transpose(1, 2)
        x = x.float()
        if sampled != 1:
            batch_size, channel_num, now_point_num = x.shape
            indices = torch.randint(0, now_point_num, (batch_size, sampled)).to(device)
            y = y.to(device)
            x = torch.gather(x, 2, indices.unsqueeze(1).expand(-1, channel_num, -1))
            y = torch.gather(y, 1, indices)
        else:
            y = y.to(device)
        output = model(x)

        # Acc calculation
        predicted_classes = torch.argmax(output, dim=-2)
        correct_points = (predicted_classes == y).sum().item()
        total_points = y.shape[0] * y.shape[1]
        correct_points_all += correct_points
        total_points_all += total_points

        # TP, TN, FP, FN
        true_positive += torch.sum((y == 1) & (predicted_classes == 1))
        false_positive += torch.sum((y == 0) & (predicted_classes == 1))
        true_negative += torch.sum((y == 0) & (predicted_classes == 0))
        false_negative += torch.sum((y == 1) & (predicted_classes == 0))

    precision = true_positive / (true_positive + false_positive)
    recall = true_positive / (true_positive + false_negative)
    F1 = 2 * precision * recall / (precision + recall)
    IoU = true_positive / (true_positive + false_positive + false_negative)

    print('Acc: %s' % (correct_points_all / total_points_all))
    print('TP: %s' % true_positive.item())
    print('FP: %s' % false_positive.item())
    print('TN: %s' % true_negative.item())
    print('FN: %s' % false_negative.item())
    print('Precision: %s' % precision.item())
    print('Recall: %s' % recall.item())
    print('F1: %s' % F1.item())
    print('IoU: %s' % IoU.item())


        
