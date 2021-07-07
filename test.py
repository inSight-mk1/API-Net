import os
import time
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import numpy as np
from models import API_Net
from torchvision import datasets, models, transforms
from datasets import RandomDataset, BatchDataset, BalancedBatchSampler
from utils import accuracy, AverageMeter, save_checkpoint


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def main():
    batch_size = 32
    n_workers = 4
    ckpt_path = 'model_best.pth.tar'

    torch.manual_seed(2)
    torch.cuda.manual_seed_all(2)
    np.random.seed(2)

    # create model
    model = API_Net()
    model = model.to(device)
    model.conv = nn.DataParallel(model.conv)

    checkpoint = torch.load(ckpt_path)
    model.load_state_dict(checkpoint['state_dict'])

    criterion = nn.CrossEntropyLoss().to(device)

    cudnn.benchmark = True

    val_dataset = RandomDataset(transform=transforms.Compose([
        transforms.Resize([512, 512]),
        transforms.CenterCrop([448, 448]),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225)
        )]))
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=n_workers, pin_memory=True)
    prec1 = validate(val_loader, model, criterion)


def validate(val_loader, model, criterion, print_freq=1):
    batch_time = AverageMeter()
    softmax_losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()
    end = time.time()

    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):

            input_var = input.to(device)
            target_var = target.to(device).squeeze()

            # compute output
            logits = model(input_var, targets=None, flag='val')
            softmax_loss = criterion(logits, target_var)


            prec1= accuracy(logits, target_var, 1)
            prec5 = accuracy(logits, target_var, 5)
            softmax_losses.update(softmax_loss.item(), logits.size(0))
            top1.update(prec1, logits.size(0))
            top5.update(prec5, logits.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % print_freq == 0:
                print('Time: {time}\nTest: [{0}/{1}]\t'
                        'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                        'SoftmaxLoss {softmax_loss.val:.4f} ({softmax_loss.avg:.4f})\t'
                        'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                        'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                        i, len(val_loader), batch_time=batch_time, softmax_loss=softmax_losses,
                        top1=top1, top5=top5, time=time.asctime(time.localtime(time.time()))))
        print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'.format(top1=top1, top5=top5))

    return top1.avg


if __name__ == '__main__':
    main()
