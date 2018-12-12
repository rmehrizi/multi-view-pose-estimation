# Rahil Mehrizi, Oct 2018
"""Predicting 3D Poses from single-view 2D Joints"""

import torch
import torch.backends.cudnn as cudnn
import torch.optim
from torch.nn.utils import clip_grad_norm
import os, time
import numpy as np
from collections import OrderedDict

from options.train_options import TrainOptions
from utils.util import AverageMeter
from utils.util import TrainHistory
from utils.checkpoint import Checkpoint
import PoseError
import data
from linear import MakeLinearModel

cudnn.benchmark = True

def main():
    opt = TrainOptions().parse() 
    train_history = TrainHistory()
    checkpoint = Checkpoint(opt)
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu_id

    """Architecture"""
    net = MakeLinearModel(1024,16)
    net = torch.nn.DataParallel(net).cuda()

    """Uploading Mean and SD"""
    path_to_data = '/home/rahil/PoseEstimator/dataset/'

    #mean and sd of 2d poses in training dataset
    Mean_2D = np.loadtxt(path_to_data + 'Mean_2D.txt')
    Mean_2D = Mean_2D.astype('float32')
    Mean_2D = torch.from_numpy (Mean_2D)

    Mean_Delta = np.loadtxt(path_to_data + 'Mean_Delta.txt')
    Mean_Delta = Mean_Delta.astype('float32')
    Mean_Delta = torch.from_numpy (Mean_Delta)

    SD_2D = np.loadtxt(path_to_data + 'SD_2D.txt')
    SD_2D = SD_2D.astype('float32')
    SD_2D = torch.from_numpy (SD_2D)

    SD_Delta = np.loadtxt(path_to_data + 'SD_Delta.txt')
    SD_Delta = SD_Delta.astype('float32')
    SD_Delta = torch.from_numpy (SD_Delta)

    """Loading Data"""
    train_list = 'train_list.txt' 
    train_loader = torch.utils.data.DataLoader(
        data.PtsList(train_list, is_train=True),
        batch_size=opt.bs, shuffle=True,
        num_workers=opt.nThreads, pin_memory=True)

    val_list = 'valid_list.txt'
    val_loader = torch.utils.data.DataLoader(
        data.PtsList(val_list,is_train=False ),
        batch_size=opt.bs, shuffle=False,
        num_workers=opt.nThreads, pin_memory=True)


    """optimizer"""
    optimizer = torch.optim.Adam( net.parameters(), lr=opt.lr, betas=(0.9,0.999), weight_decay=0)

    """training and validation"""
    for epoch in range(0, opt.nEpochs):
        adjust_learning_rate(optimizer, epoch, opt.lr)

        # train for one epoch
        train_loss = train(train_loader, net, Mean_2D, Mean_Delta, SD_2D, SD_Delta, optimizer, epoch, opt)

        # evaluate on validation set
        val_loss,val_pckh = validate(val_loader, net, Mean_2D, Mean_Delta, SD_2D, SD_Delta, epoch, opt)

        # update training history
        e = OrderedDict( [('epoch', epoch)] )
        lr = OrderedDict( [('lr', opt.lr)] )
        loss = OrderedDict( [('train_loss', train_loss),('val_loss', val_loss)] )
        pckh = OrderedDict( [('val_pckh', val_pckh)] )
        train_history.update(e, lr, loss, pckh)
        checkpoint.save_checkpoint(net, train_history, 'best-single.pth.tar')

def train(train_loader, net,  Mean_2D, Mean_Delta, SD_2D, SD_Delta, optimizer, epoch, opt):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    dist_errors = AverageMeter()

    # switch to train mode
    net.train()

    end = time.time()
    DIST_ERROR=0

    for i, (pts3d, pts2d, name) in enumerate(train_loader):

        """measure data loading time"""
        data_time.update(time.time() - end)

        # input and groundtruth
        pts2d = (pts2d - Mean_2D)/SD_2D 
        pts2d = torch.autograd.Variable(pts2d.cuda(async=True),requires_grad=False) 

        pts3d = pts3d.narrow(1,1,16) #remove pelvis center
        pts3d = (pts3d - Mean_Delta)/SD_Delta 

        target_var = torch.autograd.Variable(pts3d.cuda(async=True),requires_grad=False)  

        # output 
        output = net(pts2d)
    
        #loss   
        loss =  (output - target_var)**2 
        loss = loss.sum() / loss.numel()

        # gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        #max norm constrain
        torch.nn.utils.clip_grad_norm(net.parameters(),1)
  
        optimizer.step()

        #measure optimization time
        batch_time.update(time.time() - end)
        end = time.time()

        # print log
        losses.update(loss.data[0])
        loss_dict = OrderedDict( [('loss', losses.avg),
                                  ('dist_error', dist_errors.avg)] )
        #3d body pose error calculation
        s3d = pts3d * SD_Delta + Mean_Delta
        pred_pts = output.cuda().data 
        pred_pts = pred_pts * SD_Delta.cuda() + Mean_Delta.cuda()

        dist_error = PoseError.Dist_Err(pred_pts, s3d.cuda())
        dist_errors.update(dist_error)

        DIST_ERROR=DIST_ERROR+dist_error

        if i % opt.print_freq == 0:
            print('Epoch:[{0}][{1}/{2}] '
                  'Bs:[{3}] '
                  'dist_error:[{4:.4f}] '
                  'Time:[{batch_time.val:.3f}]({batch_time.avg:.3f}) '
                  'Data:[{data_time.val:.3f}]({data_time.avg:.3f}) '
                  'Loss:[{loss.val:.4f}]({loss.avg:.4f})'.format(
                   epoch, i, len(train_loader), opt.bs, dist_error,
                   batch_time=batch_time, data_time=data_time, loss=losses))

    DIST_ERROR = DIST_ERROR/len(train_loader)
    print('the average dist_error is')
    print(DIST_ERROR)

    return losses.avg


def validate(val_loader, net, Mean_2D, Mean_Delta, SD_2D, SD_Delta, epoch, opt):

    batch_time = AverageMeter()
    losses_det = AverageMeter()
    losses = AverageMeter()
    dist_errors = AverageMeter()

    # switch to evaluate mode
    net.eval()

    end = time.time()
    DIST_ERROR=0

    for i, (pts3d, pts2d, name) in enumerate(val_loader):  

        # input and groundtruth
        pts2d = (pts2d - Mean_2D)/SD_2D 
        pts2d = torch.autograd.Variable(pts2d.cuda(async=True),requires_grad=False) 

        pts3d = pts3d.narrow(1,1,16) #remove pelvis center
        pts3d = (pts3d - Mean_Delta)/SD_Delta 

        target_var = torch.autograd.Variable(pts3d.cuda(async=True),requires_grad=False)  

        # output 
        output = net(pts2d)

        #loss
        loss =  (output - target_var)**2 
        loss = loss.sum() / loss.numel()

        #3d body pose error calculation
        s3d = pts3d * SD_Delta + Mean_Delta
        pred_pts = output.cuda().data 
        pred_pts = pred_pts * SD_Delta.cuda() + Mean_Delta.cuda()

        dist_error = PoseError.Dist_Err(pred_pts, s3d.cuda())
        dist_errors.update(dist_error)

        DIST_ERROR=DIST_ERROR+dist_error

        """measure elapsed time"""
        batch_time.update(time.time() - end)
        end = time.time()

        # print log
        losses.update(loss.data[0])
        dist_errors.update(dist_error)
        loss_dict = OrderedDict( [('loss', losses.avg),
                                  ('dist_error', dist_errors.avg)] )

        print('Epoch:[{0}][{1}/{2}] '
              'dist_error:[{3:.4f}] '.format(
               epoch, i, len(val_loader), dist_error, ))     

    DIST_ERROR = DIST_ERROR/len(val_loader)
    print('the average dist_error is')
    print(DIST_ERROR)

    return losses.avg, dist_errors.avg

def adjust_learning_rate(optimizer, epoch, lr0):
    lr = lr0 * (0.96 ** (epoch // 4))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    print('Epoch:[%d]\tlr:[%f]' % (epoch, lr))


if __name__ == '__main__':
    main()


