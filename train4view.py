# Rahil Mehrizi, Oct 2018
"""Predicting 3D Poses from Multi-view 2D Joints"""

import torch
import torch.backends.cudnn as cudnn
import torch.optim
from torch.nn.utils import clip_grad_norm
import os, time
import numpy as np
from collections import OrderedDict
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from options.train_options import TrainOptions
from utils.util import AverageMeter
from utils.util import TrainHistory
from utils.checkpoint import Checkpoint
import PoseError
import data4view
from linear4view import MakeLinearModel
import PXIO
from Plot import Plot2d, Plot3d

evaluate_mode= False
demo_mode= False


cudnn.benchmark = True 
def main():
    opt = TrainOptions().parse() 
    train_history = TrainHistory()
    checkpoint = Checkpoint(opt)
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu_id

    """Architecture"""
    net = MakeLinearModel(1024,16)

    net = torch.nn.DataParallel(net).cuda()
    checkpoint.load_checkpoint(net, train_history, '/best-single.pth.tar')

    """Uploading Mean and SD"""
    path_to_data = '.../multi-view-pose-estimation/dataset/'

    #mean and sd of 2d poses in training dataset
    Mean_2D = np.loadtxt(path_to_data + 'Mean_2D.txt')
    Mean_2D = Mean_2D.astype('float32')
    Mean_2D = torch.from_numpy (Mean_2D)

    Mean_Delta = np.loadtxt(path_to_data + 'Mean_Delta.txt')
    Mean_Delta = Mean_Delta.astype('float32')
    Mean_Delta = torch.from_numpy (Mean_Delta)
    Mean_Delta = torch.autograd.Variable(Mean_Delta.cuda(async=True),requires_grad=False)

    Mean_3D = np.loadtxt(path_to_data + 'Mean_3D.txt')
    Mean_3D = Mean_3D.astype('float32')
    Mean_3D = torch.from_numpy (Mean_3D)
    Mean_3D = torch.autograd.Variable(Mean_3D.cuda(async=True),requires_grad=False)

    SD_2D = np.loadtxt(path_to_data + 'SD_2D.txt')
    SD_2D = SD_2D.astype('float32')
    SD_2D = torch.from_numpy (SD_2D)

    SD_Delta = np.loadtxt(path_to_data + 'SD_Delta.txt')
    SD_Delta = SD_Delta.astype('float32')
    SD_Delta = torch.from_numpy (SD_Delta)
    SD_Delta = torch.autograd.Variable(SD_Delta.cuda(async=True),requires_grad=False)

    SD_3D = np.loadtxt(path_to_data + 'SD_3D.txt')
    SD_3D = SD_3D.astype('float32')
    SD_3D = torch.from_numpy (SD_3D)
    SD_3D = torch.autograd.Variable(SD_3D.cuda(async=True),requires_grad=False)

    """Loading Data"""
    train_list = 'train_list_4view.txt'
    train_loader = torch.utils.data.DataLoader(
        data4view.PtsList(train_list, is_train=True),
        batch_size=opt.bs, shuffle=True,
        num_workers=opt.nThreads, pin_memory=True)

    val_list = 'valid_list_4view.txt'
    val_loader = torch.utils.data.DataLoader(
        data4view.PtsList(val_list,is_train=False ),
        batch_size=opt.bs, shuffle=False,
        num_workers=opt.nThreads, pin_memory=True)

    demo_list = 'demo_list_4view.txt'
    demo_loader = torch.utils.data.DataLoader(
        data4view.PtsList(demo_list,is_train=False ),
        batch_size=1, shuffle=False,
        num_workers=opt.nThreads, pin_memory=True)

    """Optimizer"""
    optimizer = torch.optim.Adam( net.parameters(), lr=opt.lr, betas=(0.9,0.999), weight_decay=0)

    """Validation"""
    if evaluate_mode:
        # evaluate on validation set
        checkpoint.load_checkpoint(net, train_history, '/best-multi.pth.tar')
        val_loss,val_pckh = validate(val_loader, net, Mean_2D, Mean_Delta, Mean_3D, SD_2D, SD_Delta, SD_3D, 0, opt)
        return

    """Demo"""
    if demo_mode:
        # Grab a random batch to visualize
        checkpoint.load_checkpoint(net, train_history, '/best-multi.pth.tar')
        demo(demo_loader, net, Mean_2D, Mean_Delta, Mean_3D, SD_2D, SD_Delta, SD_3D, 0, opt)
        return

    """Training"""
    for epoch in range(0, opt.nEpochs):
        adjust_learning_rate(optimizer, epoch, opt.lr)

        # train for one epoch
        train_loss = train(train_loader, net, Mean_2D, Mean_Delta, Mean_3D, SD_2D, SD_Delta, SD_3D, optimizer, epoch, opt)

        # evaluate on validation set
        val_loss,val_pckh = validate(val_loader, net, Mean_2D, Mean_Delta, Mean_3D, SD_2D, SD_Delta, SD_3D, epoch, opt)

        # update training history
        e = OrderedDict( [('epoch', epoch)] )
        lr = OrderedDict( [('lr', opt.lr)] )
        loss = OrderedDict( [('train_loss', train_loss),('val_loss', val_loss)] )
        pckh = OrderedDict( [('val_pckh', val_pckh)] )
        train_history.update(e, lr, loss, pckh)
        checkpoint.save_checkpoint(net, train_history)


def train(train_loader, net, Mean_2D, Mean_Delta, Mean_3D, SD_2D, SD_Delta, SD_3D, optimizer, epoch, opt):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    dist_errors = AverageMeter()

    # switch to train mode
    net.train()

    end = time.time()
    DIST_ERROR=0

    for i, (pts3dg, root, pts2d1, pts2d2, pts2d3, pts2d4, R1, V1, R2, V2, R3, V3, R4, V4, name) in enumerate(train_loader):

        #measure data loading time
        data_time.update(time.time() - end)

        # input and groundtruth
        pts2d1 = (pts2d1 - Mean_2D)/SD_2D 
        pts2d1 = torch.autograd.Variable(pts2d1.cuda(async=True),requires_grad=False) 

        pts2d2 = (pts2d2 - Mean_2D)/SD_2D 
        pts2d2 = torch.autograd.Variable(pts2d2.cuda(async=True),requires_grad=False) 

        pts2d3 = (pts2d3 - Mean_2D)/SD_2D 
        pts2d3 = torch.autograd.Variable(pts2d3.cuda(async=True),requires_grad=False) 

        pts2d4 = (pts2d4 - Mean_2D)/SD_2D 
        pts2d4 = torch.autograd.Variable(pts2d4.cuda(async=True),requires_grad=False) 

        pts3dg = pts3dg.narrow(1,1,16) #remove pelvis center
        pts3dg = pts3dg.contiguous()
        pts3dg = torch.autograd.Variable(pts3dg.cuda(async=True),requires_grad=False) 

        R1 = torch.autograd.Variable(R1.cuda(async=True),requires_grad=False)
        R2 = torch.autograd.Variable(R2.cuda(async=True),requires_grad=False)
        R3 = torch.autograd.Variable(R3.cuda(async=True),requires_grad=False)
        R4 = torch.autograd.Variable(R4.cuda(async=True),requires_grad=False)

        V1 = torch.autograd.Variable(V1.cuda(async=True),requires_grad=False)
        V2 = torch.autograd.Variable(V2.cuda(async=True),requires_grad=False)
        V3 = torch.autograd.Variable(V3.cuda(async=True),requires_grad=False)
        V4 = torch.autograd.Variable(V4.cuda(async=True),requires_grad=False)

        # output 
        output = net(pts2d1, pts2d2, pts2d3, pts2d4, Mean_Delta, SD_Delta, R1, R2, R3, R4, V1, V2, V3, V4)

        #loss   
        loss =   (output - pts3dg)**2 
        loss = loss.sum() / loss.numel()

        # gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()        
        torch.nn.utils.clip_grad_norm(net.parameters(),1) #max norm constrain  
        optimizer.step()

        #measure optimization time
        batch_time.update(time.time() - end)
        end = time.time()

        # print log
        losses.update(loss.data[0])
        loss_dict = OrderedDict( [('loss', losses.avg),
                                  ('dist_error', dist_errors.avg)] )

        #3d body pose error calculation
        pred_pts = output.cuda().data 
        s3d = pts3dg.cuda().data 
        dist_error = PoseError.Dist_Err(pred_pts, s3d.cuda())
        dist_errors.update(dist_error)

        DIST_ERROR = DIST_ERROR + dist_error

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


def validate(val_loader, net, Mean_2D, Mean_Delta, Mean_3D, SD_2D, SD_Delta, SD_3D, epoch, opt):

    batch_time = AverageMeter()
    losses_det = AverageMeter()
    losses = AverageMeter()
    dist_errors = AverageMeter()

    # switch to evaluate mode
    net.eval()

    end = time.time()
    DIST_ERROR=0

    for i, (pts3dg,  root, pts2d1, pts2d2, pts2d3, pts2d4, R1, V1, R2, V2, R3, V3, R4, V4, name) in enumerate(val_loader): 

        # input and groundtruth 

        pts2d1 = (pts2d1 - Mean_2D)/SD_2D 
        pts2d1 = torch.autograd.Variable(pts2d1.cuda(async=True),requires_grad=False) 

        pts2d2 = (pts2d2 - Mean_2D)/SD_2D 
        pts2d2 = torch.autograd.Variable(pts2d2.cuda(async=True),requires_grad=False) 

        pts2d3 = (pts2d3 - Mean_2D)/SD_2D 
        pts2d3 = torch.autograd.Variable(pts2d3.cuda(async=True),requires_grad=False) 

        pts2d4 = (pts2d4 - Mean_2D)/SD_2D 
        pts2d4 = torch.autograd.Variable(pts2d4.cuda(async=True),requires_grad=False) 

        pts3dg = pts3dg.narrow(1,1,16) #remove pelvis center
        pts3dg = pts3dg.contiguous()
        pts3dg = torch.autograd.Variable(pts3dg.cuda(async=True),requires_grad=False) 

        R1 = torch.autograd.Variable(R1.cuda(async=True),requires_grad=False)
        R2 = torch.autograd.Variable(R2.cuda(async=True),requires_grad=False)
        R3 = torch.autograd.Variable(R3.cuda(async=True),requires_grad=False)
        R4 = torch.autograd.Variable(R4.cuda(async=True),requires_grad=False)

        V1 = torch.autograd.Variable(V1.cuda(async=True),requires_grad=False)
        V2 = torch.autograd.Variable(V2.cuda(async=True),requires_grad=False)
        V3 = torch.autograd.Variable(V3.cuda(async=True),requires_grad=False)
        V4 = torch.autograd.Variable(V4.cuda(async=True),requires_grad=False)

        # output 
        output = net(pts2d1, pts2d2, pts2d3, pts2d4, Mean_Delta, SD_Delta, R1, R2, R3, R4, V1, V2, V3, V4)     

        #loss   
        loss =   (output - pts3dg)**2 
        loss = loss.sum() / loss.numel()

        #3d body pose error calculation
        pred_pts = output.cuda().data 
        s3d = pts3dg.cuda().data 


        dist_error = PoseError.Dist_Err(pred_pts, s3d.cuda())
        DIST_ERROR = DIST_ERROR + dist_error

        #measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # print log
        losses.update(loss.data[0])
        dist_errors.update(dist_error)
        loss_dict = OrderedDict( [('loss', losses.avg),
                                  ('dist_error', dist_errors.avg)] )

        print('Epoch:[{0}][{1}/{2}] '
              'dist_error:[{3:.4f}] '.format(
               epoch, i, len(val_loader), dist_error))

        #Adding root position back
        root = root.unsqueeze(1).expand(root.size(0), pred_pts.size(1), root.size(1))
        pred_pts = pred_pts + root.cuda()

        #saving results
        for n in range(output.size(0)):
            token = name[n].split('/')
            save_path = '/home/rahil/PoseEstimator/Results/' + token[5] + '/' + token[7] + '/'
            PXIO.CreateFolderIfNotExist(save_path)
            save_name = save_path + token[8] 
            np.savetxt(save_name, pred_pts[n].cpu().numpy(), fmt='%.3f')

      
    DIST_ERROR = DIST_ERROR/(len(val_loader))
    print('the average dist_error is')
    print(DIST_ERROR)

    return losses.avg, dist_errors.avg

def demo(demo_loader, net, Mean_2D, Mean_Delta, Mean_3D, SD_2D, SD_Delta, SD_3D, epoch, opt):

    # switch to evaluate mode
    net.eval()

    fig = plt.figure()
    gs1 = gridspec.GridSpec(4,12)
    gs1.update(wspace=0.3, hspace=0.05) # set the spacing between axes.
    figuretitles = '   '

    for i, (pts3dg, root, pts2d1, pts2d2, pts2d3, pts2d4, R1, V1, R2, V2, R3, V3, R4, V4, name) in enumerate(demo_loader):
        
        token = name[0].split('/')
        figuretitle = token[6]+'_'+token[9]+'_frame'+token[10][:-4]+'      '
        figuretitles = figuretitles + figuretitle

        #plotting 2d poses
        ax = plt.subplot(gs1[2*i])
        ax.set_title(figuretitle, y = 1.2, loc='left', fontsize=14)
        ax.set_ylabel('view 1')
        Plot = Plot2d(pts2d1.cpu().numpy(), ax)
        ax = plt.subplot(gs1[2*i+12])
        ax.set_ylabel('view 2')
        Plot = Plot2d(pts2d2.cpu().numpy(), ax)
        ax = plt.subplot(gs1[2*i+24])
        ax.set_ylabel('view 3')
        Plot = Plot2d(pts2d3.cpu().numpy(), ax)
        ax = plt.subplot(gs1[2*i+36])
        ax.set_ylabel('view 4')
        Plot = Plot2d(pts2d4.cpu().numpy(), ax)

        #plotting groundtruth 3d pose
        ax = plt.subplot(gs1[0:2, 2*i+1], projection='3d')
        ax.set_title("3D GroundTruth", fontsize=10)
        pts3dg = pts3dg.narrow(1,1,16) #remove pelvis center
        Plot3d(pts3dg.cpu().numpy(), ax, ['c','m','y']) 

        # input and groundtruth
        pts2d1 = (pts2d1 - Mean_2D)/SD_2D 
        pts2d1 = torch.autograd.Variable(pts2d1.cuda(async=True),requires_grad=False) 

        pts2d2 = (pts2d2 - Mean_2D)/SD_2D 
        pts2d2 = torch.autograd.Variable(pts2d2.cuda(async=True),requires_grad=False) 

        pts2d3 = (pts2d3 - Mean_2D)/SD_2D 
        pts2d3 = torch.autograd.Variable(pts2d3.cuda(async=True),requires_grad=False) 

        pts2d4 = (pts2d4 - Mean_2D)/SD_2D 
        pts2d4 = torch.autograd.Variable(pts2d4.cuda(async=True),requires_grad=False) 

        pts3dg = pts3dg.contiguous()
        pts3dg = torch.autograd.Variable(pts3dg.cuda(async=True),requires_grad=False) 
        ###target_var = (pts3dg - Mean_3D)/SD_3D 

        R1 = torch.autograd.Variable(R1.cuda(async=True),requires_grad=False)
        R2 = torch.autograd.Variable(R2.cuda(async=True),requires_grad=False)
        R3 = torch.autograd.Variable(R3.cuda(async=True),requires_grad=False)
        R4 = torch.autograd.Variable(R4.cuda(async=True),requires_grad=False)

        V1 = torch.autograd.Variable(V1.cuda(async=True),requires_grad=False)
        V2 = torch.autograd.Variable(V2.cuda(async=True),requires_grad=False)
        V3 = torch.autograd.Variable(V3.cuda(async=True),requires_grad=False)
        V4 = torch.autograd.Variable(V4.cuda(async=True),requires_grad=False)

        # output 
        output = net(pts2d1, pts2d2, pts2d3, pts2d4, Mean_Delta, SD_Delta, R1, R2, R3, R4, V1, V2, V3, V4)     

        #Adding root position back
        pred_pts = output.cuda().data 
        root = root.unsqueeze(1).expand(root.size(0), pred_pts.size(1), root.size(1))
        pred_pts = output.cuda().data  + root.cuda()

        #plotting predicted 3d pose
        ax = plt.subplot(gs1[2:4, 2*i+1], projection='3d')
        ax.set_title("3D Predicted", fontsize=10)
        Plot3d(pred_pts.cpu().numpy(), ax, ['b','r','g']) 
    
    plt.show()

    return 

def adjust_learning_rate(optimizer, epoch, lr0):
    lr = lr0 * (0.96 ** (epoch // 4))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    print('Epoch:[%d]\tlr:[%f]' % (epoch, lr))


if __name__ == '__main__':
    main()


