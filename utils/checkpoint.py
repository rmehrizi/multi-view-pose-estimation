# Rahil Mehrizi, Oct 2018
"""Saving and Loading Checkpoint"""

import os, shutil
import torch

class Checkpoint():
    def __init__(self, opt):
        self.opt = opt
        exp_dir = os.path.join(opt.exp_dir, opt.exp_id)
        if opt.resume_prefix != '':
            if 'pth' in opt.resume_prefix:
                trunc_index = opt.resume_prefix.index('pth')
                opt.resume_prefix = opt.resume_prefix[0:trunc_index - 1]
            self.save_path = os.path.join(exp_dir, opt.resume_prefix)
        else:
            self.save_path = exp_dir
        self.save_dir = os.path.join(opt.exp_dir, opt.exp_id)
   

    def save_checkpoint(self, net, train_history, name):
        lr_prefix = ('/lr-%.15f' % train_history.lr[-1]['lr']).rstrip('0').rstrip('.')
        save_path = self.save_path + lr_prefix + ('-%d.pth.tar' % train_history.epoch[-1]['epoch'])

        checkpoint = { 'train_history': train_history.state_dict(), 
                       'state_dict': net.state_dict()}
        torch.save( checkpoint, save_path )
        print("=> saving '{}'".format(save_path))
        if train_history.is_best:
            print("=> saving '{}'".format(self.save_dir +'/' + name))
            save_path2 = os.path.join(self.save_dir, name)
            shutil.copyfile(save_path, save_path2)

    def load_checkpoint(self, net, train_history, name):
        save_path = self.save_dir + name
        self.save_path = self.save_path 

        if os.path.isfile(save_path):
            print("=> loading checkpoint '{}'".format(save_path))
            checkpoint = torch.load(save_path)
            train_history.load_state_dict( checkpoint['train_history'] )
            new_params = net.state_dict()  #new layers
            old_params = checkpoint['state_dict'] #from checkpoint
            new_params.update(old_params)
            net.load_state_dict(new_params)
        else:
            print("=> no checkpoint found at '{}'".format(save_path))


  
