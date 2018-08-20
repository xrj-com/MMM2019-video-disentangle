import os
import glob
import shutil
import torch
import numpy as np


def save_checkpoint(state, is_best, filename='checkpoint',  experiment_dir='', id = None):
    if id is not None:
        best_model = 'model_best{}.pth.tar'.format(id)
        chkpt_name = filename + '_{}.pth.tar'.format(id)
    else:
        best_model = 'model_best.pth.tar'
        chkpt_name = filename + '.pth.tar'

    best_model_filename = os.path.join(experiment_dir, best_model)
    model_filename = os.path.join(experiment_dir, chkpt_name)
 
    torch.save(state, model_filename)
    if is_best:
        shutil.copyfile(model_filename, best_model_filename)

def backup_src(src_root, target_root, reg='*.py'):
    print('BackUping the src')
    file_names = glob.glob(os.path.join(src_root, reg))
    if not os.path.exists(target_root):
        os.makedirs(target_root)
    for file in file_names:
        source = os.path.join(src_root, file)
        target = os.path.join(target_root, file)
        print('\tCopy ' + source + ' to ' + target)
        shutil.copyfile(os.path.join(src_root, file), os.path.join(target_root, file))
    print('Finish!')



if __name__ == "__main__":
    src_root = './'
    backup_src(src_root, None)