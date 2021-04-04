import torch
import torch.nn as nn
import argparse
import os.path as osp
import os
from eval.evaluator import Eval_thread
from eval.dataloader import EvalDataset

from config import davis_path, fbms_path, mcl_path, uvsd_path, \
    visal_path, vos_path, segtrack_path, davsod_path


# from concurrent.futures import ThreadPoolExecutor

if __name__ == "__main__":
    dataset_names = ['davis']
    snapshot = '192000'
    ckpt_path = '/home/amax/code/SVS/ckpt'
    exp_name = 'VideoSaliency_2021-04-03 11:31:50'
    gt_root = {'davis': os.path.join(davis_path, 'GT'),
               'DAVSOD': os.path.join(davsod_path, 'GT'),
               'VOS': os.path.join(vos_path, 'GT'),
               'ViSal': os.path.join(visal_path, 'GT'),
               'SegTrack-V2': os.path.join(segtrack_path, 'GT')}
    threads = []
    for dataset in dataset_names:
        save_path = os.path.join(ckpt_path, exp_name, '(%s) %s_%s' % (exp_name, dataset, snapshot))
        # save_path = 'TENet/DAVIS-MGA'
        loader = EvalDataset(osp.join(save_path), osp.join(gt_root[dataset]))
        thread = Eval_thread(loader, 'ours', dataset, './', True)
        threads.append(thread)
    for thread in threads:
        print(thread.run())
