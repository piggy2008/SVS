import os
import numpy as np
from PIL import Image
from misc import check_mkdir, AvgMeter, cal_precision_recall_mae, cal_fmeasure

ckpt_path = './ckpt'
exp_name = 'VideoSaliency_2020-11-28 11:12:53'
name = 'davis'
# root = '/home/qub/data/saliency/FBMS/FBMS_Testset2'
root = '/home/ty/data/davis/davis_test2'
# root = '/home/ty/data/VOS/VOS_test'
# root = '/home/ty/data/SegTrack-V2/SegTrackV2_test'
# root = '/home/ty/data/ViSal/ViSal_test'
# root = '/home/ty/data/MCL/MCL_test'
# root = '/home/ty/data/DAVSOD/DAVSOD_test'

gt_root = '/home/ty/data/davis/GT'
# gt_root = '/home/ty/data/VOS/GT'
# gt_root = '/home/qub/data/saliency/FBMS/GT'
# gt_root = '/home/ty/data/MCL/GT'
# gt_root = '/home/ty/data/ViSal/GT'
# gt_root = '/home/ty/data/DAVSOD/GT'
# gt_root = '/home/ty/data/SegTrack-V2/GT'
args = {
    'snapshot': '44000',  # your snapshot filename (exclude extension name)
    'crf_refine': False,  # whether to use crf to refine results
    'save_results': True  # whether to save the resulting masks
}

precision_record, recall_record, = [AvgMeter() for _ in range(256)], [AvgMeter() for _ in range(256)]
mae_record = AvgMeter()
results = {}

save_path = os.path.join(ckpt_path, exp_name, '(%s) %s_%s' % (exp_name, name, args['snapshot']))
folders = os.listdir(save_path)
folders.sort()
for folder in folders:
    imgs = os.listdir(os.path.join(save_path, folder))
    imgs.sort()

    for img in imgs:
        print(os.path.join(folder, img))
        if name == 'VOS' or name == 'DAVSOD':
            image = Image.open(os.path.join(root, folder, img[:-4] + '.png')).convert('RGB')
        else:
            image = Image.open(os.path.join(root, folder, img[:-4] + '.jpg')).convert('RGB')
        gt = np.array(Image.open(os.path.join(gt_root, folder, img)).convert('L'))
        pred = np.array(Image.open(os.path.join(save_path, folder, img)).convert('L'))
        # if args['crf_refine']:
        #     pred = crf_refine(np.array(image), pred)
        precision, recall, mae = cal_precision_recall_mae(pred, gt)

        for pidx, pdata in enumerate(zip(precision, recall)):
            p, r = pdata
            precision_record[pidx].update(p)
            recall_record[pidx].update(r)
        mae_record.update(mae)

fmeasure = cal_fmeasure([precord.avg for precord in precision_record],
                                    [rrecord.avg for rrecord in recall_record])

results[name] = {'fmeasure': fmeasure, 'mae': mae_record.avg}

print ('test results:')
print (results)

# VideoSaliency_2020-11-02 03:43:38 20000
# {'davis': {'fmeasure': 0.8996040940700104, 'mae': 0.01981267903497454}}

# VideoSaliency_2020-11-21 14:21:23 24000
# {'davis': {'fmeasure': 0.9031097105476225, 'mae': 0.018686559761604743}}
