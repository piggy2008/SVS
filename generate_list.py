import os
from PIL import Image
path = '/data/ty/DAVSOD/flow'
file = open('/data/ty/Pre-train/pretrain_all_seq_DAVSOD_flow.txt', 'w')

def validata_path():
    folders = os.listdir(path)
    folders.sort()
    for folder in folders:
        imgs = os.listdir(os.path.join(path, folder))
        imgs.sort()
        for img in imgs:
            pre, name = img.split('_')
            os.rename(os.path.join(path, folder, img), os.path.join(path, folder, name))
            print(name)

def generate_flow_list():
    folders = os.listdir(path)
    # folders2 = os.listdir(path + '/flow')
    folders.sort()
    for folder in folders:
        imgs = os.listdir(os.path.join(path, folder, 'Imgs'))
        imgs.sort()
        imgs = [os.path.splitext(img)[0] for img in imgs]
        imgs2 = os.listdir(os.path.join(path, folder, 'flow'))
        imgs2.sort()
        imgs2 = [os.path.splitext(img)[0] for img in imgs2]
        for img in imgs:
            if img in imgs2:
                img_path = os.path.join('DAVSOD', folder, 'Imgs', img + '.png')
                gt_path = os.path.join('DAVSOD', folder, 'GT_object_level', img + '.png')
                flow_path = os.path.join('DAVSOD', folder, 'flow', img + '.jpg')
                file.writelines(img_path + ' ' + flow_path + ' ' + gt_path + '\n')

    file.close()

if __name__ == '__main__':
    validata_path()