import torch
import torch.nn.functional as F
import sys
sys.path.append('./models')
import numpy as np
import os, argparse
import cv2
from Encoder import Mnet
from data import test_dataset
import time
from thop import profile
from torch import nn

parser = argparse.ArgumentParser()
parser.add_argument('--testsize', type=int, default=384, help='testing size')
parser.add_argument('--gpu_id', type=str, default='1', help='select gpu id')
parser.add_argument('--test_path', type=str, default='./dataset/test/WJ_838', help='test dataset path')
opt = parser.parse_args()

dataset_path = opt.test_path

#set device for test
os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id

#load the model
model = Mnet()

# model = nn.DataParallel(Mnet())
model.cuda()
model.eval()
fps = 0

# RGBT-Test
if __name__ == '__main__':
    for i in ['MAE',]:   #,'AVG',
        test_datasets = ['WJ_838']
        # test_datasets = ['RGBD_185','RGBT_156','V723_252','WJSX_800','WJ_838']
        for dataset in test_datasets:
            time_s = time.time()
            # model_path = os.path.join('./model/', 'Best_' + str(i) + '_epoch.pth') #Best_AVG_  ，  epoch_180   best_MAE_epoch
            model_path = os.path.join('./model/', 'Best_' + 'MAE' + '_epoch.pth')  # Best_AVG_  ，  epoch_180   best_MAE_epoch
            model.load_state_dict(torch.load(model_path))
            sal_save_path = os.path.join('./output/', dataset + '-' + str(i) + '/')
            if not os.path.exists(sal_save_path):
                os.makedirs(sal_save_path)
                # os.makedirs(edge_save_path)
            image_root = dataset_path +  '/input/'
            gt_root = dataset_path + '/target/'
            test_loader = test_dataset(image_root, gt_root, opt.testsize)
            nums = test_loader.size
            # r_input = torch.randn(1, 3, 256, 256).cuda().float()
            # flops, parameters = profile(model, (r_input,))
            # print("The number of flops: {}".format(flops))
            # print("The number of params: {}".format(parameters))
            w_dict = {}
            for i in range(test_loader.size):
                image, gt, name, img_for_post, image1 = test_loader.load_data()
                gt = np.asarray(gt, np.float32)
                gt /= (gt.max() + 1e-8)
                image = image.cuda()
                score, score1, score2, s_sig = model(image)
                res = F.interpolate(score, size=gt.shape, mode='bilinear', align_corners=False)
                res = res.sigmoid().data.cpu().numpy().squeeze()
                res = (res - res.min()) / (res.max() - res.min() + 1e-8)
                # img = np.asarray(image1, np.float32)
                # img /= (img.max() + 1e-8)
                # img = img.squeeze()
                # im_out = res
                # im_out[im_out >= 0.7] = 1
                # im_out[im_out != 1] = 0
                # img[im_out == 1] = 1

                cv2.imwrite(sal_save_path + name, res * 255)
            time_e = time.time()
            fps += (nums / (time_e - time_s))
            print("FPS:%f" % (nums / (time_e - time_s)))
            print('Test Done!')
        print("Total FPS %f" % fps) # this result include I/O cost

