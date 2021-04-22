import cv2
import argparse
import os
import PerceptualSimilarity.models as models
from util import util
from skimage.measure import compare_psnr,compare_ssim
from tqdm import tqdm

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-d0','--dir0', type=str, )
parser.add_argument('-d1','--dir1', type=str, )
parser.add_argument('--use_gpu', action='store_true', help='turn on flag to use GPU')

opt = parser.parse_args()

## Initializing the model
model = models.PerceptualLoss(model='net-lin',net='alex',use_gpu=opt.use_gpu)

# crawl directories
files = os.listdir(opt.dir0)

total_dist = 0
total_psnr = 0
total_ssim = 0
count =0

for file in tqdm(files):
	file1 = file[:4] + '.jpg'
	if(os.path.exists(os.path.join(opt.dir1,file1))):
		# Load images
		img0 = util.im2tensor(util.load_image(os.path.join(opt.dir0,file))) # RGB image from [-1,1]
		img1 = util.im2tensor(util.load_image(os.path.join(opt.dir1,file1)))

		if(opt.use_gpu):
			img0 = img0.cuda()
			img1 = img1.cuda()

		# Compute distance
		dist01 = model.forward(img0,img1)
		total_dist += dist01.item()

		I0 = cv2.imread(os.path.join(opt.dir0,file))
		I1 = cv2.imread(os.path.join(opt.dir1,file1))
		total_psnr += compare_psnr(I0,I1)
		total_ssim += compare_ssim(I0,I1,multichannel=True)
		count +=1

print ('Avg LPIPS: ', total_dist/len(files))
print ('Avg PSNR: ', total_psnr/len(files))
print ('Avg SSIM: ', total_ssim/len(files))
print ('Total files: ', count)
