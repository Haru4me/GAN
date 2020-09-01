import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
import imageio
import os

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets,transforms

from models import Generator


def img_samples(G, epoch, device, test=False):

	if test == False:

		dirr = "train_imgs"

	else:

		dirr = "test_imgs"

	unloader = transforms.ToPILImage()
	noise = torch.FloatTensor(np.random.normal(0, 1, (batch_size, 128))).to(device)
	plt.figure(figsize=(10,10))
	new_imgs = G(noise).cpu()
    
	for i in range(36):
		plt.subplot(6,6,i+1)
		img = unloader(new_imgs.data[i])
		plt.imshow(img, cmap='gist_gray')
		plt.axis('off')

	plt.savefig("./{0}/img_{1}.png".format(dirr,epoch))
	plt.close()


if __name__ == "__main__":


	batch_size=128
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	path = './saved_model/G_ab_100.pth'
	G = torch.load(path,map_location=torch.device('cpu'))

	for i in tqdm(range(20), desc="Test"):

		img_samples(G, i, device, test=True)


	images = []
	for i in range(20):
		images.append(imageio.imread("./test_imgs/img_%i.png"%i))

	imageio.mimsave("./test_imgs/MNIST_gen.gif", images)
	


