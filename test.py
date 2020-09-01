import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
import imageio
import os

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets,transforms

from models import Discriminator, Generator


def img_samples(G,epoch , test=False):

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


def train(train_data, generator, discriminator, opt_g, opt_d, loss, num_epoch=100, k=1):

	print("Star training...")

	for epoch in tqdm(range(num_epoch), desc="Epoch"):
	    
		g_loss_run = []
		d_loss_run = []

		for batch,i in train_data:
	        
			real = batch.to(device)
			noise = torch.FloatTensor(np.random.normal(0, 1, (batch_size, 128))).to(device)
			gen = generator(noise)

			valid = torch.autograd.Variable(torch.Tensor(real.size(0), 1).fill_(1.0), requires_grad=False).to(device)
			fake = torch.autograd.Variable(torch.Tensor(real.size(0), 1).fill_(0.0), requires_grad=False).to(device)

			"""
				Train Discriminator
			"""
	        
			for _ in range(k):
	            
				discriminator.train()
				opt_d.zero_grad()
	            
				d_loss = loss(discriminator(real), valid) + loss(discriminator(gen.detach()), fake)
	            
				d_loss.backward()
				opt_d.step()
				d_loss_run.append(d_loss.item())
	            
			"""
				Train Generator
			"""

			generator.train()
			opt_g.zero_grad()
	        
			g_loss = loss(discriminator(gen),valid)
	        #print(g_loss.item(), d_loss.item())
	            
			g_loss.backward()
			opt_g.step()
			g_loss_run.append(g_loss.item())
	    
		img_samples(G,epoch)
	    
		if epoch % 10 == 0:

			print("D loss: {0}\t||\tG loss: {1}".format(np.mean(d_loss_run),np.mean(g_loss_run)))
			torch.save(generator, f'./saved_model/G_{epoch}.pth')
			torch.save(discriminator, f'./saved_model/D_{epoch}.pth')

	print("Finished!")


if __name__ == "__main__":

	#os.makedirs('./test_imgs')
	#os.makedirs('./train_imgs')

	batch_size=128
	transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
	dataload = DataLoader(datasets.MNIST("./", train=True, download=True, transform=transform),
							batch_size=batch_size,
							shuffle=True,
							drop_last=True)

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	G = Generator().to(device)
	D = Discriminator().to(device)

	optimizer_G = torch.optim.Adam(G.parameters(), lr=2e-4, betas=(0.5, 0.999))
	optimizer_D = torch.optim.Adam(D.parameters(), lr=2e-4, betas=(0.5, 0.999))

	loss = torch.nn.BCELoss().to(device)

	train(dataload, G, D, optimizer_G, optimizer_D, loss)

	images = []
	for i in tqdm(range(100), desc="Test"):
		images.append(imageio.imread("./test_imgs/img_%i.png"%i))

	imageio.mimsave("./test_imgs/MNIST_gen.gif", images)





