import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import *
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import time
import copy
import os

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#Hyperparameters
num_epochs=4 #increase this for better accuracy
batch_size=4
learning_rate=0.001

#CIFAR-10 dataset

#initially images are PILimage in range [0,1]->then converted to tensors of normalized range [-1,1]
transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])

train_data=torchvision.datasets.CIFAR10(root='./data',train=True,transform=transform,download=True)

test_data=torchvision.datasets.CIFAR10(root='./data',train=False,transform=transform)

train_loader=torch.utils.data.DataLoader(dataset=train_data,batch_size=batch_size,shuffle=True)

test_loader=torch.utils.data.DataLoader(dataset=test_data,batch_size=batch_size,shuffle=False)

classes=('airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck')

def imshow(inp, title=None):
    
    inp = inp.cpu() if device else inp
    inp = inp.numpy().transpose((1, 2, 0))
    
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    #plt.pause(5)
    
images, labels = next(iter(train_loader)) 
out = torchvision.utils.make_grid(images)

imshow(out, title=[train_data.classes[x] for x in labels])


#model->Transfer Learning
model=models.resnet18(weights='ResNet18_Weights.DEFAULT')
n_features=model.fc.in_features # last fully connected layer
model.fc=nn.Linear(n_features,10)
model.to(device)
#print(model)

loss=nn.CrossEntropyLoss() 
optimizer=torch.optim.SGD(model.parameters(),lr=learning_rate)#Stochastic Gradient Descent

print(1)
#training
num_steps=len(train_loader)
for epoch in range(num_epochs):
	for i,(image,label) in enumerate(train_loader):
		image=image.to(device)
		label=label.to(device)

		#forward pass
		output=model(image)
		l=loss(output,label)
		print(1)
		#backward pass
		optimizer.zero_grad()
		l.backward()

		#update parameters
		optimizer.step()

		if (i+1)%2000==0:
			print(f'epoch {epoch+1}/{num_epochs}, step {i+1}/{num_steps}, loss={l.item():.4f}')

#testing
with torch.no_grad(): #because gradient calculation or backward propagation is not needed in testing
	n_correct,n_samples=0,0
	n_class_correct=[0 for i in range(10)]
	n_class_samples=[0 for i in range(10)]
	for image,label in test_loader:
		image=image.to(device)
		label=label.to(device)
		output=model(image)

		_,pred=torch.max(output,1) #to get class labels with max probability, max returns (value,index)
		n_samples+=label.size(0)
		n_correct+=(pred==label).sum().item()

		for i in range(batch_size):
			lab=label[i]
			predicted=pred[i]
			if (lab==predicted):
				n_class_correct[lab]+=1
			n_class_samples[lab]+=1

	accuracy=100.0*n_correct/n_samples
	print(f'Accuracy of the model={accuracy}%')

	for i in range(10):
		acc=100.0*n_class_correct[i]/n_class_samples[i]
		print(f'Accuracy of {classes[i]}={acc}%')

