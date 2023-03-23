import torch
import torch.nn as nn
import torch.nn.functional as F #for relu activation
import torchvision #for computer vision datasets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

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

#printing a sample
ex=next(iter(train_loader))
sample,label=ex
#print(sample.shape, label.shape) -> torch.Size([4, 3, 32, 32]) torch.Size([4]) -> 4 samples in a batch, 3 color channels, 32*32 image size, and 4 labels for each sample
#print(sample,label)

'''
#to view images
img = next(iter(train_loader))[0][0]
plt.imshow(transforms.ToPILImage()(img))
#plt.show()
#the result is different from the original images because we have applied normalization transformation
'''

class ConvNet(nn.Module):
	def __init__(self):
		super(ConvNet,self).__init__()
		self.conv1=nn.Conv2d(3,6,5) #paramters->input channels, output channels, filter size
		self.pool=nn.MaxPool2d(2,2)
		self.conv2=nn.Conv2d(6,16,5) #number of input channels in next layers should be equal to output channels of previous layers
		self.fc1=nn.Linear(16*5*5,120)
		self.fc2=nn.Linear(120,60)
		self.fc3=nn.Linear(60,10) #output=num_classes
	
	def forward(self,x):
		out=self.pool(F.relu(self.conv1(x)))
		out=self.pool(F.relu(self.conv2(out)))
		out=out.view(-1,16*5*5)
		out=F.relu(self.fc1(out))
		out=F.relu(self.fc2(out))
		out=self.fc3(out)
		#no softmax at the end because CrossEntropyLoss itself applies softmax
		return out

model=ConvNet().to(device)

loss=nn.CrossEntropyLoss() #This loss chosen because we have a case of multiple class image classification, it applies softmax itself at the end layer
optimizer=torch.optim.SGD(model.parameters(),lr=learning_rate) #Stochastic Gradient Descent

#training
num_steps=len(train_loader)
for epoch in range(num_epochs):
	for i,(image,label) in enumerate(train_loader):
		image=image.to(device)
		label=label.to(device)

		#forward pass
		output=model(image)
		#print(output.shape)->(batch_size,num_classes)
		l=loss(output,label)

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

