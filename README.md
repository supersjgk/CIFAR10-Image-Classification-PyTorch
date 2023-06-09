# CIFAR10-Image-Classification-PyTorch
CIFAR-10 dataset can be found here: https://www.cs.toronto.edu/~kriz/cifar.html

Convolutional Neural Networks:
The images in CIFAR-10 have 3 color channels: red,green,blue. So, an image is represented as a matrix of dimensions width*height*number_of_chanels. Images are of much larger size so we use Convolutional Neural Networks. They are capable of learning much better than simple Feed forward NNs and perform better classification. To learn about their working go to: https://towardsdatascience.com/a-comprehensive-guide-to-convolutional-neural-networks-the-eli5-way-3bd2b1164a53

The following model architecture has been used:

![1_vkQ0hXDaQv57sALXAJquxA](https://user-images.githubusercontent.com/75927878/227108440-14e2230c-fd1b-4865-a77f-bdf61de6ea75.jpg)

To get better accuracy, ResNet-18 model is used for transfer learning: Read about it here: https://machinelearningmastery.com/transfer-learning-for-deep-learning/

To compare the two architectures, I've run the model with just 4 epochs and the difference in accuracies is clear (NOTE: use a much larger number of epochs for practicality and better accuracy).

With simple CNN architecture:

![Screenshot (366)](https://user-images.githubusercontent.com/75927878/227425492-049428bc-cecf-4dd4-9b68-5c6388a2a573.png)

With Transfer Learning using ResNet-18 architecture:

![Screenshot (365)](https://user-images.githubusercontent.com/75927878/227425564-a0332df1-21c1-4b54-8534-c17bb48a223e.png)
