# Micro-GAN for educational purpose
## Andrej Karpathy's Micrograd: 
A minimalist educational project designed to demonstrate the core concepts of automatic differentiation and backpropagation used in training neural networks. 

## Intro
Building upon the foundation of Micrograd, Micro-GAN extends its functionality by adding activation functions like ReLU and Sigmoid. This allows us to improve the understanding of optimization in Generative Adversarial Networks (GANs) within a simplified educational framework. The project uses simple 2x2 images (faces) as learning target from a teaching video: A friendly introduction to Generative Adversarial Networks(Serrano.Academy), simplifying the training process.

## MicroGAN
Generator (G): Generates "fake" data samples from random noise.
Discriminator (D): Distinguishes between "real" and "fake" data.
Adversarial Training: Trains G to produce data that fools D, while training D to correctly classify real and fake data. 

## Required Python packages
install numpy package: 
`pip install numpy` 
install matplotlib package: 
```
python -m pip install -U pip
python -m pip install -U matplotlib
```
## Create ValueWrapper 

```
a = Value(-4.0)
b = Value(2.0)
c = a * b + b**2

print(f'{c.data:.4f}')  # Forward pass result
c.backward()

print(f'{a.grad:.4f}')  # Gradient with respect to a
print(f'{b.grad:.4f}')  # Gradient with respect to b

```

## Train GAN 
### Imports: 
numpy library for backpropagation and python ploting library for displaying training results
```
import numpy as np
from numpy import random
import ValueWrapper.py
from matplotlib import pyplot as plt
%matplotlib inline
```
### Basic Background before training
Based on the idea in this Introduction to GAN: Generative Adversarial Networks in Slanted Land
In this video, it define faces as the top-left and bottom-right corners of face images have hight(dark) values, while the other corners have low(light) values. To tell if an image is a face mathematically, add the pixel values of the top-left and bottom-right corners and subtract the values of the other two corners. A threshold of 1 is set:
Score ≥ 1 → classified as a face
Score < 1 → classified as not a face
https://github.com/luisguiserrano/gans/blob/master/README.md#:~:text=A%20Friendly%20Introduction%20to%20GANs
### functions as below 
```
def view_samples(samples, m, n):
```
This function visualizes a set of sample images using `matplotlib`
example: 
```
faces = [np.array([1, 0, 0, 1]), np.array([0.9, 0.1, 0.2, 0.8])]
view_samples(faces, 1, 2)
```
```
def generate_random_image():
```
This function generates a random image represented by a 1D array of 4 random values, each drawn from a uniform distribution between 0 and 1. The values simulate random pixel intensities or some form of image-like data.
```
def generate_real_face():
```
This function generates a "real" face as a 1D array of 4 values, which could represent some image-like data, such as pixel intensities or features. The values are sampled from a specific range:
The first and last values are in the range [0.75, 1].
The middle two values are in the range [0, 0.3]
```
def optimize_zero_grad(nn,learning_rate):
``` 
This function is used to update the parameters (weights) of a neural network nn using gradient descent. The gradients are applied to the parameters, and then the gradients are reset to zero. This helps in optimizing the neural network's weights during training.

## Core Training: 
### Discriminator Training: d_loss=−log(real_p)−log(1−fake_p)

### Generator Training: g_loss=−log(fake_p)
```
for epoch in range(epochs):

    real_p = D(faces[random.randint(0, number_of_real_faces - 1)])

    noise = [Value(np.random.normal(0, 1)) for _ in range(4)]

    fake_face = G(noise)
    
    fake_p = D(fake_face)

    d_loss = ( - real_p.log() - (1 - fake_p).log()) 
   

    d_errors.append(d_loss)
    
    d_loss.backward()

    optimize_zero_grad(D,lr)

    noise = [Value(np.random.normal(0, 1)) for _ in range(4)] 

    fake_face = G(noise)
    
    fake_p = D(fake_face)

    g_loss = -fake_p.log()
    
    g_errors.append(g_loss)

    g_loss.backward()

    optimize_zero_grad(G,lr)
    
    # Adjust learning rate (simple linear decay)
    lr = lr * (1 - epoch / epochs)
```
Example output: 
```Epoch 0: D Loss: 1.382, G Loss: 0.745
Epoch 100: D Loss: 0.675, G Loss: 1.045
...
```

    if epoch % 100 == 0:
        print(f"Epoch {epoch}: D Loss: {d_loss.data}, G Loss: {g_loss.data}")
```
