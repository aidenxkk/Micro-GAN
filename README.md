# machinelearning
## Andrej Karpathy's Micrograd Background: 
A minimalist educational project designed to demonstrate the core concepts of automatic differentiation and backpropagation used in training neural networks. 
## MicroGAN: A GAN framework for education purpose 
Based on the basic structure of MLP in Micrograd, this project implements MicroGAN, a simplified Generative Adversarial Network (GAN) framework designed for experimentation and learning. 
## Intro
Generator (G): Generates "fake" data samples from random noise.
Discriminator (D): Distinguishes between "real" and "fake" data.
Adversarial Training: Trains G to produce data that fools D, while training D to correctly classify real and fake data. 

## Features

- Custom `Value` class for autograd, which tracks operations and computes gradients using the chain rule.
- Support for operations like addition, multiplication, power, ReLU, Sigmoid, Tanh, and more.
- Backpropagation implementation to calculate gradients with respect to the loss.
- GAN training with both generator and discriminator.
- Batch processing for training the GAN.
- Visualization tools to view training errors and generated images.

## Required Python packages
install numpy package: 
`pip install numpy` 
install matplotlib package: 
```
python -m pip install -U pip
python -m pip install -U matplotlib
```
## Create ValueWrapper 

Based on karpath's micrograd, below is a slightly contrived example showing a number of possible supported operations:

```python
from micrograd.engine import Value

a = Value(-4.0)
b = Value(2.0)
c = a + b
d = a * b + b**3
c += c + 1
c += 1 + c + (-a)
d += d * 2 + (b + a).relu()
d += 3 * d + (b - a).relu()
e = c - d
f = e**2
g = f / 2.0
g += 10.0 / f
print(f'{g.data:.4f}') # prints 24.7041, the outcome of this forward pass
g.backward()
print(f'{a.grad:.4f}') # prints 138.8338, i.e. the numerical value of dg/da
print(f'{b.grad:.4f}') # prints 645.5773, i.e. the numerical value of dg/db
```


## Test ValueWrapper 
here is a small test for ValueWrapper: 
```
def test_sanity_check():

    x = Value(-4.0)
    z = 2 * x + 2 + x
    q = z.relu() + z * x
    h = (z * z).relu()
    y = h + q + q * x
    y.backward()
    xmg, ymg = x, y

    x = torch.Tensor([-4.0]).double()
    x.requires_grad = True
    z = 2 * x + 2 + x
    q = z.relu() + z * x
    h = (z * z).relu()
    y = h + q + q * x
    y.backward()
    xpt, ypt = x, y

    # forward pass went well
    assert ymg.data == ypt.data.item()
    # backward pass went well
    assert xmg.grad == xpt.grad.item()

def test_more_ops():

    a = Value(-4.0)
    b = Value(2.0)
    c = a + b
    d = a * b + b**3
    c += c + 1
    c += 1 + c + (-a)
    d += d * 2 + (b + a).relu()
    d += 3 * d + (b - a).relu()
    e = c - d
    f = e**2
    g = f / 2.0
    g += 10.0 / f
    g.backward()
    amg, bmg, gmg = a, b, g

    a = torch.Tensor([-4.0]).double()
    b = torch.Tensor([2.0]).double()
    a.requires_grad = True
    b.requires_grad = True
    c = a + b
    d = a * b + b**3
    c = c + c + 1
    c = c + 1 + c + (-a)
    d = d + d * 2 + (b + a).relu()
    d = d + 3 * d + (b - a).relu()
    e = c - d
    f = e**2
    g = f / 2.0
    g = g + 10.0 / f
    g.backward()
    apt, bpt, gpt = a, b, g

    tol = 1e-6
    # forward pass went well
    assert abs(gmg.data - gpt.data.item()) < tol
    # backward pass went well
    assert abs(amg.grad - apt.grad.item()) < tol
    assert abs(bmg.grad - bpt.grad.item()) < tol

test_sanity_check()
test_more_ops()
```

## Train GAN 
### Imports 
```
import numpy as np
from numpy import random
import ValueWrapper.py
from matplotlib import pyplot as plt
%matplotlib inline
```
### Basic methods needed before training
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

## Core Training below: 
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

    if epoch % 100 == 0:
        print(f"Epoch {epoch}: D Loss: {d_loss.data}, G Loss: {g_loss.data}")
```
