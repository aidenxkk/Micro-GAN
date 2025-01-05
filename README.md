# machinelearning
## Andrej Karpathy's Micrograd Background: 
A minimalist educational project designed to demonstrate the core concepts of automatic differentiation and backpropagation used in training neural networks. 
## MicroGAN: A GAN framework for education purpose 
Based on the basic structure of MLP in Micrograd, this project implements MicroGAN, a simplified Generative Adversarial Network (GAN) framework designed for experimentation and learning. 
### Intro
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

To use the code, simply clone the repository:



