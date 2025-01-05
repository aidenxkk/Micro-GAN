# Imports

import numpy as np
from numpy import random
import ValueWrapper.py
from matplotlib import pyplot as plt
%matplotlib inline

# Drawing function

def view_samples(samples, m, n):
    fig, axes = plt.subplots(figsize=(10, 10), nrows=m, ncols=n, sharey=True, sharex=True)
    for ax, img in zip(axes.flatten(), samples):
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        im = ax.imshow(1-img.reshape((2,2)), cmap='Greys_r')  
    return fig, axes
# Examples of faces
faces = [np.array([1,0,0,1]),
         np.array([0.9,0.1,0.2,0.8]),
         np.array([0.9,0.2,0.1,0.8]),
         np.array([0.8,0.1,0.2,0.9]),
         np.array([0.8,0.2,0.1,0.9])]
    
_ = view_samples(faces, 1, 5)
# Examples of noisy images
noise = [np.random.randn(2,2) for i in range(2)]
def generate_random_image():
    return [np.random.random(), np.random.random(), np.random.random(), np.random.random()]
_ = view_samples(noise, 1,2)
### GAN training without batching ####
import numpy as np

np.random.seed(42)
random.seed(42)

number_of_real_faces = 1000
def generate_real_face():
    # Generate the first value in the range [0.75, 1]
    first_value = np.random.uniform(0.75, 1)
    
    # Generate the second and third values in the range [0, 0.3]
    middle_values = np.random.uniform(0, 0.3, 2)
    
    # Generate the last value in the range [0.75, 1]
    last_value = np.random.uniform(0.75, 1)
    
    # Combine all values into a single array
    return np.array([first_value, *middle_values, last_value])

# Generate real faces
faces = [generate_real_face() for _ in range(number_of_real_faces)]  


lr = 0.01
epochs = 5000

D = MLP(4,[4,1])
G = MLP(4,[4,4])

d_errors = []
g_errors = []

def optimize_zero_grad(nn,learning_rate):
    for p in nn.parameters():
        p.data -= learning_rate * p.grad
        p.grad = 0
        
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

  ##### batch processing based on above. ###############

import numpy as np

np.random.seed(42)
random.seed(42)

number_of_real_faces = 1000

def generate_real_face():
    # Generate the first and last value in the range [0.75, 1]
    first_value = np.random.uniform(0.75, 1)
    last_value = np.random.uniform(0.75, 1)

    # Generate the second and third values in the range [0, 0.3]
    middle_values = np.random.uniform(0, 0.3, 2)
        
    # Combine all values into a single array
    return np.array([first_value, *middle_values, last_value])

# Generate real faces
faces = [generate_real_face() for _ in range(number_of_real_faces)]  

lr = 0.01
epochs = 500
batch_size = 32

D = MLP(4,[4,1])
G = MLP(4,[4,4])

d_errors = []
g_errors = []

def optimize_zero_grad(nn,learning_rate):
    for p in nn.parameters():
        p.data -= learning_rate * p.grad
        p.grad = 0

# GAN training
        
for epoch in range(epochs):
    
    # Train Discriminator
    # Fetch a batch of images of "real faces"
    batch_of_faces = [faces[random.randint(0, number_of_real_faces - 1)] for _ in range(batch_size)]
                      
    real_pedictions = [D(x) for x in batch_of_faces]
                      
    noises = [[Value(np.random.normal(0, 1)) for _ in range(4)] for _ in range(batch_size)]

    fake_faces = [G(z) for z in noises]
    
    fake_predictions = [ D(face) for face in fake_faces]

    # Discriminator loss function 
    d_loss = sum (
        - real_p.log() - (1 - fake_p).log() 
        for real_p, fake_p in zip(real_predictions, fake_predictions)
    )/len(real_predictions)
   
    d_errors.append(d_loss)
    
    d_loss.backward()

    optimize_zero_grad(D,lr)
    
    # Train Generator 

    noises = [[Value(np.random.normal(0, 1)) for _ in range(4)] for _ in range(batch_size)]

    fake_face = [G(z) for z in noises]
    
    fake_predictions = [D(face) for face in fake_faces]

    # Generator loss function
    g_loss = sum(- fake_p.log() for fake_p in fake_predictions) / len(fake_predictions)
    
    g_errors.append(g_loss)

    g_loss.backward()

    optimize_zero_grad(G,lr)
    
    # Adjust learning rate (simple linear decay)
    lr = lr * (1 - epoch / epochs)

    if epoch % 100 == 0:
        print(f"Epoch {epoch}: D Loss: {d_loss.data}, G Loss: {g_loss.data}")
draw_dot(d_loss)
draw_dot(g_loss)
