import matplotlib.pyplot as plt
import numpy as np


def dot_plot(x,y) : 
    # Create a new figure
    fig, ax = plt.subplots()

    # Plot the data as a scatter plot
    ax.scatter(x, y)

    # Set labels and title
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_title('Dot Plot of Array x')

    # Show the grid
    ax.grid(True)

    # Display the plot
    plt.show()

plt.plot([x.data for x in g_errors ] )
plt.title("Generator error function")
plt.legend("gen")
plt.show()
plt.plot([x.data for x in d_errors ] )
plt.legend('disc')
generated_images = []
for i in range(20):
    z = [np.random.normal(-1, 1) for _ in range(4)]
    generated_image = G(z)
    generated_images.append(generated_image)
images_render = [] 

for i in generated_images:
    images_render.append(np.array([x.data for x in i]))
        
_ = view_samples(images_render, 4, 5)
generated_image
plt.title("Discriminator error function")
