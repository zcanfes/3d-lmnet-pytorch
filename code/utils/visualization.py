import k3d
import numpy as np
import torch
import matplotlib.pyplot as plt

def visualize_pointcloud(points):
    try:
        points=points.cpu().numpy()
    catch:
        continue
    plt_points = k3d.points(positions=points.astype(np.float32) ,point_size=0.2, shader='3d',color=0x3f6bc5)
    plot = k3d.plot(grid=(-0.55, -0.55, -0.55, 0.55, 0.55, 0.55))
    plot += plt_points
    plot.display()
                        

def visualize_image(img):
    plt.imshow(img)
