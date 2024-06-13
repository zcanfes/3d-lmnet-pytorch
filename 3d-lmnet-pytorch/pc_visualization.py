import os
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import yaml
import argparse

# Util function for loading point clouds|
import numpy as np

# Data structures and functions for rendering
from PIL import Image
from pytorch3d.structures import Pointclouds
from pytorch3d.vis.plotly_vis import AxisArgs, plot_batch_individually, plot_scene
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVOrthographicCameras, 
    PointsRasterizationSettings,
    PointsRenderer,
    PulsarPointsRenderer,
    PointsRasterizer,
    AlphaCompositor,
    NormWeightedCompositor,
    FoVPerspectiveCameras
)

# Setup
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
else:
    device = torch.device("cpu")

def load_config(config_path):
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file {config_path} not found.")
    
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


def visualize():
    # Load configuration file
    parser = argparse.ArgumentParser(description="Load configuration file")
    parser.add_argument('--config', type=str, required=True, help='Path to the configuration file')
    args = parser.parse_args()

    config = load_config(args.config)
    print(config)

    INPUT_IMG_PATH = config['input_img_path']
    GT_POINT_CLOUD_PATH = config['gt_point_cloud_path']
    PREDICTED_POINT_CLOUD_PATH = config['predicted_point_cloud_path']

    file_names = [GT_POINT_CLOUD_PATH, PREDICTED_POINT_CLOUD_PATH]

    # Initialize a camera.
    R, T = look_at_view_transform(20, 14, -46)
    cameras = FoVOrthographicCameras(device=device, R=R, T=T, znear=0.01)

    # Define the settings for rasterization and shading. Here we set the output image to be of size
    # 512x512. As we are rendering images for visualization purposes only we will set faces_per_pixel=1
    # and blur_radius=0.0. Refer to rasterize_points.py for explanations of these parameters. 
    raster_settings = PointsRasterizationSettings(
        image_size=512, 
        radius = 0.005,
        points_per_pixel = 10
    )

    # Create a points renderer by compositing points using an weighted compositor (3D points are
    # weighted according to their distance to a pixel and accumulated using a weighted sum)
    renderer = PointsRenderer(
        rasterizer=PointsRasterizer(cameras=cameras, raster_settings=raster_settings),
        compositor=NormWeightedCompositor(background_color=(255,255,255)),
    )

    if INPUT_IMG_PATH != '':
        # Load and save the input image
        image = np.load(f"{INPUT_IMG_PATH}.npy")
        im = Image.fromarray(image[0].transpose(1,2,0).astype(np.uint8))
        im.save(f"{INPUT_IMG_PATH}.png")


    # Load point clouds
    for name in file_names:
        input = np.load(f'{name}.npy')

        if input.shape[2]!=3:
            input=input.transpose(0,2,1)

        x = input[0,:, 0]
        y = input[0,:, 1]
        z = input[0,:, 2]

        pts = np.stack((y,-x,z), axis = 1) 
        verts = torch.Tensor(pts).to(device)
        # You can change the color of point clouds here
        rgb =  (torch.tensor([72.45,251.85,528]) * torch.ones(pts.shape) / 1000.).to(device)

        point_cloud = Pointclouds(points=[verts], features=[rgb])

        images = renderer(point_cloud)
        plt.figure(figsize=(9, 11))
        plt.imshow(images[0, ..., :3].cpu().numpy())
        plt.axis("off");
        plt.tight_layout()
        # Save rendered point cloud image
        plt.savefig(f'{name}.png', dpi=300)


if __name__ == "__main__":
    visualize()