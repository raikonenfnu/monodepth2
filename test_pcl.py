# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import os
import sys
import glob
import argparse
import numpy as np
import PIL.Image as pil
import matplotlib as mpl
import matplotlib.cm as cm
from GeometryExtractor import segmentor
import open3d as o3d

import torch
from torchvision import transforms, datasets

import networks
from layers import disp_to_depth
from utils import download_model_if_doesnt_exist
from evaluate_depth import STEREO_SCALE_FACTOR

def parse_args():
    parser = argparse.ArgumentParser(
        description='Simple testing funtion for Monodepthv2 models.')

    parser.add_argument('--image_path', type=str,
                        help='path to a test image or folder of images', required=True)
    parser.add_argument('--model_name', type=str,
                        help='name of a pretrained model to use',
                        choices=[
                            "mono_640x192",
                            "stereo_640x192",
                            "mono+stereo_640x192",
                            "mono_no_pt_640x192",
                            "stereo_no_pt_640x192",
                            "mono+stereo_no_pt_640x192",
                            "mono_1024x320",
                            "stereo_1024x320",
                            "mono+stereo_1024x320"])
    parser.add_argument('--ext', type=str,
                        help='image extension to search for in folder', default="jpg")
    parser.add_argument('--debug_pcl', type=bool,
                        help='Visualize removed pointcloud', default=False)
    parser.add_argument("--no_cuda",
                        help='if set, disables CUDA',
                        action='store_true')
    parser.add_argument("--pred_metric_depth",
                        help='if set, predicts metric depth instead of disparity. (This only '
                             'makes sense for stereo-trained KITTI models).',
                        action='store_true')

    return parser.parse_args()


def Generate3dGrid(depth_map):
    ''' Similarity Triangle
    We can obtain 3D pose through the ratio:
    x_pixel/x_real = fx/depth
    Camera Axes:
        x: to right
        y: to bottom
        z: to front
    '''
    # TODO: Calibrate camera to get proper intrinsicss
    # TODO: Get rid of floor planes
    # TODO: Get 2d GRID MAP
    z_real = np.squeeze(depth_map)
    x_pixel = np.zeros_like(z_real)
    for i in range(x_pixel.shape[1]):
        x_pixel[:,i] = i
    y_pixel = np.zeros_like(z_real)
    for j in range(y_pixel.shape[0]):
        y_pixel[j,:] = j
    # From niantic/monodepth2 intrinsics
    fx = 0.58*640
    fy = 1.92*192
    cx = 0.5*640
    cy = 0.5*192
    # From calib.txt in KITTI
    # fx = 718.856
    # fy = 718.856
    # cx = 607.1928
    # cy = 185.2157
    x_real = np.squeeze((x_pixel-cx)*depth_map/fx)
    y_real = np.squeeze((y_pixel-cy)*depth_map/fy)
    return np.dstack([x_real, y_real, z_real]).reshape(-1,3)


def RemoveBackgroundManual(points, colors, background_thresh=20):
    crop_ind = np.argwhere(points[:,2] > background_thresh)
    points = np.delete(points,crop_ind,axis=0) #Remove Floor plane from PointCloud
    colors = np.delete(colors,crop_ind,axis=0) #Remove Floor plane from Color
    return points, colors

def RemoveGroundManual(points, colors):
    crop_ind = np.argwhere(np.abs(points[:,1]-1.5) < 0.5)
    points = np.delete(points,crop_ind,axis=0) #Remove Floor plane from PointCloud
    colors = np.delete(colors,crop_ind,axis=0) #Remove Floor plane from Color
    return points, colors

def RemoveGroundRansac(points, colors, seg, debug_pcl):
    # Visualize Deleted Points
    plane,floor_ind,_ = seg.RANSACLargestPlane(points,1000,30)
    for i in floor_ind:
        print(points[i,:])
    if debug_pcl:
        ground_points = np.squeeze(points[floor_ind,:])
        ground_colors = np.squeeze(colors[floor_ind,:])

    # Remove Ground
    # floor_ind = np.argwhere(np.abs(points[:,1]) < 0.5)
    points = np.delete(points,floor_ind,axis=0) #Remove Floor plane from PointCloud
    colors = np.delete(colors,floor_ind,axis=0) #Remove Floor plane from Color
    if debug_pcl:
        ground_pcd = o3d.geometry.PointCloud()
        origin_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=3, origin=[0, 0, 0])
        ground_pcd.points = o3d.utility.Vector3dVector(ground_points)
        ground_pcd.colors = o3d.utility.Vector3dVector(ground_colors)
        o3d.visualization.draw_geometries([ground_pcd, origin_frame])
    return points, colors


def GenerateOccupancyGrid(points, size_x=10, size_y=10, unit_size=1):
    grid_image = points[:,(0,2)]
    print(np.min(grid_image[:,0]),np.max(grid_image[:,0]))
    print(np.min(grid_image[:,1]),np.max(grid_image[:,1]))

def test_simple(args):
    """Function to predict for a single image or folder of images
    """
    assert args.model_name is not None, \
        "You must specify the --model_name parameter; see README.md for an example"

    if torch.cuda.is_available() and not args.no_cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    if args.pred_metric_depth and "stereo" not in args.model_name:
        print("Warning: The --pred_metric_depth flag only makes sense for stereo-trained KITTI "
              "models. For mono-trained models, output depths will not in metric space.")

    download_model_if_doesnt_exist(args.model_name)
    model_path = os.path.join("models", args.model_name)
    print("-> Loading model from ", model_path)
    encoder_path = os.path.join(model_path, "encoder.pth")
    depth_decoder_path = os.path.join(model_path, "depth.pth")

    # LOADING PRETRAINED MODEL
    print("   Loading pretrained encoder")
    encoder = networks.ResnetEncoder(18, False)
    loaded_dict_enc = torch.load(encoder_path, map_location=device)

    # extract the height and width of image that this model was trained with
    feed_height = loaded_dict_enc['height']
    feed_width = loaded_dict_enc['width']
    filtered_dict_enc = {k: v for k, v in loaded_dict_enc.items() if k in encoder.state_dict()}
    encoder.load_state_dict(filtered_dict_enc)
    encoder.to(device)
    encoder.eval()

    print("   Loading pretrained decoder")
    depth_decoder = networks.DepthDecoder(
        num_ch_enc=encoder.num_ch_enc, scales=range(4))

    loaded_dict = torch.load(depth_decoder_path, map_location=device)
    depth_decoder.load_state_dict(loaded_dict)

    depth_decoder.to(device)
    depth_decoder.eval()

    # FINDING INPUT IMAGES
    if os.path.isfile(args.image_path):
        # Only testing on a single image
        paths = [args.image_path]
        output_directory = os.path.dirname(args.image_path)
    elif os.path.isdir(args.image_path):
        # Searching folder for images
        paths = glob.glob(os.path.join(args.image_path, '*.{}'.format(args.ext)))
        output_directory = args.image_path
    else:
        raise Exception("Can not find args.image_path: {}".format(args.image_path))

    print("-> Predicting on {:d} test images".format(len(paths)))

    # PREDICTING ON EACH IMAGE IN TURN
    seg = segmentor()
    with torch.no_grad():
        for idx, image_path in enumerate(paths):

            if image_path.endswith("_disp.jpg"):
                # don't try to predict disparity for a disparity image!
                continue

            # Load image and preprocess
            original_image = pil.open(image_path).convert('RGB')
            original_width, original_height = original_image.size
            color_image = original_image.resize((feed_width, feed_height), pil.LANCZOS)
            input_image = transforms.ToTensor()(color_image).unsqueeze(0)

            # PREDICTION
            input_image = input_image.to(device)
            features = encoder(input_image)
            outputs = depth_decoder(features)

            disp = outputs[("disp", 0)]
            disp_resized = torch.nn.functional.interpolate(
                disp, (original_height, original_width), mode="bilinear", align_corners=False)

            # Saving numpy file
            output_name = os.path.splitext(os.path.basename(image_path))[0]
            scaled_disp, depth = disp_to_depth(disp, 0.1, 100)
            metric_depth = STEREO_SCALE_FACTOR * depth.cpu().numpy()

            # Pass xyz to Open3D.o3d.geometry.PointCloud and visualize
            points = Generate3dGrid(metric_depth)
            origin_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=3, origin=[0, 0, 0])

            colors = np.squeeze(np.array(color_image))/255.0
            colors = colors.reshape(-1,3)

            # Remove Ground
            # points, colors = RemoveGroundRansac(points, colors, seg, args.debug_pcl)
            points, colors = RemoveBackgroundManual(points, colors, background_thresh=10)
            points, colors = RemoveGroundManual(points, colors)
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            pcd.colors = o3d.utility.Vector3dVector(colors)
            o3d.visualization.draw_geometries([origin_frame, pcd])

            # Occupancy Grid
            GenerateOccupancyGrid(points)

            # Saving colormapped depth image
            disp_resized_np = disp_resized.squeeze().cpu().numpy()
            vmax = np.percentile(disp_resized_np, 95)
            normalizer = mpl.colors.Normalize(vmin=disp_resized_np.min(), vmax=vmax)
            mapper = cm.ScalarMappable(norm=normalizer, cmap='magma')
            colormapped_im = (mapper.to_rgba(disp_resized_np)[:, :, :3] * 255).astype(np.uint8)
            im = pil.fromarray(colormapped_im)

            name_dest_im = os.path.join(output_directory, "{}_disp.jpeg".format(output_name))
            im.save(name_dest_im)

            print("   Processed {:d} of {:d} images - saved predictions to:".format(
                idx + 1, len(paths)))
            print("   - {}".format(name_dest_im))

    print('-> Done!')

if __name__ == '__main__':
    args = parse_args()
    test_simple(args)
