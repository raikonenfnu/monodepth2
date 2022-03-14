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
import cv2
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
    parser.add_argument("--no_cuda",
                        help='if set, disables CUDA',
                        action='store_true')
    parser.add_argument("--pred_metric_depth",
                        help='if set, predicts metric depth instead of disparity. (This only '
                             'makes sense for stereo-trained KITTI models).',
                        action='store_true')

    return parser.parse_args()


def draw_output(depth_map, max_percent=95, min_percent=0):
    if depth_map is None:
        return None

    vmax = np.percentile(depth_map, max_percent)
    vmin = np.percentile(depth_map, min_percent)
    normalizer = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    mapper = mpl.cm.ScalarMappable(norm=normalizer, cmap='viridis')

    colormapped_im = (mapper.to_rgba(
        depth_map)[:, :, :3][:, :, ::-1] * 255).astype(np.uint8)

    return colormapped_im[:, :, ::-1]

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
    cap = cv2.VideoCapture(0) #rb5 ip & port (same from command)
    vis = o3d.visualization.Visualizer()
    vis.create_window()

    # geometry is the point cloud used in your animaiton
    pcd = o3d.geometry.PointCloud()
    origin_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=3, origin=[0, 0, 0])
    vis.add_geometry(pcd)
    vis.add_geometry(origin_frame)

    with torch.no_grad():
        while(True):
            ret, original_frame = cap.read()
            original_width, original_height,_ = original_frame.shape
            resized_image = cv2.resize(original_frame, dsize=(feed_width, feed_height))

            # Load image and preprocess
            # input_image = input_image.resize((feed_width, feed_height), pil.LANCZOS)
            input_image = transforms.ToTensor()(resized_image).unsqueeze(0)

            # PREDICTION
            input_image_device = input_image.to(device)
            features = encoder(input_image_device)
            outputs = depth_decoder(features)

            disp = outputs[("disp", 0)]
            disp_resized = torch.nn.functional.interpolate(
                disp, (original_height, original_width), mode="bilinear", align_corners=False)

            scaled_disp, depth = disp_to_depth(disp, 0.1, 100)
            metric_depth = STEREO_SCALE_FACTOR * depth.cpu().numpy()

            # Pass xyz to Open3D.o3d.geometry.PointCloud and visualize
            points = Generate3dGrid(metric_depth)
            colors = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
            colors = colors/255.0
            colors = colors.reshape(-1,3)

            points, colors = RemoveBackgroundManual(points, colors, background_thresh=10)
            points, colors = RemoveGroundManual(points, colors)

            # OG Pointcloud
            pcd.points = o3d.utility.Vector3dVector(points)
            pcd.colors = o3d.utility.Vector3dVector(colors)
            vis.update_geometry(pcd)
            vis.poll_events()
            vis.update_renderer()

            # Saving colormapped depth image
            disp_resized_np = disp_resized.squeeze().cpu().numpy()
            colormapped_im = draw_output(disp_resized_np)
            cv2.imshow("camera",np.asarray(original_frame))
            cv2.imshow("depth",colormapped_im)
            c = cv2.waitKey(1)
            if c == 27:
                break

    cap.release()
    cv2.destroyAllWindows()
    vis.destroy_window()
    print('-> Done!')


if __name__ == '__main__':
    args = parse_args()
    test_simple(args)
