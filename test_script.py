# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import os
import skimage.transform
import numpy as np
import PIL.Image as pil

from datasets.mono_dataset import MonoDataset
import time

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from dense_reprojection import *

import json

from utils import *
from kitti_utils import *
from layers import *

import datasets
import networks
from IPython import embed

class KITTIDataset(MonoDataset):
    """Superclass for different types of KITTI dataset loaders
    """
    def __init__(self, *args, **kwargs):
        super(KITTIDataset, self).__init__(*args, **kwargs)

        # NOTE: Make sure your intrinsics matrix is *normalized* by the original image size    
        self.K = np.array([[0.58, 0, 0.5, 0],
                           [0, 1.92, 0.5, 0],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]], dtype=np.float32)

        self.full_res_shape = (1242, 375)
        self.side_map = {"2": 2, "3": 3, "l": 2, "r": 3}

    def check_depth(self):
        line = self.filenames[0].split()
        scene_name = line[0]
        frame_index = int(line[1])

        velo_filename = os.path.join(
            self.data_path,
            scene_name,
            "velodyne_points/data/{:010d}.bin".format(int(frame_index)))

        return os.path.isfile(velo_filename)

    def get_color(self, folder, frame_index, side, do_flip):
        color = self.loader(self.get_image_path(folder, frame_index, side))

        if do_flip:
            color = color.transpose(pil.FLIP_LEFT_RIGHT)

        return color


class KITTIRAWDataset(KITTIDataset):
    """KITTI dataset which loads the original velodyne depth maps for ground truth
    """
    def __init__(self, *args, **kwargs):
        super(KITTIRAWDataset, self).__init__(*args, **kwargs)

    def get_image_path(self, folder, frame_index, side):
        f_str = "{:010d}{}".format(frame_index, self.img_ext)
        image_path = os.path.join(
            self.data_path, folder, "image_0{}/data".format(self.side_map[side]), f_str)
        return image_path

    def get_seg_path(self, folder, frame_index, side):
        f_str = "{:010d}{}".format(frame_index, self.img_ext)
        image_path = os.path.join(
            self.data_path, folder, "image_0{}/seg_label".format(self.side_map[side]), f_str)
        return image_path

    def get_depth(self, folder, frame_index, side, do_flip):
        calib_path = os.path.join(self.data_path, folder.split("/")[0])

        velo_filename = os.path.join(
            self.data_path,
            folder,
            "velodyne_points/data/{:010d}.bin".format(int(frame_index)))

        depth_gt = generate_depth_map(calib_path, velo_filename, self.side_map[side])
        depth_gt = skimage.transform.resize(
            depth_gt, self.full_res_shape[::-1], order=0, preserve_range=True, mode='constant')

        if do_flip:
            depth_gt = np.fliplr(depth_gt)

        return depth_gt


class KITTIOdomDataset(KITTIDataset):
    """KITTI dataset for odometry training and testing
    """
    def __init__(self, *args, **kwargs):
        super(KITTIOdomDataset, self).__init__(*args, **kwargs)

    def get_image_path(self, folder, frame_index, side):
        f_str = "{:06d}{}".format(frame_index, self.img_ext)
        image_path = os.path.join(
            self.data_path,
            "sequences/{:02d}".format(int(folder)),
            "image_{}".format(self.side_map[side]),
            f_str)
        return image_path


class KITTIDepthDataset(KITTIDataset):
    """KITTI dataset which uses the updated ground truth depth maps
    """
    def __init__(self, *args, **kwargs):
        super(KITTIDepthDataset, self).__init__(*args, **kwargs)

    def get_image_path(self, folder, frame_index, side):
        f_str = "{:010d}{}".format(frame_index, self.img_ext)
        image_path = os.path.join(
            self.data_path,
            folder,
            "image_0{}/data".format(self.side_map[side]),
            f_str)
        return image_path

    def get_depth(self, folder, frame_index, side, do_flip):
        f_str = "{:010d}.png".format(frame_index)
        depth_path = os.path.join(
            self.data_path,
            folder,
            "proj_depth/groundtruth/image_0{}".format(self.side_map[side]),
            f_str)

        depth_gt = pil.open(depth_path)
        depth_gt = depth_gt.resize(self.full_res_shape, pil.NEAREST)
        depth_gt = np.array(depth_gt).astype(np.float32) / 256

        if do_flip:
            depth_gt = np.fliplr(depth_gt)

        return depth_gt





def compute_loss(inputs, model_outputs):
    """Generate the warped (reprojected) color images for a minibatch.
    Generated images are saved into the `outputs` dictionary.
    """
    loss = {}
    for scale in [0, 1, 2, 3]:
        im0_depth = model_outputs[("depth", 0, scale)]
        im1_depth = model_outputs[("depth", 1, scale)]
        forward_pose = model_outputs[("forward_pose", scale)]
        backward_pose = model_outputs[("backward_pose", scale)]
        if True:
            source_scale = scale
        else:
            im0_depth = F.interpolate(
                im0_depth, [192, 640], mode="bilinear", align_corners=False)
            im1_depth = F.interpolate(
                im1_depth, [192, 640], mode="bilinear", align_corners=False)
            forward_pose =  F.interpolate(
                forward_pose, [192, 640], mode="bilinear", align_corners=False)
            backward_pose =  F.interpolate(
                backward_pose, [192, 640], mode="bilinear", align_corners=False)
            source_scale = 0

        im0 = inputs[('color', 0, source_scale)]
        im1 = inputs[('color', 1, source_scale)]
        intrinsics = inputs[('K', source_scale)][:, :3, :3]
        intrinsics_inv = inputs[("inv_K", source_scale)][:, :3, :3]
        forward_flow = compute_rigid_flow(im0_depth, forward_pose, intrinsics, intrinsics_inv)
        backward_flow = compute_rigid_flow(im1_depth, backward_pose, intrinsics, intrinsics_inv)

        backward_flow_from_forward_flow = flow_inverse_warp(forward_flow, backward_flow)
        forward_flow_from_backward_flow = flow_inverse_warp(backward_flow, forward_flow)
        forward_flow_mask = compute_flow_mask(forward_flow, forward_flow_from_backward_flow)
        backward_flow_mask = compute_flow_mask(backward_flow, backward_flow_from_forward_flow)

        forward_flow_mask, backward_flow_mask = compute_flow_mask(forward_flow, backward_flow)
        im0_hat, im0_transformed_depth, im1_sampled_depth, valid_mask0 = inverse_warp(im1, im0_depth, forward_pose, intrinsics, intrinsics_inv, im1_depth)
        im1_hat, im1_transformed_depth, im0_sampled_depth, valid_mask1 = inverse_warp(im0, im1_depth, backward_pose, intrinsics, intrinsics_inv, im0_depth)
        im0_mask = (valid_mask0 & forward_flow_mask).float()
        im1_mask = (valid_mask1 & backward_flow_mask).float()
        im0_recon_loss = torch.sum(perception_similarity_loss(im0_hat, im0) * im0_mask) / torch.sum(im0_mask).clamp(min=1)
        im1_recon_loss = torch.sum(perception_similarity_loss(im1_hat, im1) * im1_mask) / torch.sum(im1_mask).clamp(min=1)
        im0_smooth_loss = edge_aware_smooth_loss(im0_depth, aux=im0)
        im1_smooth_loss = edge_aware_smooth_loss(im1_depth, aux=im1)

        model_outputs[("color", 0, scale)] = im0_hat
        model_outputs[("color", 1, scale)] = im1_hat
        model_outputs[("flow mask", 0, scale)] = forward_flow_mask
        model_outputs[("flow_mask", 1, scale)] = backward_flow_mask
        
        loss[f'scale{scale}'] = 1 * (im0_recon_loss + im1_recon_loss) + 0.001 * (im0_smooth_loss + im1_smooth_loss)
    return loss

def reduce_loss(losses, opt_scales):
    final_loss = 0
    for scale, scale_weight in enumerate(opt_scales):
        final_loss += scale_weight * loss[f'scale{scale}']
    return final_loss

fpath = os.path.join(os.path.dirname(__file__), "splits", "eigen_zhou", "{}_files.txt")
train_filenames = readlines(fpath.format("train"))
val_filenames = readlines(fpath.format("val"))
img_ext = '.png'
train_dataset = KITTIRAWDataset('/mnt/remote/pure_dataset/perception_datasets/kitti_data', train_filenames, 192, 640,
            [0, 1], 4, is_train=True, img_ext=img_ext)
train_loader = DataLoader(
            train_dataset, 2, True,
            num_workers=1, pin_memory=True, drop_last=True)
encoder = networks.ResnetEncoder(18, False)
depth_decoder = networks.DepthDecoder(encoder.num_ch_enc)
pose_decoder = networks.SimplePoseDecoder(depth_decoder.num_ch_dec, scales=range(4), num_ch_dec=[16, 32, 64, 128]*2)
for batch_idx, inputs in enumerate(train_loader):
    print(inputs.keys())
    all_color_aug = torch.cat([inputs[("color_aug", i, 0)] for i in [0, 1]])
    all_features = encoder(all_color_aug)
    print(f' encoder features shapes {[x.shape for x in all_features]}')
    outputs, decoder_features = depth_decoder(all_features)
    print(f' output shapes {[(key, value.shape) for key,value in outputs.items()]}')
    print(f' decoder features shapes {[(key, value.shape) for key,value in decoder_features.items()]}')
    all_outputs = [torch.split(outputs[('depth', scale)], 2) for scale in range(4)]
    all_features = [torch.split(decoder_features[('features', scale)], 2) for scale in range(4)]
    intrinsics = [inputs[("K", scale)] for scale in range(4)]
    inv_intrinsics = [inputs[("inv_K", scale)] for scale in range(4)]
    pose_outputs = pose_decoder(all_features, all_outputs, intrinsics)
    print([(key, value.shape) for key, value in pose_outputs.items()])
    loss = compute_loss(inputs, pose_outputs)
    final_loss = reduce_loss(loss, [1, 0.5, 0.25, 0.125])
    print(final_loss)
    break