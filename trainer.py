# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import numpy as np
import time

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import json

from utils import *
from kitti_utils import *
from layers import *

import datasets
import networks
from IPython import embed
from dense_reprojection import *
from vis_utils import *
class Trainer:
    def __init__(self, options):
        self.opt = options
        self.log_path = os.path.join(self.opt.log_dir, self.opt.model_name)

        # checking height and width are multiples of 32
        assert self.opt.height % 32 == 0, "'height' must be a multiple of 32"
        assert self.opt.width % 32 == 0, "'width' must be a multiple of 32"

        self.models = {}
        self.parameters_to_train = []

        self.device = torch.device("cpu" if self.opt.no_cuda else "cuda")

        self.num_scales = len(self.opt.scales)
        self.num_input_frames = len(self.opt.frame_ids)
        self.num_pose_frames = 2

        assert self.opt.frame_ids[0] == 0, "frame_ids must start with 0"

        self.use_pose_net = not (self.opt.use_stereo and self.opt.frame_ids == [0])

        if self.opt.use_stereo:
            self.opt.frame_ids.append("s")

        self.models["encoder"] = networks.ResnetEncoder(self.opt.num_layers, self.opt.weights_init == "pretrained")
        self.models["encoder"].to(self.device)
        self.parameters_to_train += list(self.models["encoder"].parameters())

        self.models["depth"] = networks.DepthDecoder(self.models["encoder"].num_ch_enc, self.opt.scales)
        self.models["depth"].to(self.device)
        self.parameters_to_train += list(self.models["depth"].parameters())

        self.models["pose"] = networks.SimplePoseDecoder(self.models["depth"].num_ch_dec)
        self.models["pose"].to(self.device)
        self.parameters_to_train += list(self.models["pose"].parameters())

        self.model_optimizer = optim.Adam(self.parameters_to_train, self.opt.learning_rate)
        self.model_lr_scheduler = optim.lr_scheduler.StepLR(
            self.model_optimizer, self.opt.scheduler_step_size, 0.1)

        if self.opt.load_weights_folder is not None:
            self.load_model()

        print("Training model named:\n  ", self.opt.model_name)
        print("Models and tensorboard events files are saved to:\n  ", self.opt.log_dir)
        print("Training is using:\n  ", self.device)

        # data
        datasets_dict = {"kitti": datasets.KITTIRAWDataset,
                         "kitti_odom": datasets.KITTIOdomDataset}
        self.dataset = datasets_dict[self.opt.dataset]

        fpath = os.path.join(os.path.dirname(__file__), "splits", self.opt.split, "{}_files.txt")

        train_filenames = readlines(fpath.format("train"))
        val_filenames = readlines(fpath.format("val"))
        img_ext = '.png' if self.opt.png else '.jpg'

        num_train_samples = len(train_filenames)
        self.num_total_steps = num_train_samples // self.opt.batch_size * self.opt.num_epochs

        train_dataset = self.dataset(
            self.opt.data_path, train_filenames, self.opt.height, self.opt.width,
            self.opt.frame_ids, 4, is_train=True, img_ext=img_ext)
        self.train_loader = DataLoader(
            train_dataset, self.opt.batch_size, True,
            num_workers=self.opt.num_workers, pin_memory=True, drop_last=True)
        val_dataset = self.dataset(
            self.opt.data_path, val_filenames, self.opt.height, self.opt.width,
            self.opt.frame_ids, 4, is_train=False, img_ext=img_ext)
        self.val_loader = DataLoader(
            val_dataset, self.opt.batch_size, True,
            num_workers=self.opt.num_workers, pin_memory=True, drop_last=True)
        self.val_iter = iter(self.val_loader)

        self.writers = {}
        for mode in ["train", "val"]:
            self.writers[mode] = SummaryWriter(os.path.join(self.log_path, mode))

        self.depth_metric_names = [
            "de/abs_rel", "de/sq_rel", "de/rms", "de/log_rms", "da/a1", "da/a2", "da/a3"]

        print("Using split:\n  ", self.opt.split)
        print("There are {:d} training items and {:d} validation items\n".format(
            len(train_dataset), len(val_dataset)))

        self.save_opts()

    def set_train(self):
        """Convert all models to training mode
        """
        for m in self.models.values():
            m.train()

    def set_eval(self):
        """Convert all models to testing/evaluation mode
        """
        for m in self.models.values():
            m.eval()

    def train(self):
        """Run the entire training pipeline
        """
        self.epoch = 0
        self.step = 0
        self.start_time = time.time()
        for self.epoch in range(self.opt.num_epochs):
            self.run_epoch()
            if (self.epoch + 1) % self.opt.save_frequency == 0:
                self.save_model()

    def run_epoch(self):
        """Run a single epoch of training and validation
        """
        self.model_lr_scheduler.step()

        print("Training")
        self.set_train()

        for batch_idx, inputs in enumerate(self.train_loader):

            before_op_time = time.time()

            outputs, losses = self.process_batch(inputs)

            self.model_optimizer.zero_grad()
            losses["loss"].backward()
            self.model_optimizer.step()

            duration = time.time() - before_op_time

            # log less frequently after the first 2000 steps to save time & disk space
            early_phase = batch_idx % self.opt.log_frequency == 0 and self.step < 2000
            late_phase = self.step % 2000 == 0

            if early_phase or late_phase:
                self.log_time(batch_idx, duration, losses["loss"].cpu().data)

                if "depth_gt" in inputs:
                    self.compute_depth_losses(inputs, outputs, losses)

                self.log("train", inputs, outputs, losses)
                self.val()

            self.step += 1

    def process_batch(self, inputs):
        """Pass a minibatch through the network and generate images and losses
        """
        for key, ipt in inputs.items():
            inputs[key] = ipt.to(self.device)

        # If we are using a shared encoder for both depth and pose (as advocated
        # in monodepthv1), then all images are fed separately through the depth encoder.
        all_color_aug = torch.cat([inputs[("color_aug", i, 0)] for i in self.opt.frame_ids])
        all_features = self.models["encoder"](all_color_aug)
        outputs, decoder_features = self.models["depth"](all_features)
        all_features = [torch.split(decoder_features[('features', scale)], self.opt.batch_size) for scale in range(len(self.opt.scales))]
        all_outputs = [torch.split(outputs[('depth', scale)], self.opt.batch_size) for scale in range(len(self.opt.scales))]
        intrinsics = [inputs[("K", scale)] for scale in range(len(self.opt.scales))]
        inv_intrinsics = [inputs[("inv_K", scale)] for scale in range(len(self.opt.scales))]
        final_outputs = self.models["pose"](all_features, all_outputs, intrinsics, inv_intrinsics)

        losses = self.compute_loss(inputs, final_outputs)
        final_loss = self.reduce_loss(losses)

        return final_outputs, final_loss

    def val(self):
        """Validate the model on a single minibatch
        """
        self.set_eval()
        try:
            inputs = self.val_iter.next()
        except StopIteration:
            self.val_iter = iter(self.val_loader)
            inputs = self.val_iter.next()

        with torch.no_grad():
            outputs, losses = self.process_batch(inputs)

            if "depth_gt" in inputs:
                self.compute_depth_losses(inputs, outputs, losses)

            self.log("val", inputs, outputs, losses)
            del inputs, outputs, losses

        self.set_train()

    def reduce_loss(self, losses):
        final_loss = 0
        for scale, scale_weight in enumerate(self.opt.scale_weights):
            final_loss += scale_weight * losses[f'scale{scale}']
        return {'loss': final_loss}

    def compute_loss(self, inputs, model_outputs):
        """Generate the warped (reprojected) color images for a minibatch.
        Generated images are saved into the `outputs` dictionary.
        """
        loss = {}
        for scale in self.opt.scales:
            im0_depth = model_outputs[("depth", 0, scale)]
            im1_depth = model_outputs[("depth", 1, scale)]
            forward_pose = model_outputs[("forward_pose", scale)]
            backward_pose = model_outputs[("backward_pose", scale)]
            if self.opt.v1_multiscale:
                source_scale = scale
            else:
                im0_depth = F.interpolate(
                    im0_depth, [self.opt.height, self.opt.width], mode="bilinear", align_corners=False)
                im1_depth = F.interpolate(
                    im1_depth, [self.opt.height, self.opt.width], mode="bilinear", align_corners=False)
                forward_pose =  F.interpolate(
                    forward_pose, [self.opt.height, self.opt.width], mode="bilinear", align_corners=False)
                backward_pose =  F.interpolate(
                    backward_pose, [self.opt.height, self.opt.width], mode="bilinear", align_corners=False)
                source_scale = 0

            im0 = inputs[('color', 0, source_scale)]
            im1 = inputs[('color', 1, source_scale)]
            intrinsics = inputs[('K', source_scale)][:, :3, :3]
            forward_flow = compute_rigid_flow(im0_depth, forward_pose, intrinsics, intrinsics_inv)
            backward_flow = compute_rigid_flow(im1_depth, backward_pose, intrinsics, intrinsics_inv)

            backward_flow_from_forward_flow = flow_inverse_warp(forward_flow, backward_flow)
            forward_flow_from_backward_flow = flow_inverse_warp(backward_flow, forward_flow)
            forward_flow_mask = compute_flow_mask(forward_flow, forward_flow_from_backward_flow)
            backward_flow_mask = compute_flow_mask(backward_flow, backward_flow_from_forward_flow)

            im0_hat, im0_transformed_depth, im1_sampled_depth, valid_mask0 = inverse_warp(im1, im0_depth, forward_pose, intrinsics, im1_depth)
            im1_hat, im1_transformed_depth, im0_sampled_depth, valid_mask1 = inverse_warp(im0, im1_depth, backward_pose, intrinsics, im0_depth)
            im0_mask = (valid_mask0 & forward_flow_mask).float()
            im1_mask = (valid_mask1 & backward_flow_mask).float()
            im0_recon_loss = torch.sum(perception_similarity_loss(im0_hat, im0) * im0_mask) / torch.sum(im0_mask).clamp(min=1)
            im1_recon_loss = torch.sum(perception_similarity_loss(im1_hat, im1) * im1_mask) / torch.sum(im1_mask).clamp(min=1)
            im0_smooth_loss = edge_aware_smooth_loss(im0_depth, aux=im0)
            im1_smooth_loss = edge_aware_smooth_loss(im1_depth, aux=im1)
            model_outputs[("color", 0, scale)] = im0_hat
            model_outputs[("color", 1, scale)] = im1_hat
            model_outputs[("flow_mask", 0, scale)] = forward_flow_mask
            model_outputs[("flow_mask", 1, scale)] = backward_flow_mask
            model_outputs[("flow", 0, scale)] = forward_flow
            model_outputs[("flow", 1, scale)] = backward_flow
            loss[f'scale{scale}'] = self.opt.photo_loss_weight * (im0_recon_loss + im1_recon_loss) + self.opt.smooth_loss_weight * (im0_smooth_loss + im1_smooth_loss)
        return loss

    def compute_depth_losses(self, inputs, outputs, losses):
        """Compute depth metrics, to allow monitoring during training

        This isn't particularly accurate as it averages over the entire batch,
        so is only used to give an indication of validation performance
        """
        depth_pred = outputs[("depth", 0, 0)]
        depth_pred = torch.clamp(F.interpolate(
            depth_pred, [375, 1242], mode="bilinear", align_corners=False), 1e-3, 80)
        depth_pred = depth_pred.detach()

        depth_gt = inputs["depth_gt"]
        mask = depth_gt > 0

        # garg/eigen crop
        crop_mask = torch.zeros_like(mask)
        crop_mask[:, :, 153:371, 44:1197] = 1
        mask = mask * crop_mask

        depth_gt = depth_gt[mask]
        depth_pred = depth_pred[mask]
        depth_pred *= torch.median(depth_gt) / torch.median(depth_pred)

        depth_pred = torch.clamp(depth_pred, min=1e-3, max=80)

        depth_errors = compute_depth_errors(depth_gt, depth_pred)

        for i, metric in enumerate(self.depth_metric_names):
            losses[metric] = np.array(depth_errors[i].cpu())

    def log_time(self, batch_idx, duration, loss):
        """Print a logging statement to the terminal
        """
        samples_per_sec = self.opt.batch_size / duration
        time_sofar = time.time() - self.start_time
        training_time_left = (
            self.num_total_steps / self.step - 1.0) * time_sofar if self.step > 0 else 0
        print_string = "epoch {:>3} | batch {:>6} | examples/s: {:5.1f}" + \
            " | loss: {:.5f} | time elapsed: {} | time left: {}"
        print(print_string.format(self.epoch, batch_idx, samples_per_sec, loss,
                                  sec_to_hm_str(time_sofar), sec_to_hm_str(training_time_left)))

    def log(self, mode, inputs, outputs, losses):
        """Write an event to the tensorboard events file
        """
        writer = self.writers[mode]
        for l, v in losses.items():
            writer.add_scalar("{}".format(l), v, self.step)

        for j in range(min(4, self.opt.batch_size)):  # write a maxmimum of four images
            for s in self.opt.scales:
                for frame_id in self.opt.frame_ids:
                    writer.add_image(
                        "color_{}_{}/{}".format(frame_id, s, j),
                        inputs[("color", frame_id, s)][j].data, self.step)
                    if s == 0:
                        writer.add_image(
                            "color_pred_{}_{}/{}".format(frame_id, s, j),
                            outputs[("color", frame_id, s)][j].data, self.step)

                    writer.add_image(
                        "flow_pred_{}_{}/{}".format(frame_id, s, j),
                        flow_to_image(outputs[("flow", frame_id, s)][j].detach().cpu().numpy()), self.step)

                    writer.add_image(
                        "mask_from_flow_{}_{}/{}".format(frame_id, s, j),
                        heatmap_image(outputs[("flow_mask",frame_id, s)][j].float().detach().cpu().numpy()), self.step)
                    writer.add_image(
                        "depth_{}_{}/{}".format(frame_id, s, j),
                    heatmap_image(outputs[("depth",frame_id, s)][j].detach().cpu().numpy()), self.step)
                        
                

    def save_opts(self):
        """Save options to disk so we know what we ran this experiment with
        """
        models_dir = os.path.join(self.log_path, "models")
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
        to_save = self.opt.__dict__.copy()

        with open(os.path.join(models_dir, 'opt.json'), 'w') as f:
            json.dump(to_save, f, indent=2)

    def save_model(self):
        """Save model weights to disk
        """
        save_folder = os.path.join(self.log_path, "models", "weights_{}".format(self.epoch))
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        for model_name, model in self.models.items():
            save_path = os.path.join(save_folder, "{}.pth".format(model_name))
            to_save = model.state_dict()
            if model_name == 'encoder':
                # save the sizes - these are needed at prediction time
                to_save['height'] = self.opt.height
                to_save['width'] = self.opt.width
                to_save['use_stereo'] = self.opt.use_stereo
            torch.save(to_save, save_path)

        save_path = os.path.join(save_folder, "{}.pth".format("adam"))
        torch.save(self.model_optimizer.state_dict(), save_path)

    def load_model(self):
        """Load model(s) from disk
        """
        self.opt.load_weights_folder = os.path.expanduser(self.opt.load_weights_folder)

        assert os.path.isdir(self.opt.load_weights_folder), \
            "Cannot find folder {}".format(self.opt.load_weights_folder)
        print("loading model from folder {}".format(self.opt.load_weights_folder))

        for n in self.opt.models_to_load:
            print("Loading {} weights...".format(n))
            path = os.path.join(self.opt.load_weights_folder, "{}.pth".format(n))
            model_dict = self.models[n].state_dict()
            pretrained_dict = torch.load(path)
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            self.models[n].load_state_dict(model_dict)

        # loading adam state
        optimizer_load_path = os.path.join(self.opt.load_weights_folder, "adam.pth")
        if os.path.isfile(optimizer_load_path):
            print("Loading Adam weights")
            optimizer_dict = torch.load(optimizer_load_path)
            self.model_optimizer.load_state_dict(optimizer_dict)
        else:
            print("Cannot find Adam weights so Adam is randomly initialized")
