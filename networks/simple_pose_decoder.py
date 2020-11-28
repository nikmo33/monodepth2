# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import numpy as np
import torch
import torch.nn as nn

from collections import OrderedDict
from layers import *


class SimplePoseDecoder(nn.Module):
    def __init__(self, num_ch_enc, scales=range(4), num_output_channels=6, use_skips=True, num_ch_dec=[16, 32, 64, 128], compute_backward_pose = True, egomotion_scale_factor=0.01):
        super(SimplePoseDecoder, self).__init__()

        self.num_output_channels = num_output_channels
        self.use_skips = use_skips
        self.upsample_mode = 'nearest'
        self.scales = scales
        self.compute_backward_pose = compute_backward_pose
        self.egomotion_scale_factor = egomotion_scale_factor
        self.num_ch_dec = np.array(num_ch_dec)
        # decoder
        self.convs = OrderedDict()
        for i in range(3, -1, -1):
            self.convs[("poseconv", i)] = Conv3x3(self.num_ch_dec[i]*2, self.num_output_channels)

        self.decoder = nn.ModuleList(list(self.convs.values()))
        self.sigmoid = nn.Sigmoid()

    def forward(self, all_features, all_outputs, intrinsics, intrinsics_inv):
        input_features = [torch.cat((im1_feat, im2_feat),dim=1) for (im1_feat, im2_feat) in all_features]
        d11, d12, d13, d14 = [output[0] for output in all_outputs]
        d21, d22, d23, d24 = [output[1] for output in all_outputs]
        outputs = {
            ("depth", 0, 0): d11,
            ("depth", 0, 1): d12,
            ("depth", 0, 2): d13,
            ("depth", 0, 3): d14,
            ("depth", 1, 0): d21,
            ("depth", 1, 1): d22,
            ("depth", 1, 2): d23,
            ("depth", 1, 3): d24,
            
        }
        # decoder
        for i in range(3, -1, -1):
            x = input_features[i]
            outputs[("forward_pose", i)] = self.egomotion_scale_factor * self.convs[("poseconv", i)](x)

        if self.compute_backward_pose:
            backward_features = [torch.cat((im2_feat, im1_feat),dim=1) for (im1_feat, im2_feat) in all_features]
            for i in range(3, -1, -1):
                x = input_features[i]
                outputs[("backward_pose", i)] = self.egomotion_scale_factor * self.convs[("poseconv", i)](x)
        
        return outputs
