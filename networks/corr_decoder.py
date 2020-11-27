# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

import torch
import torch.nn as nn
from torch.autograd import Variable
import os
os.environ['PYTHON_EGG_CACHE'] = 'tmp/' # a writable directory 
from networks.spatial_correlation_sampler import SpatialCorrelationSampler
import numpy as np
from dense_reprojection import inverse_warp

def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):   
    return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, 
                        padding=padding, dilation=dilation, bias=True),
            nn.LeakyReLU(0.1))

def predict_pose(in_planes):
    return nn.Conv2d(in_planes,6,kernel_size=3,stride=1,padding=1,bias=True)

def deconv(in_planes, out_planes, kernel_size=4, stride=2, padding=1):
    return nn.ConvTranspose2d(in_planes, out_planes, kernel_size, stride, padding, bias=True)

class CorrDecoder(nn.Module):
    def __init__(self, num_decoder_channels, max_displacement=4, lrelu_slope=0.01):
        super(CorrDecoder, self).__init__()
        self.corr    = SpatialCorrelationSampler(kernel_size=1, patch_size=max_displacement * 2 + 1, stride=1, padding=0, dilation_patch=1)
        self.leakyRELU = nn.LeakyReLU(0.1)
        
        nd = (2*max_displacement+1)**2
        dd = np.cumsum([128,128,96,64,32])

        
        od = nd
        self.conv4_0 = conv(od,      128, kernel_size=3, stride=1)
        self.conv4_1 = conv(od+dd[0],128, kernel_size=3, stride=1)
        self.conv4_2 = conv(od+dd[1],96,  kernel_size=3, stride=1)
        self.conv4_3 = conv(od+dd[2],64,  kernel_size=3, stride=1)
        self.conv4_4 = conv(od+dd[3],32,  kernel_size=3, stride=1)
        self.predict_pose4 = predict_pose(od+dd[4]) 
        self.deconv4 = deconv(6, 6, kernel_size=4, stride=2, padding=1) 
        self.upfeat4 = deconv(od+dd[4], 6, kernel_size=4, stride=2, padding=1) 
        
        od = nd+num_decoder_channels[2]+12
        self.conv3_0 = conv(od,      128, kernel_size=3, stride=1)
        self.conv3_1 = conv(od+dd[0],128, kernel_size=3, stride=1)
        self.conv3_2 = conv(od+dd[1],96,  kernel_size=3, stride=1)
        self.conv3_3 = conv(od+dd[2],64,  kernel_size=3, stride=1)
        self.conv3_4 = conv(od+dd[3],32,  kernel_size=3, stride=1)
        self.predict_pose3 = predict_pose(od+dd[4]) 
        self.deconv3 = deconv(6, 6, kernel_size=4, stride=2, padding=1) 
        self.upfeat3 = deconv(od+dd[4], 6, kernel_size=4, stride=2, padding=1) 
        
        od = nd+num_decoder_channels[1]+12
        self.conv2_0 = conv(od,      128, kernel_size=3, stride=1)
        self.conv2_1 = conv(od+dd[0],128, kernel_size=3, stride=1)
        self.conv2_2 = conv(od+dd[1],96,  kernel_size=3, stride=1)
        self.conv2_3 = conv(od+dd[2],64,  kernel_size=3, stride=1)
        self.conv2_4 = conv(od+dd[3],32,  kernel_size=3, stride=1)
        self.predict_pose2 = predict_pose(od+dd[4]) 
        self.deconv2 = deconv(6, 6, kernel_size=4, stride=2, padding=1) 
        self.upfeat2 = deconv(od+dd[4], 6, kernel_size=4, stride=2, padding=1) 
        
        od = nd+num_decoder_channels[0]+12
        self.conv1_0 = conv(od,      128, kernel_size=3, stride=1)
        self.conv1_1 = conv(od+dd[0],128, kernel_size=3, stride=1)
        self.conv1_2 = conv(od+dd[1],96,  kernel_size=3, stride=1)
        self.conv1_3 = conv(od+dd[2],64,  kernel_size=3, stride=1)
        self.conv1_4 = conv(od+dd[3],32,  kernel_size=3, stride=1)
        self.predict_pose = predict_pose(od+dd[4]) 

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal(m.weight.data, mode='fan_in')
                if m.bias is not None:
                    m.bias.data.zero_()


    def forward(self, all_features, all_outputs, intrinsics):
        self.outputs = {}
        c11, c12, c13, c14 = [feat[0] for feat in all_features]
        c21, c22, c23, c24 = [feat[1] for feat in all_features]

        d11, d12, d12, d12 = [output[0] for output in all_outputs]
        d21, d22, d22, d22 = [output[1] for output in all_outputs]

        i11, i12, i13, i14 = intrinsics

        corr4 = self.corr(c14, c24)  
        corr4 = self.leakyRELU(corr4)
        # x = torch.cat((corr4, c14, up_pose5, up_feat5), 1)
        x = torch.cat((self.conv4_0(corr4), corr4),1)
        x = torch.cat((self.conv4_1(x), x),1)
        x = torch.cat((self.conv4_2(x), x),1)
        x = torch.cat((self.conv4_3(x), x),1)
        x = torch.cat((self.conv4_4(x), x),1)
        pose4 = self.predict_pose4(x)
        up_pose4 = self.deconv4(pose4)
        up_feat4 = self.upfeat4(x)


        warp3 = inverse_warp(c23, d13, up_pose4, i13)
        corr3 = self.corr(c13, warp3) 
        corr3 = self.leakyRELU(corr3)
        

        x = torch.cat((corr3, c13, up_pose4, up_feat4), 1)
        x = torch.cat((self.conv3_0(x), x),1)
        x = torch.cat((self.conv3_1(x), x),1)
        x = torch.cat((self.conv3_2(x), x),1)
        x = torch.cat((self.conv3_3(x), x),1)
        x = torch.cat((self.conv3_4(x), x),1)
        pose3 = self.predict_pose3(x)
        up_pose3 = self.deconv3(pose3)
        up_feat3 = self.upfeat3(x)


        warp2 = inverse_warp(c22, d12, up_pose3, i12) 
        corr2 = self.corr(c12, warp2)
        corr2 = self.leakyRELU(corr2)
        x = torch.cat((corr2, c12, up_pose3, up_feat3), 1)
        x = torch.cat((self.conv2_0(x), x),1)
        x = torch.cat((self.conv2_1(x), x),1)
        x = torch.cat((self.conv2_2(x), x),1)
        x = torch.cat((self.conv2_3(x), x),1)
        x = torch.cat((self.conv2_4(x), x),1)
        pose2 = self.predict_pose2(x)
        up_pose2 = self.deconv2(pose2)
        up_feat2 = self.upfeat2(x)
 
        warp1 = inverse_warp(c21, d11, up_pose2, i11) 
        corr1 = self.corr(c11, warp1)
        corr1 = self.leakyRELU(corr1)
        x = torch.cat((corr1, c11, up_pose2, up_feat2), 1)
        x = torch.cat((self.conv1_0(x), x),1)
        x = torch.cat((self.conv1_1(x), x),1)
        x = torch.cat((self.conv1_2(x), x),1)
        x = torch.cat((self.conv1_3(x), x),1)
        x = torch.cat((self.conv1_4(x), x),1)
        pose = self.predict_pose(x)
        
        return {
            ("pose", 0): pose,
            ("pose", 1): pose2,
            ("pose", 2): pose3,
            ("pose", 3): pose4,
        }