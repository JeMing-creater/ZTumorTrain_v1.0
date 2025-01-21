# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.nn as nn
import torch.nn.functional as F
# from __future__ import annotations
import time
from mamba_ssm import Mamba

class SwishImplementation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i):
        ctx.save_for_backward(i)
        return i * torch.sigmoid(i)

    @staticmethod
    def backward(ctx, grad_output):
        sigmoid_i = torch.sigmoid(ctx.saved_variables[0])
        return grad_output * (sigmoid_i * (1 + ctx.saved_variables[0] * (1 - sigmoid_i)))
    
# Swish激活函数
class Swish(nn.Module):
    def forward(self, x):
        return SwishImplementation.apply(x)

class MlpChannel(nn.Module):
    def __init__(self,hidden_size, mlp_dim, shallow=True):
        super().__init__()
        self.fc1 = nn.Conv3d(hidden_size, mlp_dim, 1)
        if shallow == True:
            self.act = nn.GELU()
        else:
            self.act = Swish()
        # self.act = nn.ReLU()
        self.fc2 = nn.Conv3d(mlp_dim, hidden_size, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x

class GMPBlock(nn.Module):
    def __init__(self, in_channles, shallow=True) -> None:
        super().__init__()

        self.proj = nn.Conv3d(in_channles, in_channles, 3, 1, 1)
        self.norm = nn.InstanceNorm3d(in_channles)
        if shallow == True:
            self.nonliner = nn.GELU()
        else:
            self.nonliner = Swish()
        # self.nonliner = nn.ReLU()

        self.proj2 = nn.Conv3d(in_channles, in_channles, 3, 1, 1)
        self.norm2 = nn.InstanceNorm3d(in_channles)
        if shallow == True:
            self.nonliner2 = nn.GELU()
        else:
            self.nonliner2 = Swish()

        self.proj3 = nn.Conv3d(in_channles, in_channles, 1, 1, 0)
        self.norm3 = nn.InstanceNorm3d(in_channles)
        if shallow == True:
            self.nonliner3 = nn.GELU()
        else:
            self.nonliner3 = Swish()

        self.proj4 = nn.Conv3d(in_channles, in_channles, 1, 1, 0)
        self.norm4 = nn.InstanceNorm3d(in_channles)
        if shallow == True:
            self.nonliner4 = nn.GELU()
        else:
            self.nonliner4 = Swish()

    def forward(self, x):

        x_residual = x 

        x1 = self.proj(x)
        x1 = self.norm(x1)
        x1 = self.nonliner(x1)

        x1 = self.proj2(x1)
        x1 = self.norm2(x1)
        x1 = self.nonliner2(x1)

        x2 = self.proj3(x)
        x2 = self.norm3(x2)
        x2 = self.nonliner3(x2)

        x = x1 + x2
        x = self.proj4(x)
        x = self.norm4(x)
        x = self.nonliner4(x)
        
        return x + x_residual
    
class AMambaBlock(nn.Module):
    def __init__(self, dim, d_state = 16, d_conv = 4, expand = 2, head=4, num_slices=4, step = 1):
        super(AMambaBlock, self).__init__()
        self.dim = dim
        self.step = step
        self.num_heads = head
        self.head_dim = dim // head
        self.norm = nn.LayerNorm(dim)
        
        self.mamba = Mamba(
                d_model=dim, # Model dimension d_model
                d_state=d_state,  # SSM state expansion factor
                d_conv=d_conv,    # Local convolution width
                expand=expand,    # Block expansion factor
                bimamba_type="v3",
                nslices = num_slices
        )
        # qkv
        self.qkv = nn.Conv3d(dim, dim * 3, kernel_size=1, bias=False)
        
        
    def forward(self, x):
        x_skip = x
        B, C = x.shape[:2]
        assert C == self.dim
        n_tokens = x.shape[2:].numel()
        img_dims = x.shape[2:]
        
        x_flat = x.reshape(B, C, n_tokens).transpose(-1, -2)
        x_norm = self.norm(x_flat)
        
        x_mamba = self.mamba(x_norm)

        out = x_mamba.transpose(-1, -2).reshape(B, C, *img_dims)
        
        B, C, H, W, Z = x.shape
        q, k, v = self.qkv(out).view(B, self.num_heads, -1, H * W * Z).split(
            [self.head_dim, self.head_dim, self.head_dim],
            dim=2)
        attn = (q.transpose(-2, -1) @ k).softmax(-1)
        out = (v @ attn.transpose(-2, -1)).view(B, -1, H, W, Z)
      
        out = out + x_skip
        
        
        return out 

class Encoder(nn.Module):
    def __init__(self, in_chans=4, kernel_sizes=[4, 2, 2, 2], depths=[1, 1, 1, 1], dims=[48, 96, 192, 384], num_slices_list = [64, 32, 16, 8],
                 out_indices=[0, 1, 2, 3], heads=[1, 2, 4, 4]):
        super().__init__()
        self.downsample_layers = nn.ModuleList() # stem and 3 intermediate downsampling conv layers
        stem = nn.Sequential(
              nn.Conv3d(in_chans, dims[0], kernel_size=kernel_sizes[0], stride=kernel_sizes[0]),
              )
        
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                nn.InstanceNorm3d(dims[i]),
                nn.Conv3d(dims[i], dims[i+1], kernel_size=kernel_sizes[i+1], stride=kernel_sizes[i+1]),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList()
        self.gscs = nn.ModuleList()
        self.stages = nn.ModuleList()
        self.gscs = nn.ModuleList()
        cur = 0
        for i in range(4):
            shallow = True
            if i > 1:
                shallow = False
            gsc = GMPBlock(dims[i], shallow)

            
            stage = nn.Sequential(
                *[AMambaBlock(dim=dims[i], num_slices=num_slices_list[i], head = heads[i], step=i) for j in range(depths[i])]
            )

            self.stages.append(stage)
            
            self.gscs.append(gsc)
            cur += depths[i]

        self.out_indices = out_indices

        self.mlps = nn.ModuleList()
        for i_layer in range(4):
            layer = nn.InstanceNorm3d(dims[i_layer])
            layer_name = f'norm{i_layer}'
            self.add_module(layer_name, layer)
            if i_layer>=2:
                self.mlps.append(MlpChannel(dims[i_layer], 2 * dims[i_layer], False))
            else:
                self.mlps.append(MlpChannel(dims[i_layer], 2 * dims[i_layer], True))
        
    def forward(self, x):
        # outs = []
        feature_out = []
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.gscs[i](x)
            
            feature_out.append(self.stages[i](x))

            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                x = norm_layer(x)
                x = self.mlps[i](x)
                # outs.append(x_out)   
        return x, feature_out

class TransposedConvLayer(nn.Module):
    def __init__(self, dim_in, dim_out, head, r):
        super(TransposedConvLayer, self).__init__()
        self.transposed1 = nn.ConvTranspose3d(dim_in,
                                             dim_out,
                                             kernel_size=r,
                                             stride=r)
        self.norm = nn.GroupNorm(num_groups=1, num_channels=dim_out)
        self.transposed2 = nn.ConvTranspose3d(dim_out*2,
                                             dim_out,
                                             kernel_size=1,
                                             stride=1)

    def forward(self, x, feature):
        x = self.transposed1(x)
        x = torch.cat((x, feature), dim=1)
        # x = self.Atten(x)
        x = self.transposed2(x)
        x = self.norm(x)
        return x

class HWABlock(nn.Module):
    def __init__(self, in_chans = 2, kernel_sizes = [1,2,4,8]):
        super(HWABlock, self).__init__()
        self.dwa1 = nn.Conv3d(1, 1, kernel_size=kernel_sizes[0], stride=kernel_sizes[0])
        self.dwa2 = nn.Conv3d(1, 1, kernel_size=kernel_sizes[1], stride=kernel_sizes[1])
        self.dwa3 = nn.Conv3d(1, 1, kernel_size=kernel_sizes[2], stride=kernel_sizes[2])
        self.dwa4 = nn.Conv3d(1, 1, kernel_size=kernel_sizes[3], stride=kernel_sizes[3])
        self.fussion = nn.Conv3d(
            in_channels=4,  # 输入通道数
            out_channels=in_chans,  # 输出通道数
            kernel_size=3,  # 内核大小
            stride=1,  # 步长
            padding=1,  # 填充，以保持空间尺寸不变
            bias=True  # 是否使用偏置项
        )
        self.weights = nn.Parameter(torch.ones(in_chans))
        
    def dw_change(self, x, dw):
        x_ = dw(x)
        upsampled_tensor = F.interpolate(
            x_,
            size = (x.shape[2],x.shape[3],x.shape[4]),
            mode = 'trilinear',
            align_corners = True 
        )
        return upsampled_tensor
    
    def forward(self, x):
        _, num_channels, _, _, _ = x.shape
        normalized_weights = F.softmax(self.weights, dim=0)
        all_tensor = []
        for i in range(num_channels):
            now_tensor = []
            channel_tensor = x[:, i, :, :, :].unsqueeze(1)
            now_tensor.append(self.dw_change(channel_tensor, self.dwa1))
            now_tensor.append(self.dw_change(channel_tensor, self.dwa2))
            now_tensor.append(self.dw_change(channel_tensor, self.dwa3))
            now_tensor.append(self.dw_change(channel_tensor, self.dwa4))
            now_tensor = torch.cat(now_tensor, dim=1)
            now_tensor = self.fussion(now_tensor)
            
            # weighted_sum = sum(w * t for w, t in zip(normalized_weights, now_tensor))
            all_tensor.append(now_tensor)
        
        x = sum(w * t for w, t in zip(normalized_weights, all_tensor))
        
        return x
         
class HWAUNETR(nn.Module):
    def __init__(self, in_chans=4, out_chans=3, fussion = [1,2,4,8], kernel_sizes=[4, 2, 2, 2], depths=[1, 1, 1, 1], dims=[48, 96, 192, 384], heads=[1, 2, 4, 4], hidden_size=768, num_slices_list = [64, 32, 16, 8],
                out_indices=[0, 1, 2, 3]):
        super(HWAUNETR, self).__init__()
        self.fussion = HWABlock(in_chans=in_chans, kernel_sizes = fussion)
        self.Encoder = Encoder(in_chans=in_chans, kernel_sizes=kernel_sizes, depths=depths, dims=dims, num_slices_list = num_slices_list,
                out_indices=out_indices, heads=heads)

        self.hidden_downsample = nn.Conv3d(dims[3], hidden_size, kernel_size=2, stride=2)
        
        self.TSconv1 = TransposedConvLayer(dim_in=hidden_size, dim_out=dims[3], head=heads[3], r=2)
        
        self.TSconv2 = TransposedConvLayer(dim_in=dims[3], dim_out=dims[2], head=heads[2], r=kernel_sizes[3])
        self.TSconv3 = TransposedConvLayer(dim_in=dims[2], dim_out=dims[1], head=heads[1], r=kernel_sizes[2])
        self.TSconv4 = TransposedConvLayer(dim_in=dims[1], dim_out=dims[0], head=heads[0], r=kernel_sizes[1])

        self.SegHead = nn.ConvTranspose3d(dims[0],out_chans,kernel_size=kernel_sizes[0],stride=kernel_sizes[0])
        
    def forward(self, x):
        x = self.fussion(x)
        
        outs, feature_out = self.Encoder(x)
        
        deep_feature = self.hidden_downsample(outs)
        
        x = self.TSconv1(deep_feature, feature_out[-1])
        x = self.TSconv2(x, feature_out[-2])
        x = self.TSconv3(x, feature_out[-3])
        x = self.TSconv4(x, feature_out[-4])
        x = self.SegHead(x)
        
        return x
    
def test_weight(model, x):
    for i in range(0, 3):
        _ = model(x)
    start_time = time.time()
    output = model(x)
    end_time = time.time()
    need_time = end_time - start_time
    from thop import profile
    flops, params = profile(model, inputs=(x,))
    throughout = round(x.shape[0] / (need_time / 1), 3)
    return flops, params, throughout


def Unitconversion(flops, params, throughout):
    print('params : {} M'.format(round(params / (1000**2), 2)))
    print('flop : {} G'.format(round(flops / (1000**3), 2)))
    print('throughout: {} FPS'.format(throughout))

if __name__ == '__main__':
    device = 'cuda:4'
    # x = torch.randn(size=(1, 4, 96, 96, 96)).to(device)
    # x = torch.randn(size=(1, 4, 128, 128, 128)).to(device)
    x = torch.randn(size=(4, 2, 128, 128, 64)).to(device)
    # model = SegMamba(in_chans=4,out_chans=3).to(device)
    
    model = HWAUNETR(in_chans=2, out_chans=2, fussion = [1, 2, 4, 8], kernel_sizes=[4, 2, 2, 2], depths=[1, 1, 1, 1], dims=[48, 96, 192, 384], heads=[1, 2, 4, 4], hidden_size=768, num_slices_list = [64, 32, 16, 8], out_indices=[0, 1, 2, 3]).to(device)
    
    # model = SppFusion().to(device)
    
    print(model(x).shape)
    # flops, param, throughout = test_weight(model, x)
    # Unitconversion(flops, param, throughout)