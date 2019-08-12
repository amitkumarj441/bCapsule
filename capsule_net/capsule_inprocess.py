"""
Extend on top of https://arxiv.org/abs/1710.09829
"""
import numpy as np
import sys

import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable

class CapsuleLayer(nn.Module):
    def __init__(self, 
                  input_dim, 
                  output_dim, 
                  input_atoms, 
                  output_atoms, 
                  paras=None, 
                  num_routing=3, 
                  leaky=False, 
                  kernel_size=None, 
                  stride=None,
                  ):
        super(CapsuleLayer, self).__init__()
        self.input_shape = (input_dim, input_atoms)  # omit batch dimension
        self.output_shape = (output_dim, output_atoms)
        self.num_routing = num_routing
        self.leaky = leaky
        if paras is None:
            self.weights = nn.Parameter(torch.randn(input_dim, input_atoms, output_dim * output_atoms))
            self.biases = nn.Parameter(torch.randn(output_dim, output_atoms))
        else:
            self.weights = nn.Parameter(paras['weights'])
            self.biases = nn.Parameter(paras['biases'])

    def _squash(self, input_tensor):
        norm = torch.norm(input_tensor, dim=2, keepdim=True)
        norm_squared = norm * norm
        return (input_tensor / norm) * (norm_squared / (1 + norm_squared))

    def _leaky_route(self, x, output_dim):
        leak = torch.zeros(x.shape).to(x.device.type)
        leak = leak.sum(dim=2, keepdim=True)
        leak_x = torch.cat((leak, x), 2)
        leaky_routing = F.softmax(leak_x, dim=2)
        return leaky_routing[:, :, 1:]

    def _margin_loss(self, labels, raw_logits, margin=0.4, downweight=0.5):
        logits = raw_logits - 0.5
        positive_cost = labels * torch.lt(logits, margin).float() * torch.pow(logits - margin, 2)
        negative_cost = (1 - labels) * torch.gt(logits, -margin).float() * torch.pow(logits + margin, 2)
        margin_loss = 0.5 * positive_cost + downweight * 0.5 * negative_cost
        per_example_loss = torch.sum(margin_loss, dim=-1)
        loss = torch.mean(per_example_loss)
        return loss

    def forward(self, x, fast_weights=None):
        x = x.unsqueeze(-1).repeat(1, 1, 1, self.output_shape[0]*self.output_shape[1])  # [b, i, i_o, j*j_o]
        if fast_weights is None:
            votes = torch.sum(x * self.weights, dim=2)  # [b, i, j*j_o]
        else:
            votes = torch.sum(x * fast_weights['weights'], dim=2)  # [b, i, j*j_o]
        votes_reshaped = torch.reshape(votes,
                                       [-1, self.input_shape[0], self.output_shape[0], self.output_shape[1]])  # [b, i, j, j_o]

        # routing loop
        logits = torch.zeros(x.shape[0], self.input_shape[0], self.output_shape[0]).to(x.device.type)  # [b, i, j]
        for i in range(self.num_routing):
            if self.leaky:
                route = self._leaky_route(logits, self.output_shape[0])
            else:
                route = F.softmax(logits, dim=2)
            route = route.unsqueeze(-1)  # [b, i, j, 1]
            preactivate_unrolled = route * votes_reshaped   # [b, i, j, j_o]

            if fast_weights is None:
                s = preactivate_unrolled.sum(1, keepdim=True) + self.biases  # [b, 1, j, j_o]
            else:
                s = preactivate_unrolled.sum(1, keepdim=True) + fast_weights['biases']  # [b, 1, j, j_o]

            v = self._squash(s)

            distances = (votes_reshaped * v).sum(dim=3)  # [b, i, j]
            logits = logits + distances

        return v


class CapsuleClassification(nn.Module):
    def __init__(self, num_classes, input_atoms, out_atoms):
        super(CapsuleClassification, self).__init__()
        self.num_classes = num_classes
        self.out_atoms = out_atoms
        self.input_atoms = input_atoms
        self.W = nn.Parameter(torch.randn(1, 1, self.num_classes, out_atoms, self.input_atoms))
        self.softmax = nn.Softmax(dim=2)
    
    def dynamic_routing(self,
                        input_tensor,
                        mul,
                        num_classes,
                        num_routing):
        b_ij = Variable(torch.zeros((1, mul, num_classes, 1))).cuda()
        batch_size = input_tensor.size(0)
        input_tensor = torch.stack([input_tensor] * self.num_classes, dim=2).unsqueeze(4)
        W = torch.cat([self.W] * batch_size, dim=0)
        u_hat = torch.matmul(W, input_tensor)
        for i in range(num_routing):
            c_ij = F.softmax(b_ij, dim=1)
            c_ij = torch.cat([c_ij] * batch_size, dim=0).unsqueeze(4)

            s_j = (c_ij * u_hat).sum(dim=1, keepdim=True)

            v_j = squash(s_j, layer='fc')
            v_j1 = torch.cat([v_j] * mul, dim=1)

            u_vj1 = torch.matmul(u_hat.transpose(3, 4), v_j1).squeeze(4).mean(dim=0, keepdim=True)

            b_ij = b_ij + u_vj1
        return v_j.squeeze(1)
        
        
    def forward(self, input):
        batch, input_atoms, mul = input.size()
        input_tensor = input.view(batch, mul, input_atoms)
        activation = self.dynamic_routing(input_tensor=input_tensor,
                                          mul=mul,
                                          num_classes=self.num_classes,
                                          num_routing=3)
        return activation
    
    

class CapsuleCNN(nn.Module):
    def __init__(self,
                 input_dim,
                 out_dim,
                 input_atoms=8,
                 out_atoms=8,
                 stride=2,
                 kernel_size=5,
                 **routing):
        super(CapsuleCNN, self).__init__()
        self.input_dim = input_dim
        self.input_atoms = input_atoms
        self.out_dim = out_dim
        self.out_atoms = out_atoms
        self.conv_capsule1 = nn.Conv2d(input_dim, out_atoms * out_dim, (kernel_size, 1), stride=stride)

    def forward(self, input_tensor):
        input_shape = input_tensor.size()
        batch, _, in_height, in_width = input_shape
        conv = self.conv_capsule1(input_tensor)
        conv_shape = conv.size()
        print('conv.shape', conv_shape)
        _, _, conv_height, conv_width = conv_shape
        conv_reshaped = conv.view(batch, self.out_atoms, conv_height * conv_width * self.out_dim)
        return squash(conv_reshaped, layer='conv')
    
  
class CNNUnit(nn.Module):
    def __init__(self,
                 input_dim,
                 out_dim,
                 kernel_size,
                 stride,
                 ):
        super(CNNUnit, self).__init__()
        self.conv0 = nn.Conv2d(in_channels=input_dim,
                               out_channels=out_dim,
                               kernel_size=(kernel_size, 1),
                               stride=stride,
                               bias=True)

    def forward(self, input):
        return self.conv0(input)
    
    
 
