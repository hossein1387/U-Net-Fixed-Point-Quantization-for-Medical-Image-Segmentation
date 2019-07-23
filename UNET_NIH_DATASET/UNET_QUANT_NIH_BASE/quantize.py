import torch 
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import math
from torch.autograd import Variable

def to_nearest_power_of_two(v):
    # Return nearest power of two: pow(2, round(log2(value)))
    # import ipdb as pdb; pdb.set_trace()
    return torch.sign(v) * 2**torch.round(torch.log(torch.abs(v))/torch.log(torch.FloatTensor([2.]).type(v.type())))

def round_clamp(W, numbits=8):
    W = W.clamp(0,2**(numbits)-1)
    W = W.mul(2**(numbits)).round().div(2**(numbits))
    return W

def _to_fixed_point(v, i, f):
    # import ipdb as pdb; pdb.set_trace()
    pows = torch.arange(-f, 0, 1).cuda()
    max_float = Variable(torch.zeros(v.size()).type(v.data.type()) + torch.pow(2*torch.ones(pows.size()).cuda(), pows.float()).sum()) if f!=0 else Variable(torch.zeros(v.size()).type(v.data.type()))
    max_int   = Variable(torch.zeros(v.size()).type(v.data.type()) + 2**(i-1)) if i!=0 else Variable(torch.zeros(v.size()).type(v.data.type()))
    max_ =    torch.ones(v.size()).cuda()* (max_int + max_float)
    min_ = -1*torch.ones(v.size()).cuda()* (max_int + max_float)
    max_mask = v>max_
    v = max_mask.float()*max_ + (1-max_mask.float())*v
    min_mask = v<min_
    v = min_mask.float()*min_ + (1-min_mask.float())*v
    return v

def to_fixed_point(v, ibits, fbits):
    # import ipdb as pdb; pdb.set_trace()
    v_f = v.sign()*round_clamp(torch.abs(v) - torch.abs(v).floor(), fbits)
    v_i = v.sign()*round_clamp(v.abs().floor(), ibits)
    return (v_i + v_f)

def to_integer(v, num_bits):
    # import ipdb as pdb; pdb.set_trace()
    v_norm = (v-v.min()) / (v.max()-v.min())
    v_norm = 2**(num_bits) * v_norm
    v_norm = v_norm.floor()
    v_norm = v_norm - 2**(num_bits-1)
    return v_norm


def apply_quant(x, nbits):
    x = x.floor()
    if nbits < 32:
        max_val = 2**(nbits-1)-1
        min_val = -2**(nbits-1)+1
        total_range = max_val - min_val + 1
        mask = x>max_val
        mask = mask.float()
        x    = max_val*mask + (1-mask)*x
        mask = x<min_val
        mask = mask.float()
        x    = min_val*mask + (1-mask)*x
    return x

class IntNN(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, ibits):
        ctx.save_for_backward(x)
        # print np.max(x)
        # import ipdb as pdb; pdb.set_trace()
        return to_integer(x, ibits)

    @staticmethod
    def backward(ctx, dx):
        return dx, None

int_nn = IntNN.apply

class FixedNN(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, ibits, fbits, type=None):
        # import ipdb as pdb; pdb.set_trace()
        ctx.save_for_backward(x)
        return to_fixed_point(x, ibits, fbits)
        # return to_fixed_point(to_nearest_power_of_two(x), ibits, fbits)

    @staticmethod
    def backward(ctx, dx):
        #import ipdb as pdb; pdb.set_trace()
        return dx, None, None, None

fixed_nn = FixedNN.apply



class BNNSign(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        # import ipdb as pdb; pdb.set_trace()
        ctx.save_for_backward(x)
        return x.sign()
    
    @staticmethod
    def backward(ctx, dx):
        # import ipdb as pdb; pdb.set_trace()
        x, = ctx.saved_variables
        gt1  = x > +1
        lsm1 = x < -1
        gi   = 1-gt1.float()-lsm1.float()
        return gi*dx
bnn_sign = BNNSign.apply


# ternary weight approximation according to https://arxiv.org/abs/1605.04711
class Ternary(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        # import ipdb as pdb; pdb.set_trace()
        ctx.save_for_backward(x)
        w_in = x
        a,b,c,d = w_in.size()
        delta = 0.7*torch.mean(torch.mean(torch.mean(torch.abs(w_in),dim=3),dim=2),dim=1).view(-1,1,1,1)
        alpha = torch.abs(w_in)*(torch.abs(w_in)>delta).float()
        alpha = (torch.sum(torch.sum(torch.sum(alpha,dim=3),dim=2),dim=1)  \
        /torch.sum(torch.sum(torch.sum((alpha>0).float(),dim=3),dim=2),dim=1)).view(-1,1,1,1)
        w_out = -(w_in<-delta).float()*alpha + (w_in>delta).float()*alpha
        return w_out
    
    @staticmethod
    def backward(ctx, dx):
        return dx, None

ternary_q = Ternary.apply



