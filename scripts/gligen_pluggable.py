from collections import OrderedDict
from inspect import isfunction
import torch
from torch import nn
from ldm.modules.attention import FeedForward, SpatialTransformer
from ldm.modules.diffusionmodules.openaimodel import UNetModel
import itertools
from natsort import natsorted
from easing_functions import *
import numpy as np
from modules import shared

easing_function_namelist = ['LinearInOut',
                            'QuadEaseInOut',
                            'QuadEaseIn',
                            'QuadEaseOut',
                            'CubicEaseInOut',
                            'CubicEaseIn',
                            'CubicEaseOut',
                            'QuarticEaseInOut',
                            'QuarticEaseIn',
                            'QuarticEaseOut',
                            'QuinticEaseInOut',
                            'QuinticEaseIn',
                            'QuinticEaseOut',
                            'SineEaseInOut',
                            'SineEaseIn',
                            'SineEaseOut',
                            'CircularEaseIn',
                            'CircularEaseInOut',
                            'CircularEaseOut',
                            'ExponentialEaseInOut',
                            'ExponentialEaseIn',
                            'ExponentialEaseOut',
                            'ElasticEaseIn',
                            'ElasticEaseInOut',
                            'ElasticEaseOut',
                            'BackEaseIn',
                            'BackEaseInOut',
                            'BackEaseOut',
                            'BounceEaseIn',
                            'BounceEaseInOut',
                            'BounceEaseOut',

                            ]

easing_function_list = [lambda x: x,
                        QuadEaseInOut(),
                        QuadEaseIn(),
                        QuadEaseOut(),
                        CubicEaseInOut(),
                        CubicEaseIn(),
                        CubicEaseOut(),
                        QuarticEaseInOut(),
                        QuarticEaseIn(),
                        QuarticEaseOut(),
                        QuinticEaseInOut(),
                        QuinticEaseIn(),
                        QuinticEaseOut(),
                        SineEaseInOut(),
                        SineEaseIn(),
                        SineEaseOut(),
                        CircularEaseIn(),
                        CircularEaseInOut(),
                        CircularEaseOut(),
                        ExponentialEaseInOut(),
                        ExponentialEaseIn(),
                        ExponentialEaseOut(),
                        ElasticEaseIn(),
                        ElasticEaseInOut(),
                        ElasticEaseOut(),
                        BackEaseIn(),
                        BackEaseInOut(),
                        BackEaseOut(),
                        BounceEaseIn(),
                        BounceEaseInOut(),
                        BounceEaseOut(),
                        ]


try:
    import xformers
    import xformers.ops
    XFORMERS_IS_AVAILBLE = True
except:
    XFORMERS_IS_AVAILBLE = False


def exists(val):
    return val is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


class SelfAttention(nn.Module):
    def __init__(self, query_dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        self.scale = dim_head ** -0.5
        self.heads = heads
        self.dim_head = dim_head

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(query_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(nn.Linear(inner_dim, query_dim), nn.Dropout(dropout) )

    def forward_plain(self, x):
        q = self.to_q(x) # B*N*(H*C)
        k = self.to_k(x) # B*N*(H*C)
        v = self.to_v(x) # B*N*(H*C)

        B, N, HC = q.shape
        H = self.heads
        C = HC // H

        q = q.view(B,N,H,C).permute(0,2,1,3).reshape(B*H,N,C) # (B*H)*N*C
        k = k.view(B,N,H,C).permute(0,2,1,3).reshape(B*H,N,C) # (B*H)*N*C
        v = v.view(B,N,H,C).permute(0,2,1,3).reshape(B*H,N,C) # (B*H)*N*C

        sim = torch.einsum('b i c, b j c -> b i j', q, k) * self.scale  # (B*H)*N*N
        attn = sim.softmax(dim=-1) # (B*H)*N*N

        out = torch.einsum('b i j, b j c -> b i c', attn, v) # (B*H)*N*C
        out = out.view(B,H,N,C).permute(0,2,1,3).reshape(B,N,(H*C)) # B*N*(H*C)

        return self.to_out(out)

    def forward(self, x, context=None, mask=None):
        if not XFORMERS_IS_AVAILBLE:
            return self.forward_plain(x)

        q = self.to_q(x)
        context = default(context, x)
        k = self.to_k(context)
        v = self.to_v(context)

        b, _, _ = q.shape
        q, k, v = map(
            lambda t: t.unsqueeze(3)
            .reshape(b, t.shape[1], self.heads, self.dim_head)
            .permute(0, 2, 1, 3)
            .reshape(b * self.heads, t.shape[1], self.dim_head)
            .contiguous(),
            (q, k, v),
        )

        # actually compute the attention, what we cannot get enough of
        out = xformers.ops.memory_efficient_attention(q, k, v, attn_bias=None, op=None)

        if exists(mask):
            raise NotImplementedError
        out = (
            out.unsqueeze(0)
            .reshape(b, self.heads, out.shape[1], self.dim_head)
            .permute(0, 2, 1, 3)
            .reshape(b, out.shape[1], self.heads * self.dim_head)
        )
        return self.to_out(out)


class CrossAttention(nn.Module):
    def __init__(self, query_dim, key_dim, value_dim, heads=8, dim_head=64, dropout=0):
        super().__init__()
        inner_dim = dim_head * heads
        self.scale = dim_head ** -0.5
        self.heads = heads
        self.dim_head = dim_head

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(key_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(value_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(nn.Linear(inner_dim, query_dim), nn.Dropout(dropout))

    def fill_inf_from_mask(self, sim, mask):
        if mask is not None:
            B, M = mask.shape
            mask = mask.unsqueeze(1).repeat(1, self.heads, 1).reshape(B * self.heads, 1, -1)
            max_neg_value = -torch.finfo(sim.dtype).max
            sim.masked_fill_(~mask, max_neg_value)
        return sim

    def forward_plain(self, x, key, value, mask=None):

        q = self.to_q(x)  # B*N*(H*C)
        k = self.to_k(key)  # B*M*(H*C)
        v = self.to_v(value)  # B*M*(H*C)

        B, N, HC = q.shape
        _, M, _ = key.shape
        H = self.heads
        C = HC // H

        q = q.view(B, N, H, C).permute(0, 2, 1, 3).reshape(B * H, N, C)  # (B*H)*N*C
        k = k.view(B, M, H, C).permute(0, 2, 1, 3).reshape(B * H, M, C)  # (B*H)*M*C
        v = v.view(B, M, H, C).permute(0, 2, 1, 3).reshape(B * H, M, C)  # (B*H)*M*C

        sim = torch.einsum('b i d, b j d -> b i j', q, k) * self.scale  # (B*H)*N*M
        self.fill_inf_from_mask(sim, mask)
        attn = sim.softmax(dim=-1)  # (B*H)*N*M

        out = torch.einsum('b i j, b j d -> b i d', attn, v)  # (B*H)*N*C
        out = out.view(B, H, N, C).permute(0, 2, 1, 3).reshape(B, N, (H * C))  # B*N*(H*C)

        return self.to_out(out)

    def forward(self, x, key, value, mask=None):
        if not XFORMERS_IS_AVAILBLE:
            return self.forward_plain(x, key, value, mask)

        q = self.to_q(x)  # B*N*(H*C)
        k = self.to_k(key)  # B*M*(H*C)
        v = self.to_v(value)  # B*M*(H*C)

        b, _, _ = q.shape
        q, k, v = map(
            lambda t: t.unsqueeze(3)
            .reshape(b, t.shape[1], self.heads, self.dim_head)
            .permute(0, 2, 1, 3)
            .reshape(b * self.heads, t.shape[1], self.dim_head)
            .contiguous(),
            (q, k, v),
        )

        # actually compute the attention, what we cannot get enough of
        out = xformers.ops.memory_efficient_attention(q, k, v, attn_bias=None, op=None)

        if exists(mask):
            raise NotImplementedError
        out = (
            out.unsqueeze(0)
            .reshape(b, self.heads, out.shape[1], self.dim_head)
            .permute(0, 2, 1, 3)
            .reshape(b, out.shape[1], self.heads * self.dim_head)
        )
        return self.to_out(out)

class GatedSelfAttentionDense(nn.Module):
    def __init__(self, query_dim, context_dim, n_heads, d_head):
        super().__init__()

        # we need a linear projection since we need cat visual feature and obj feature
        self.linear = nn.Linear(context_dim, query_dim)

        self.attn = SelfAttention(query_dim=query_dim, heads=n_heads, dim_head=d_head)
        self.ff = FeedForward(query_dim, glu=True)

        self.norm1 = nn.LayerNorm(query_dim)
        self.norm2 = nn.LayerNorm(query_dim)

        self.register_parameter('alpha_attn', nn.Parameter(torch.tensor(0.)))
        self.register_parameter('alpha_dense', nn.Parameter(torch.tensor(0.)))

        # this can be useful: we can externally change magnitude of tanh(alpha)
        # for example, when it is set to 0, then the entire model is same as original one
        self.scale = 1

    def forward(self, x, objs):
        N_visual = x.shape[1]
        objs = self.linear(objs)

        x = x + self.scale * torch.tanh(self.alpha_attn) * self.attn(self.norm1(torch.cat([x, objs], dim=1)))[:,
                                                           0:N_visual, :]
        x = x + self.scale * torch.tanh(self.alpha_dense) * self.ff(self.norm2(x))

        return x


# class BasicTransformerBlock(nn.Module):
#     def __init__(self, query_dim, key_dim, value_dim, n_heads, d_head, fuser_type, use_checkpoint=True):
#         super().__init__()
#         self.attn1 = SelfAttention(query_dim=query_dim, heads=n_heads, dim_head=d_head)
#         self.ff = FeedForward(query_dim, glu=True)
#         self.attn2 = CrossAttention(query_dim=query_dim, key_dim=key_dim, value_dim=value_dim, heads=n_heads, dim_head=d_head)
#         self.norm1 = nn.LayerNorm(query_dim)
#         self.norm2 = nn.LayerNorm(query_dim)
#         self.norm3 = nn.LayerNorm(query_dim)
#         self.use_checkpoint = use_checkpoint
#
#
#             # note key_dim here actually is context_dim
#         self.fuser = GatedSelfAttentionDense(query_dim, key_dim, n_heads, d_head)
#
#
#     def forward(self, x, context, objs):
#         x = self.attn1(self.norm1(x)) + x
#         x = self.fuser(x, objs)  # identity mapping in the beginning
#         x = self.attn2(self.norm2(x), context, context) + x
#         x = self.ff(self.norm3(x)) + x
#         return x


class FourierEmbedder():
    def __init__(self, num_freqs=64, temperature=100):

        self.num_freqs = num_freqs
        self.temperature = temperature
        self.freq_bands = temperature ** ( torch.arange(num_freqs) / num_freqs )

    @ torch.no_grad()
    def __call__(self, x, cat_dim=-1):
        "x: arbitrary shape of tensor. dim: cat dim"
        out = []
        for freq in self.freq_bands:
            out.append( torch.sin( freq*x ) )
            out.append( torch.cos( freq*x ) )
        return torch.cat(out, cat_dim)


class PositionNet(nn.Module):
    def __init__(self, positive_len, out_dim, fourier_freqs=8):
        super().__init__()
        self.positive_len = positive_len
        self.out_dim = out_dim

        self.fourier_embedder = FourierEmbedder(num_freqs=fourier_freqs)
        self.position_dim = fourier_freqs * 2 * 4  # 2 is sin&cos, 4 is xyxy

        self.linears = nn.Sequential(
            nn.Linear(self.positive_len + self.position_dim, 512),
            nn.SiLU(),
            nn.Linear(512, 512),
            nn.SiLU(),
            nn.Linear(512, out_dim),
        )

        self.null_positive_feature = torch.nn.Parameter(torch.zeros([self.positive_len]))
        self.null_position_feature = torch.nn.Parameter(torch.zeros([self.position_dim]))

    def forward(self, boxes, masks, positive_embeddings):
        B, N, _ = boxes.shape
        masks = masks.unsqueeze(-1)

        # embedding position (it may includes padding as placeholder)
        xyxy_embedding = self.fourier_embedder(boxes)  # B*N*4 --> B*N*C

        # learnable null embedding
        positive_null = self.null_positive_feature.view(1, 1, -1)
        xyxy_null = self.null_position_feature.view(1, 1, -1)

        # replace padding with learnable null embedding
        positive_embeddings = positive_embeddings * masks + (1 - masks) * positive_null
        xyxy_embedding = xyxy_embedding * masks + (1 - masks) * xyxy_null

        objs = self.linears(torch.cat([positive_embeddings, xyxy_embedding], dim=-1))
        assert objs.shape == torch.Size([B, N, self.out_dim])
        return objs

class ProxyBasicTransformerBlock(object):
    def __init__(self, controller, org_module: torch.nn.Module = None):
        super().__init__()
        self.org_module = org_module
        self.org_forward = None
        self.fuser = None
        self.attached = False
        self.controller = controller
        self.objs = None


    def __getattr__(self, attr):
        if attr not in ['org_module', 'org_forward', 'fuser', 'attached', 'controller', 'objs'] and self.attached:
            return getattr(self.org_module, attr)

    def initialize_fuser(self, fuser_state_dict):
        query_dim = self.org_module.attn1.to_q.in_features
        key_dim = self.org_module.attn2.to_k.in_features
        n_heads = self.org_module.attn1.heads
        d_head = int(self.org_module.attn2.to_q.out_features / n_heads)
        self.fuser = GatedSelfAttentionDense(query_dim, key_dim, n_heads, d_head)
        self.fuser.load_state_dict(fuser_state_dict)



    def apply_to(self):
        if self.org_forward is not None:
            return
        self.org_forward = self.org_module.forward
        self.org_module.forward = self.forward
        self.attached = True

    def detach(self):
        if self.org_forward is None:
            return
        self.org_module.forward = self.org_forward
        self.org_forward = None
        self.attached = False

    def forward(self, x, context):
        x = self.attn1( self.norm1(x) ) + x
        x = self.fuser(x,  self.controller.batch_objs_input) # identity mapping in the beginning
        x = self.attn2(self.norm2(x), context, context) + x
        x = self.ff(self.norm3(x)) + x
        return x


class ProxyUNetModel(object):
    def __init__(self, controller, org_module: torch.nn.Module = None):
        super().__init__()
        self.org_module = org_module
        self.org_forward = None
        self.attached = False
        self.controller = controller
        self.objs = None

    def __getattr__(self, attr):
        if attr not in ['org_module', 'org_forward', 'fuser', 'attached', 'controller', 'objs'] and self.attached:
            return getattr(self.org_module, attr)

    def apply_to(self):
        if self.org_forward is not None:
            return
        self.org_forward = self.org_module.forward
        self.org_module.forward = self.forward
        self.attached = True

    def detach(self):
        if self.org_forward is None:
            return
        self.org_module.forward = self.org_forward
        self.org_forward = None
        self.attached = False

    def forward(self, x, timesteps=None, context=None, y=None, **kwargs):
        self.controller.unet_signal(timesteps=timesteps, x=x)
        return self.org_forward(x, timesteps=timesteps, context=context, y=y, **kwargs)



known_block_prefixes = [
    'input_blocks.0.',
    'input_blocks.1.',
    'input_blocks.2.',
    'input_blocks.3.',
    'input_blocks.4.',
    'input_blocks.5.',
    'input_blocks.6.',
    'input_blocks.7.',
    'input_blocks.8.',
    'input_blocks.9.',
    'input_blocks.10.',
    'input_blocks.11.',
    'middle_block.',
    'output_blocks.0.',
    'output_blocks.1.',
    'output_blocks.2.',
    'output_blocks.3.',
    'output_blocks.4.',
    'output_blocks.5.',
    'output_blocks.6.',
    'output_blocks.7.',
    'output_blocks.8.',
    'output_blocks.9.',
    'output_blocks.10.',
    'output_blocks.11.',
]


def alpha_generator(length, type=[1, 0, 0]):
    """
    length is total timestpes needed for sampling.
    type should be a list containing three values which sum should be 1

    It means the percentage of three stages:
    alpha=1 stage
    linear deacy stage
    alpha=0 stage.

    For example if length=100, type=[0.8,0.1,0.1]
    then the first 800 stpes, alpha will be 1, and then linearly decay to 0 in the next 100 steps,
    and the last 100 stpes are 0.
    """

    assert len(type) == 3
    assert type[0] + type[1] + type[2] == 1

    stage0_length = int(type[0] * length)
    stage1_length = int(type[1] * length)
    stage2_length = length - stage0_length - stage1_length

    if stage1_length != 0:
        decay_alphas = np.arange(start=0, stop=1, step=1 / stage1_length)[::-1]
        decay_alphas = list(decay_alphas)
    else:
        decay_alphas = []

    alphas = [1] * stage0_length + decay_alphas + [0] * stage2_length

    assert len(alphas) == length

    return alphas


def timestep_to_alpha(timestep):
    timestep = timestep[0]
    if timestep > 800:
        return 1.0
    elif 500 < timestep < 800:
        # linear
        return (timestep - 500.0) / 300.0
    elif 0 <= timestep < 500:
        return 0.0


class PluggableGLIGEN:
    def __init__(self, ori_unet:UNetModel, gligen_state_dict):
        super().__init__()
        self.proxy_blocks=[]
        self.gated_self_attention_modules = []
        self.empty_objs = None
        self.objs=None
        self.batch_objs_input = None
        self.batch_size = 1
        gligen_state_dict_keys = natsorted(gligen_state_dict.keys())
        gligen_sorted_dict = OrderedDict({k: gligen_state_dict[k] for k in gligen_state_dict_keys})
        cur_gligen_dict_pointer = 0
        initial_state_dict_size = len(gligen_sorted_dict)
        for block_idx, unet_block in enumerate(itertools.chain(ori_unet.input_blocks, [ori_unet.middle_block], ori_unet.output_blocks)):
            cur_block_prefix = known_block_prefixes[block_idx]
            cur_block_fuse_state_dict = {key: value for key, value in gligen_sorted_dict.items() if key.startswith(cur_block_prefix)}


            if len(cur_block_fuse_state_dict) != 0:
                if len(cur_block_fuse_state_dict) != 17:
                    raise Exception('State dict for block {} is not correct'.format(cur_block_prefix))
                verify_cur_block_fuse_state_dict = dict(itertools.islice(gligen_sorted_dict.items(), 17))
                for key, value in verify_cur_block_fuse_state_dict.items():
                    if not key.startswith(cur_block_prefix):
                        raise Exception('State dict for block {} is not correct'.format(cur_block_prefix))
                    gligen_sorted_dict.popitem(last=False)

                # trim state_dict keys
                key_after_fuser_pointer = list(cur_block_fuse_state_dict.keys())[0].index('fuser.') + len('fuser.')
                cur_block_fuse_state_dict = {key[key_after_fuser_pointer:]: value for key, value in cur_block_fuse_state_dict.items()}


            else:
                continue


            for module in unet_block.modules():
                if type(module) is SpatialTransformer:
                    spatial_transformer = module
                    for basic_transformer_block in spatial_transformer.transformer_blocks:
                        cur_proxy_block = ProxyBasicTransformerBlock(self, basic_transformer_block)
                        cur_proxy_block.initialize_fuser(cur_block_fuse_state_dict)
                        # cur_proxy_block.apply_to()
                        self.gated_self_attention_modules.append(cur_proxy_block.fuser)
                        self.proxy_blocks.append(cur_proxy_block)

        verify_position_net_state_dict = gligen_sorted_dict
        for key, value in verify_position_net_state_dict.items():
            if not key.startswith('position_net.'):
                raise Exception('State dict for position_net is not correct')

        # trim state_dict keys
        key_after_position_net_pointer = list(gligen_sorted_dict.keys())[0].index('position_net.') + len(
            'position_net.')
        position_net_state_dict = {key[key_after_position_net_pointer:]: value for key, value in gligen_sorted_dict.items()}

        self.position_net = PositionNet(positive_len=768, out_dim=768)
        self.position_net.load_state_dict(position_net_state_dict)

        self.unet_proxy = ProxyUNetModel(self, ori_unet)
        # generate placeholder objs for unconditional generation
        max_objs = 30
        boxes = torch.zeros(max_objs, 4).unsqueeze(0)
        masks = torch.zeros(max_objs).unsqueeze(0)
        text_embeddings = torch.zeros(max_objs, 768).unsqueeze(0)
        self.empty_objs = self.position_net(boxes, masks, text_embeddings)


    def update_objs(self, boxes, masks, text_embeddings, batch_size):
        self.objs = self.position_net(boxes, masks, text_embeddings)
        self.batch_size = batch_size
        for module in self.gated_self_attention_modules:
            module.to(device=shared.device, dtype=shared.sd_model.dtype)

    def attach_all(self):
        for proxy_block in self.proxy_blocks:
            proxy_block.apply_to()
        self.unet_proxy.apply_to()

    def detach_all(self):
        for proxy_block in self.proxy_blocks:
            proxy_block.detach()
        self.unet_proxy.detach()

    def unet_signal(self, timesteps, x):
        calculated_alpha = timestep_to_alpha(timesteps)
        # calculated_alpha = torch.Tensor([calculated_alpha]).to(device=x.device, dtype=x.dtype)
        for module in self.gated_self_attention_modules:
            module.scale = calculated_alpha


        # repeat objs according to batch size-1 then append empty_objs for unconditional
        single_batch_slice_size = x.shape[0] // self.batch_size
        self.objs = self.objs.to(device=x.device, dtype=x.dtype)
        self.empty_objs = self.empty_objs.to(device=x.device, dtype=x.dtype)
        if single_batch_slice_size == 1:
            # dealing with unmatched cond and uncond situation
            #TODO: need indication of whether current batch is cond or uncond
            self.batch_objs_input = self.objs
            return



        self.batch_objs_input = torch.cat([self.objs.repeat(single_batch_slice_size - 1, 1, 1), self.empty_objs], dim=0)
        if self.batch_size != 1:
            self.batch_objs_input = self.batch_objs_input.repeat(self.batch_size, 1, 1)



