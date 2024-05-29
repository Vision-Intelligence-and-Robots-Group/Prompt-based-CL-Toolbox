"""
    This main code is from JH-LEE-KR's GitHub repository
    Ref: https://github.com/JH-LEE-KR/dualprompt-pytorch
    Author: JH-LEE-KR
    Implements the DualP method using PyTorch.
"""
import torch
import torch.nn as nn
from functools import partial
from models.vision_transformer import VisionTransformer,PatchEmbed,Attention
from timm.models.helpers import checkpoint_seq
from models.vision_transformer import resolve_pretrained_cfg, build_model_with_cfg, checkpoint_filter_fn
from models.vision_transformer import LayerScale, DropPath, Mlp

from prompts.dualp_prompts import Dualp_E_Prompt as EPrompt

def _create_vision_transformer(variant, pretrained=False, **kwargs):
    if kwargs.get('features_only', None):
        raise RuntimeError('features_only not implemented for Vision Transformer models.')

    pretrained_cfg = resolve_pretrained_cfg(variant, pretrained_cfg=kwargs.pop('pretrained_cfg', None))
    model = build_model_with_cfg(
        ViT_Prompts, variant, pretrained,
        pretrained_cfg=pretrained_cfg,
        pretrained_filter_fn=checkpoint_filter_fn,
        pretrained_custom_load='npz' in pretrained_cfg['url'],
        **kwargs)
    return model

class Block(nn.Module):

    def __init__(
            self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0., init_values=None,
            drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, attn_layer=Attention):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = attn_layer(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x, prompt=None):
        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x), prompt)))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x
    
class ViT_Prompts(VisionTransformer):
    def __init__(
        self, img_size=224, patch_size=16, in_chans=3, num_classes=100, global_pool='token',
        embed_dim=768, depth=12, num_heads=12, mlp_ratio=4., qkv_bias=True,
        drop_rate=0., attn_drop_rate=0., drop_path_rate=0., weight_init='',
        init_values=None, embed_layer=PatchEmbed, norm_layer=None, act_layer=None, block_fn=Block, 
        no_embed_class=False, pre_norm=False, fc_norm=None,
        class_token=True, prompt_length=None, embedding_key='cls', prompt_init='uniform', prompt_pool=False, prompt_key=False, pool_size=None,
        top_k=None, batchwise_prompt=False, prompt_key_init='uniform', head_type='token', use_prompt_mask=False,
        use_g_prompt=False, g_prompt_length=None, g_prompt_layer_idx=None, use_prefix_tune_for_g_prompt=False,
        use_e_prompt=False, e_prompt_layer_idx=None, use_prefix_tune_for_e_prompt=False, same_key_value=False,
):

        super(ViT_Prompts,self).__init__(img_size=img_size, patch_size=patch_size, in_chans=in_chans, num_classes=num_classes, global_pool=global_pool,
            embed_dim=embed_dim, depth=depth, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
            drop_rate=drop_rate, attn_drop_rate=attn_drop_rate, drop_path_rate=drop_path_rate, weight_init=weight_init, init_values=init_values,
            embed_layer=embed_layer, norm_layer=norm_layer, act_layer=act_layer, block_fn=block_fn,
            no_embed_class=no_embed_class, pre_norm=pre_norm, fc_norm=fc_norm)
        
        use_fc_norm = global_pool == 'avg' if fc_norm is None else fc_norm
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        self.img_size = img_size
        self.num_classes = num_classes
        self.global_pool = global_pool
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.class_token = class_token
        self.num_prefix_tokens = 1 if class_token else 0
        self.no_embed_class = no_embed_class
        self.grad_checkpointing = False

        self.patch_embed = embed_layer(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if class_token else None
        embed_len = num_patches if no_embed_class else num_patches + self.num_prefix_tokens
        self.pos_embed = nn.Parameter(torch.randn(1, embed_len, embed_dim) * .02)
        self.pos_drop = nn.Dropout(p=drop_rate)

        self.prompt_pool = prompt_pool
        self.head_type = head_type
        self.use_prompt_mask = use_prompt_mask

        self.use_g_prompt = use_g_prompt
        self.g_prompt_layer_idx = g_prompt_layer_idx

        num_g_prompt = len(self.g_prompt_layer_idx) if self.g_prompt_layer_idx is not None else 0
        self.use_prefix_tune_for_g_prompt = use_prefix_tune_for_g_prompt
        
        self.use_e_prompt = use_e_prompt
        self.e_prompt_layer_idx = e_prompt_layer_idx
        num_e_prompt = len(self.e_prompt_layer_idx) if self.e_prompt_layer_idx is not None else 0
        self.use_prefix_tune_for_e_prompt = use_prefix_tune_for_e_prompt
        
        if not self.use_prefix_tune_for_g_prompt and not self.use_prefix_tune_for_g_prompt:
            self.use_g_prompt = False
            self.g_prompt_layer_idx = []

        if use_g_prompt and g_prompt_length is not None and len(g_prompt_layer_idx) != 0:
            if not use_prefix_tune_for_g_prompt:
                g_prompt_shape=(num_g_prompt, g_prompt_length, embed_dim)
                if prompt_init == 'zero':
                    self.g_prompt = nn.Parameter(torch.zeros(g_prompt_shape))
                elif prompt_init == 'uniform':
                    self.g_prompt = nn.Parameter(torch.randn(g_prompt_shape))
                    nn.init.uniform_(self.g_prompt, -1, 1)
            else:
                if same_key_value:
                    g_prompt_shape=(num_g_prompt, 1, g_prompt_length, num_heads, embed_dim // num_heads)
                    if prompt_init == 'zero':
                        self.g_prompt = nn.Parameter(torch.zeros(g_prompt_shape))
                    elif prompt_init == 'uniform':
                        self.g_prompt = nn.Parameter(torch.randn(g_prompt_shape))
                        nn.init.uniform_(self.g_prompt, -1, 1)
                    self.g_prompt = self.g_prompt.repeat(1, 2, 1, 1, 1)
                else:
                    g_prompt_shape=(num_g_prompt, 2, g_prompt_length, num_heads, embed_dim // num_heads)
                    if prompt_init == 'zero':
                        self.g_prompt = nn.Parameter(torch.zeros(g_prompt_shape))
                    elif prompt_init == 'uniform':
                        self.g_prompt = nn.Parameter(torch.randn(g_prompt_shape))
                        nn.init.uniform_(self.g_prompt, -1, 1)
        else:
            self.g_prompt = None
        
        if use_e_prompt and e_prompt_layer_idx is not None:
            self.e_prompt = EPrompt(length=prompt_length, embed_dim=embed_dim, embedding_key=embedding_key, prompt_init=prompt_init,
                    prompt_pool=prompt_pool, prompt_key=prompt_key, pool_size=pool_size, top_k=top_k, batchwise_prompt=batchwise_prompt,
                    prompt_key_init=prompt_key_init, num_layers=num_e_prompt, use_prefix_tune_for_e_prompt=use_prefix_tune_for_e_prompt,
                    num_heads=num_heads, same_key_value=same_key_value)
        
        if not (use_g_prompt or use_e_prompt):
            attn_layer = Attention
        elif not (use_prefix_tune_for_g_prompt or use_prefix_tune_for_e_prompt):
            # Prompt tunning
            attn_layer = Attention
        else:
            # Prefix tunning
            attn_layer = PreT_Attention
        
        self.total_prompt_len = 0
        if self.prompt_pool:
            if not self.use_prefix_tune_for_g_prompt:
                self.total_prompt_len += g_prompt_length * len(self.g_prompt_layer_idx)
            if not self.use_prefix_tune_for_e_prompt:
                self.total_prompt_len += prompt_length * top_k * len(self.e_prompt_layer_idx)
        
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.Sequential(*[
            block_fn(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, init_values=init_values,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer, attn_layer=attn_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim) if not use_fc_norm else nn.Identity()

        # Classifier Head
        self.fc_norm = norm_layer(embed_dim) if use_fc_norm else nn.Identity()
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        if weight_init != 'skip':
            self.init_weights(weight_init)

    def forward_features(self, x, task_id=-1, cls_features=None, train=False):
        x = self.patch_embed(x)

        if self.cls_token is not None:
            x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        
        x = self.pos_drop(x + self.pos_embed)

        if self.grad_checkpointing and not torch.jit.is_scripting():
            x = checkpoint_seq(self.blocks, x)
        else:
            if self.use_g_prompt or self.use_e_prompt:
                if self.use_prompt_mask and train:
                    start = task_id * self.e_prompt.top_k
                    end = (task_id + 1) * self.e_prompt.top_k
                    single_prompt_mask = torch.arange(start, end).to(x.device)
                    prompt_mask = single_prompt_mask.unsqueeze(0).expand(x.shape[0], -1)
                    if end > self.e_prompt.pool_size:
                        prompt_mask = None
                else:
                    prompt_mask = None
                
                g_prompt_counter = -1
                e_prompt_counter = -1

                res = self.e_prompt(x, prompt_mask=prompt_mask, cls_features=cls_features)
                e_prompt = res['batched_prompt']

                for i, block in enumerate(self.blocks):
                    if i in self.g_prompt_layer_idx:
                        if self.use_prefix_tune_for_g_prompt:
                            g_prompt_counter += 1
                            # Prefix tunning, [B, 2, g_prompt_length, num_heads, embed_dim // num_heads]
                            idx = torch.tensor([g_prompt_counter] * x.shape[0]).to(x.device)
                            g_prompt = self.g_prompt[idx]
                        else:
                            g_prompt=None
                        x = block(x, prompt=g_prompt)
                    
                    elif i in self.e_prompt_layer_idx:
                        e_prompt_counter += 1
                        if self.use_prefix_tune_for_e_prompt:
                            # Prefix tunning, [B, 2, top_k * e_prompt_length, num_heads, embed_dim // num_heads]
                            x = block(x, prompt=e_prompt[e_prompt_counter])
                        else:
                            # Pommpt tunning, [B, top_k * e_prompt_length, embed_dim]
                            prompt = e_prompt[e_prompt_counter]
                            x = torch.cat([prompt, x], dim=1)
                            x = block(x)
                    else:
                        x = block(x)
            else:
                x = self.blocks(x)            
                res = dict()
        x = self.norm(x)
        res['x'] = x

        return res

    def forward_head(self, res, pre_logits: bool = False):
        x = res['x']
        if self.class_token and self.head_type == 'token':
            if self.prompt_pool:
                x = x[:, self.total_prompt_len]
            else:
                x = x[:, 0]
        elif self.head_type == 'gap' and self.global_pool == 'avg':
            x = x.mean(dim=1)
        elif self.head_type == 'prompt' and self.prompt_pool:
            x = x[:, 1:(1 + self.total_prompt_len)] if self.class_token else x[:, 0:self.total_prompt_len]
            x = x.mean(dim=1)
        elif self.head_type == 'token+prompt' and self.prompt_pool and self.class_token:
            x = x[:, 0:self.total_prompt_len + 1]
            x = x.mean(dim=1)
        else:
            raise ValueError(f'Invalid classifier={self.classifier}')
        
        res['pre_logits'] = x
        x = self.fc_norm(x)
        res['logits'] = self.head(x)
        
        return res

    def forward(self, x, task_id=-1, cls_features=None, train=False):
        res = self.forward_features(x, task_id=task_id, cls_features=cls_features, train=train)
        res = self.forward_head(res)
        return res

class PreT_Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, prompt):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)   # make torchscript happy (cannot use tensor as tuple)

        if prompt is not None:
            # prefix key, value
            prompt = prompt.permute(1, 0, 3, 2, 4).contiguous() # 2, B, num_heads, prompt_length, C // num_heads
            key_prefix = prompt[0] # B, num_heads, prompt_length, embed_dim // num_heads
            value_prefix = prompt[1] # B, num_heads, prompt_length, embed_dim // num_heads

            expected_shape = (B, self.num_heads, C // self.num_heads)
            
            assert (key_prefix.shape[0], key_prefix.shape[1], key_prefix.shape[3]) == expected_shape, f'key_prefix.shape: {key_prefix.shape} not match k.shape: {k.shape}'
            assert (value_prefix.shape[0], value_prefix.shape[1], value_prefix.shape[3]) == expected_shape, f'value_prefix.shape: {value_prefix.shape} not match v.shape: {v.shape}'

            k = torch.cat([key_prefix, k], dim=2)
            v = torch.cat([value_prefix, v], dim=2)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
    