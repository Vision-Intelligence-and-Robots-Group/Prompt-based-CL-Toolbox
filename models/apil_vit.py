import copy
import torch
import torch.nn as nn
import timm
from torch.nn import functional as F
from models.linears import SimpleLinear
from timm.models.helpers import checkpoint_seq
from models.vision_transformer import _create_vision_transformer

class APrompt(nn.Module):

    def __init__(self, args):
        super(APrompt, self).__init__()        
        model_kwargs = dict(num_classes=0, global_pool='token')
        self.encoder =_create_vision_transformer(args.pretrained_model, pretrained=args.pretrained, **model_kwargs)
        
        self.numtask = 0
        self.fc = None

    @property
    def feature_dim(self):
        return self.encoder.embed_dim

    def extract_vector(self, image):
        image_features = self.encoder(image)
        image_features = F.normalize(image_features, p=2, dim=-1)
        return image_features

    def forward_with_anchor(self, image, anchor):
        x = self.encoder.patch_embed(image)
        x = self.encoder._pos_embed(x)
        if anchor.dim() == 2:
            anchor = anchor.expand(x.shape[0], -1, -1)
        x =  torch.cat([x, anchor], dim=1)  # concact anchor at the tail
        if self.encoder.grad_checkpointing and not torch.jit.is_scripting():
            x = checkpoint_seq(self.encoder.blocks, x)
        else:
            x = self.encoder.blocks(x)
        x = self.encoder.norm(x)
        x = self.encoder.forward_head(x)

        logits = self.fc(x)['logits']
        fea = F.normalize(x, p=2, dim=-1)

        return {
            'logits': logits,
            'fea': fea
        }

    def forward(self, image):
        image_features = self.encoder(image)
        logits = self.fc(image_features)['logits']

        return {
            'logits': logits,
            'fea': F.normalize(image_features, p=2, dim=-1)
        }

    def update_fc(self, nb_classes):
        fc = self.generate_fc(self.feature_dim, nb_classes)
        if self.fc is not None:
            nb_output = self.fc.out_features
            weight = copy.deepcopy(self.fc.weight.data)
            bias = copy.deepcopy(self.fc.bias.data)
            fc.weight.data[:nb_output] = weight
            fc.bias.data[:nb_output] = bias
        del self.fc
        self.fc = fc

    def generate_fc(self, in_dim, out_dim):
        fc = SimpleLinear(in_dim, out_dim)
        return fc

    def copy(self):
        return copy.deepcopy(self)

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False
        self.eval()

        return self

