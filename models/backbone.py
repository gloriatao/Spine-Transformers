"""
Backbone modules.
"""
import torch
import torch.nn.functional as F
from torch import nn
from typing import Dict, List
from models.resnet_no_pool import generate_model
from util.misc import NestedTensor
from .position_encoding import build_position_encoding

class Backbone(nn.Module):

    def __init__(self, num_channels: int, model_depth: int):
        super().__init__()
        self.body = generate_model(model_depth)
        self.num_channels = num_channels

    def forward(self, tensor_list: NestedTensor):
        layer1_x, layer2_x, layer3_x, layer4_x = self.body(tensor_list.tensors)
        xs = {'0':layer1_x, '1':layer2_x, '2':layer3_x, '3':layer4_x}
        out: Dict[str, NestedTensor] = {}
        for name, x in xs.items():
            m = tensor_list.mask
            assert m is not None
            mask = F.interpolate(m[None].float(), size=x.shape[-3:]).to(torch.bool)[0]
            out[name] = NestedTensor(x, mask)
        return out


class Joiner(nn.Sequential):
    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)

    def forward(self, tensor_list: NestedTensor):
        xs = self[0](tensor_list)
        out: List[NestedTensor] = []
        pos = []
        for name, x in xs.items():
            out.append(x)
            pos.append(self[1](x).to(x.tensors.dtype))
        return out, pos


def build_backbone(args):
    position_embedding = build_position_encoding(args)
    backbone = Backbone(num_channels=2048, model_depth=50)
    model = Joiner(backbone, position_embedding)
    model.num_channels = backbone.num_channels
    return model
