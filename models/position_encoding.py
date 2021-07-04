import torch
from torch import nn
from util.misc import NestedTensor

class PositionEmbeddingLearned(nn.Module):
    """
    Absolute pos embedding, learned.
    """
    def __init__(self, num_pos_feats=128):
        super().__init__()
        self.row_embed = nn.Embedding(50, num_pos_feats)
        self.col_embed = nn.Embedding(50, num_pos_feats)
        self.depth_embed = nn.Embedding(50, num_pos_feats)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.row_embed.weight)
        nn.init.uniform_(self.col_embed.weight)
        nn.init.uniform_(self.depth_embed.weight)

    def forward(self, tensor_list: NestedTensor):
        x = tensor_list.tensors
        h, w, d = x.shape[-3:]
        i = torch.arange(w, device=x.device)
        j = torch.arange(h, device=x.device)
        k = torch.arange(d, device=x.device)
        x_emb = self.col_embed(i)
        y_emb = self.row_embed(j)
        z_emb = self.depth_embed(k)
        pos = torch.cat([
            x_emb.unsqueeze(0).unsqueeze(2).repeat(h, 1, d, 1),
            y_emb.unsqueeze(1).unsqueeze(1).repeat(1, w, d, 1),
            z_emb.unsqueeze(0).unsqueeze(0).repeat(h, w, 1, 1),
        ], dim=-1).permute(3, 0, 1, 2).unsqueeze(0).repeat(x.shape[0], 1, 1, 1, 1)
        return pos

def build_position_encoding(args):
    N_steps = args.hidden_dim // 3
    position_embedding = PositionEmbeddingLearned(N_steps)
    return position_embedding
