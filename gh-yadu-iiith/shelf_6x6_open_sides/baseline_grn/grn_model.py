"""Geometric Reasoning Network (Ait Bouhsain, Alami, Simeon, ICLR 2025), architecture as in the authors' code
(github.com/Smail8/geometric_reasoning_networks, MIT; Network.py), with the number of grasp views V as a parameter (the paper's
Panda scenes use 5 views, the shelf has 16 grasp azimuths) and no change otherwise:

  IKModule   MLP 7 -> 512 -> 512 -> 512 -> V  on the movable node (l, w, h, x, y, z, yaw)          per-view IK reachability (BCE)
  GOModule   MLP 14 -> 512 x3 -> V, hardtanh [0, 1], on (movable node, neighbour node)              per-view grasp-obstruction ratio (MSE)
  AGFModule  node / edge encoders (7 -> 256, 2 + V -> 256), one EdgeGATv2Conv (256, 4 heads, edge features), decoder
             1024 -> 256 -> 1 + V                                                                    action + per-view feasibility (BCE)
  GRN        IK -> mask the GO outputs by the predicted reachability -> GO predictions go on the edges ->
             AGF reads the graph (edge attribute = [proximity, self-loop, V obstruction ratios]); non-blocking (self-loop) edges
             carry 1 - IK.
Batches are plain tensors (see `collate`): x [N,7], mask [N] (movable), edge_index [2,E], edge_attr [E,2], blocking [E].
"""
import numpy as np
import torch
import torch.nn as nn

from base_gnn import EdgeGATv2Conv


def mlp(n_in, hidden, n_out, depth=3):
    layers, d = [], n_in
    for _ in range(depth):
        layers += [nn.Linear(d, hidden), nn.ReLU()]
        d = hidden
    return nn.Sequential(*layers), nn.Linear(hidden, n_out)


def wrap_yaw(x, cols):
    x = x.clone()
    for c in cols:
        x[:, c] = x[:, c] % (2 * np.pi)
    return x


class IKModule(nn.Module):
    def __init__(self, V, num_features=7, hidden=512):
        super().__init__()
        self.encoder, self.decoder = mlp(num_features, hidden, V)

    def forward(self, x):                                   # logits
        return self.decoder(self.encoder(wrap_yaw(x, [6])))


class GOModule(nn.Module):
    def __init__(self, V, num_features=14, hidden=512):
        super().__init__()
        self.encoder, self.decoder = mlp(num_features, hidden, V)

    def forward(self, x, mask=None):
        pred = nn.functional.hardtanh(self.decoder(self.encoder(wrap_yaw(x, [6, 13]))), min_val=0.0, max_val=1.0)
        return pred * mask if mask is not None else pred


class AGFModule(nn.Module):
    def __init__(self, V, num_node_features=7, hidden=256):
        super().__init__()
        self.node_encoder = nn.Sequential(nn.Linear(num_node_features, hidden), nn.ReLU())
        self.edge_encoder = nn.Sequential(nn.Linear(2 + V, hidden), nn.ReLU())
        self.conv = EdgeGATv2Conv(in_channels=hidden, out_channels=hidden, heads=4, edge_dim=hidden, add_self_loops=False)
        self.decoder = nn.Sequential(nn.Linear(hidden * 4, hidden), nn.ReLU(), nn.Linear(hidden, 1 + V))

    def forward(self, x, edge_index, edge_attr):             # logits
        h = self.conv(self.node_encoder(x), edge_index, self.edge_encoder(edge_attr))
        return self.decoder(h)


class GRN(nn.Module):
    def __init__(self, V=16):
        super().__init__()
        self.V = V
        self.IKModule, self.GOModule, self.AGFModule = IKModule(V), GOModule(V), AGFModule(V)

    def forward(self, b):
        """b: batch dict.  -> F logits [N, 1+V] (use the movable rows), IK logits [N_mov, V], GO [E_block, V]"""
        x = wrap_yaw(b["x"], [6])
        mask, blocking, ei = b["mask"], b["blocking"], b["edge_index"]
        ik_logit = self.IKModule(x[mask])
        ik_mask = torch.ones(x.shape[0], self.V, device=x.device)
        ik_mask[mask] = ik_logit.sigmoid()
        nb = ei[:, blocking]                                  # (neighbour, movable) pairs
        go_in = torch.cat((x[nb[1]], x[nb[0]]), dim=1)
        go = self.GOModule(go_in, ik_mask[nb[1]])
        edge_attr = torch.cat((b["edge_attr"], torch.zeros(ei.shape[1], self.V, device=x.device)), dim=1)
        edge_attr[blocking, 2:] = go
        edge_attr[~blocking, 2:] = 1.0 - ik_mask[mask]        # self loops: one per movable node
        return self.AGFModule(x, ei, edge_attr), ik_logit, go


def collate(graphs, device="cpu"):
    """list of graph dicts (build_graphs.py) -> one batch of tensors."""
    off, xs, masks, src, dst, ea, bl = 0, [], [], [], [], [], []
    F, Fm, IK, IKm, GO, GOm, mov, ys = [], [], [], [], [], [], [], []
    for g in graphs:
        n = len(g["names"])
        xs.append(g["x"]); masks.append(~g["fixed"])
        src.append(g["src"] + off); dst.append(g["dst"] + off); ea.append(g["eattr"]); bl.append(g["blocking"])
        F.append(g["F"]); Fm.append(g["Fm"]); IK.append(g["IK"]); IKm.append(g["IKm"])
        GO.append(g["go"][g["blocking"]]); GOm.append(g["gom"][g["blocking"]])
        mov.append(g["t"] + off); ys.append(g["y"])
        off += n
    t = lambda a, dt=None: torch.as_tensor(np.concatenate(a) if a[0].ndim else np.array(a), dtype=dt).to(device)  # noqa: E731
    return dict(x=t(xs, torch.float32), mask=t(masks, torch.bool), edge_index=torch.stack((t(src, torch.long), t(dst, torch.long))),
                edge_attr=t(ea, torch.float32), blocking=t(bl, torch.bool),
                F=torch.as_tensor(np.stack(F)).float().to(device), Fm=torch.as_tensor(np.stack(Fm)).float().to(device),
                IK=torch.as_tensor(np.stack(IK)).float().to(device), IKm=torch.as_tensor(np.stack(IKm)).float().to(device),
                GO=t(GO, torch.float32), GOm=t(GOm, torch.float32), y=torch.as_tensor(ys).float().to(device))
