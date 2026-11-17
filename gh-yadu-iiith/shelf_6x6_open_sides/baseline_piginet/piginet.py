"""PIGINet (Yang, Garrett, Lozano-Perez, Kaelbling, Fox: "Sequence-Based Plan Feasibility Prediction for Efficient Task and
Motion Planning", RSS 2023) on the 6x6 shelf dataset.

NETWORK. The authors never released the model code (it lives in their private `fastamp` module); kitchen-worlds ships only
the planner and the data generation. Sajid Ansari recovered the architecture 1:1 from the authors' released checkpoints
(`piginet_models/models/*_best.pt` + wandb config num_layers=4, num_heads=2, token_mode=add) and strict-load-verified it
(icra/train/v3/exact_pigi.py on the lab server; used in arXiv 2611.09130). The module below keeps those layer names and shapes:
    img_emb   Linear 512->768, ReLU, 768->512, ReLU, 512->256     (frozen CLIP ViT-B/32 image feature of the object's crop)
    text_emb  Linear 512->256                                      (frozen CLIP text feature of the token's phrase)
    value_emb Linear 15->256                                       (continuous values + one-hot value type, zero padded)
    pe_init, pe_goal  learned (256,);  ln_pre;  4 x CLIP residual attention block (d 256, 2 heads, MLP 1024, QuickGELU);
    ln_post;  linear 256->1 on the FIRST position (a plan token);  loss = positive-weighted BCE.

SEQUENCE (paper Sec. IV):  [plan actions] + [goal literals] + [initial-state literals].  Each element of a token (text, object
image, continuous value) is embedded to 256 and the elements are fused (released config: added); plan tokens get a sinusoidal
encoding of their step, goal / initial tokens a learned one each.

OUR TASK (documented adaptations; Sajid's GraspNet port had a single action, ours has the paper's multi-step plans):
    plan     one token per pick-and-place: phrase + image of the moved object + value(its current pose and shape) + value(destination)
    goal     at(T, goal cell): phrase + image of T + value(goal location)
    initial  one token per object: phrase + image + value(pose and shape)
  Values (15): [0:3] one-hot type (object pose / destination / goal), [3:6] x, y, z in [-1, 1] over the shelf box, [6] yaw (0),
  [7] size, [8] height, [9] shape (+1 cylinder, -1 box), rest zero.  PIGINet's value carries the pose only (shape comes from the
  image); we add the exact size / height / shape, which can only help the baseline.
"""
import math
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn

D, VALDIM = 256, 15
PHRASES = ["pick up this obstacle and place it at another location on the shelf",      # 0 plan: relocation
           "pick up the target object and place it at its goal location",              # 1 plan: target move
           "the target object stands at this location on the shelf",                   # 2 goal
           "an obstacle standing on the shelf",                                        # 3 initial: obstacle
           "the target object standing on the shelf"]                                  # 4 initial: target
K_REL, K_TGT, K_GOAL, K_OBS, K_T0 = range(5)
MAX_PLAN, MAX_OBJ = 8, 14
T_MAX = MAX_PLAN + 1 + MAX_OBJ

# shelf geometry (root files: geometry.x0 / pitch / table_z, 6 x 6 cells)
X0, PITCH, TABLE_Z, ROWS, COLS = 0.42, 0.075, 0.55, 6, 6
LO = np.array([X0 - 3 * PITCH, -3 * PITCH, TABLE_Z], np.float32)
HI = np.array([X0 + 3 * PITCH, 3 * PITCH, TABLE_Z + 0.20], np.float32)


def cell_xy(cell):
    r, c = cell
    return X0 + (r - 2.5) * PITCH, (c - 2.5) * PITCH


def value(vtype, cell, obj=None, height=None):
    """15-d value element. obj given: its size / height / shape are included; height: z of the centre for a destination."""
    v = np.zeros(VALDIM, np.float32)
    v[vtype] = 1.0
    x, y = cell_xy(cell)
    h = obj["height"] if obj is not None else (height or 0.0)
    v[3:6] = np.clip(2 * (np.array([x, y, TABLE_Z + h / 2], np.float32) - LO) / (HI - LO) - 1, -1, 1)
    if obj is not None:
        v[7] = obj["size"] / 0.02 - 1
        v[8] = obj["height"] / 0.10 - 1
        v[9] = 1.0 if obj["shape"] == "cylinder" else -1.0
    return v


def tokens(sample, feats=None, rng=None):
    """sample (build_data.py) -> arrays of one sequence.  feats: {object id: 512-d CLIP image feature} of the sample's scene
    state, or None (no-image variant).  rng: shuffles the initial-state tokens (training)."""
    objs = {o["id"]: o for o in sample["objects"]}
    cells = sample["cells"]
    img, v1, v2, kind, step = [], [], [], [], []
    Z = np.zeros(512, np.float32)
    zv = np.zeros(VALDIM, np.float32)

    def f(o):
        return np.asarray(feats[o], np.float32) if feats is not None and o in feats else Z

    for n, (o, src, dst) in enumerate(sample["plan"][:MAX_PLAN]):
        img.append(f(o)); v1.append(value(0, src, objs[o])); v2.append(value(1, dst, height=objs[o]["height"]))
        kind.append(K_TGT if o == "t0" else K_REL); step.append(n)
    img.append(f("t0")); v1.append(value(2, sample["goal"], height=objs["t0"]["height"])); v2.append(zv); kind.append(K_GOAL); step.append(0)
    ids = list(objs)
    if rng is not None:
        rng.shuffle(ids)
    for o in ids[:MAX_OBJ]:
        img.append(f(o)); v1.append(value(0, cells[o], objs[o])); v2.append(zv)
        kind.append(K_T0 if o == "t0" else K_OBS); step.append(0)
    return (np.stack(img), np.stack(v1), np.stack(v2), np.array(kind, np.int64), np.array(step, np.int64))


def collate(items):
    """items: list of (tokens(...), y).  Pads to the longest sequence of the batch."""
    T = max(len(t[3]) for t, _ in items)
    B = len(items)
    img = np.zeros((B, T, 512), np.float32); v1 = np.zeros((B, T, VALDIM), np.float32); v2 = np.zeros((B, T, VALDIM), np.float32)
    kind = np.zeros((B, T), np.int64); step = np.zeros((B, T), np.int64); mask = np.zeros((B, T), bool)
    for b, ((i, a, c, k, s), _) in enumerate(items):
        n = len(k)
        img[b, :n], v1[b, :n], v2[b, :n], kind[b, :n], step[b, :n], mask[b, :n] = i, a, c, k, s, True
    y = np.array([y for _, y in items], np.float32)
    return {k: torch.from_numpy(v) for k, v in dict(img=img, v1=v1, v2=v2, kind=kind, step=step, mask=mask, y=y).items()}


# ------------------------------------------------------------------ the network (names / shapes = the released state_dict)
class QuickGELU(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.attn = nn.MultiheadAttention(d, h, batch_first=True)
        self.ln_1 = nn.LayerNorm(d)
        self.mlp = nn.Sequential(OrderedDict([("c_fc", nn.Linear(d, d * 4)), ("gelu", QuickGELU()), ("c_proj", nn.Linear(d * 4, d))]))
        self.ln_2 = nn.LayerNorm(d)

    def forward(self, x, kpm):
        h = self.ln_1(x)
        a, _ = self.attn(h, h, h, key_padding_mask=kpm, need_weights=False)
        x = x + a
        return x + self.mlp(self.ln_2(x))


class Transformer(nn.Module):
    def __init__(self, d, layers, h):
        super().__init__()
        self.resblocks = nn.ModuleList([ResidualAttentionBlock(d, h) for _ in range(layers)])

    def forward(self, x, kpm):
        for b in self.resblocks:
            x = b(x, kpm)
        return x


def sinusoid(n, d):
    pe = torch.zeros(n, d)
    pos = torch.arange(n, dtype=torch.float32)[:, None]
    div = torch.exp(torch.arange(0, d, 2, dtype=torch.float32) * (-math.log(10000.0) / d))
    pe[:, 0::2], pe[:, 1::2] = torch.sin(pos * div), torch.cos(pos * div)
    return pe


class PIGINet(nn.Module):
    def __init__(self, d=D, layers=4, heads=2, use_image=True, use_value=True):
        super().__init__()
        self.img_emb = nn.Sequential(nn.Linear(512, 768), nn.ReLU(), nn.Linear(768, 512), nn.ReLU(), nn.Linear(512, d))
        self.text_emb = nn.Sequential(nn.Linear(512, d))
        self.value_emb = nn.Sequential(nn.Linear(VALDIM, d))
        self.pe_init = nn.Parameter(torch.zeros(d))
        self.pe_goal = nn.Parameter(torch.zeros(d))
        self.ln_pre = nn.LayerNorm(d)
        self.transformer = Transformer(d, layers, heads)
        self.ln_post = nn.LayerNorm(d)
        self.linear = nn.Linear(d, 1)
        self.use_image, self.use_value = use_image, use_value
        self.register_buffer("txt", torch.zeros(len(PHRASES), 512), persistent=False)      # frozen CLIP text features
        self.register_buffer("pe_step", sinusoid(MAX_PLAN, d), persistent=False)           # plan-step encoding (paper)

    def set_text(self, t):
        self.txt = torch.as_tensor(np.asarray(t), dtype=torch.float32, device=self.txt.device)

    def forward(self, img, v1, v2, kind, step, mask):
        tok = self.text_emb(self.txt[kind])
        if self.use_image:
            tok = tok + self.img_emb(img)
        if self.use_value:
            plan = (kind <= K_TGT).float()[..., None]
            tok = tok + self.value_emb(v1) + self.value_emb(v2) * plan                    # a move has two values: from, to
        plan = (kind <= K_TGT).float()[..., None]
        goal = (kind == K_GOAL).float()[..., None]
        init = (kind >= K_OBS).float()[..., None]
        tok = tok + plan * self.pe_step[step] + goal * self.pe_goal + init * self.pe_init
        x = self.transformer(self.ln_pre(tok), ~mask.bool())
        return self.linear(self.ln_post(x[:, 0])).squeeze(-1)
