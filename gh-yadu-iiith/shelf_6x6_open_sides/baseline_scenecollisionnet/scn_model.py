"""SceneCollisionNet (Danielczuk, Mousavian, Eppner, Fox, ICRA 2021) for the shelf: the architecture of NVlabs/SceneCollisionNet
(collision_models/collision_nets.py) as ported and layer-by-layer verified by Sajid Ansari (lab server: ~/icra/train/v3/scn.py,
arXiv 2611.09130), made batch-capable and with the voxel grid set to the shelf.  Layers, sizes, initialisation (kaiming, no batch
norm) and the PointNet++ semantics are unchanged:

  scene     per-point Conv1d MLP [3, 128, 256] on voxel-centre-relative coordinates -> scatter-MAX into a voxel grid NV ->
            3D conv pyramid [256 -> 512 (k3, maxpool 2) -> 1024 (k3, maxpool 2) -> 512 (convT k2 s2)]; stack(level 1, last) =
            1024-d feature per scale-2 voxel
  object    PointNet++ set abstraction (npoint 256 / 64 / all, radii .02 / .04, nsample 64 / 128, mlps [3-64-128], [131-128-256],
            [259-256-512], no batch norm, xyz appended) -> FC 512-1024-1024   (the query body: the xArm7 hand, a constant cloud,
            so the FPS / ball-query indices are computed once)
  query     scene feature of the scale-2 voxel containing the translation (1024) + object feature (1024) + translation offset from
            the voxel centre (3) + 6D rotation = first two columns of R (6) = 2057 -> FC 1024 -> 256 -> 1  (collision logit)

Shelf adaptation: bounds / grid [8, 16, 4] over x 0.10-0.80, y -0.40-0.40, z 0.50-0.90 (voxels 8.75 x 5 x 10 cm); two queries per
grasp share the scene feature and the classifier: the hand at the grasp pose over the target's start ("pick") and the hand with the
same orientation over the goal ("place").  P(move works with this grasp) = (1 - p_pick) (1 - p_place).
"""
import numpy as np
import torch
import torch.nn as nn

BOUNDS = np.array([[0.10, -0.40, 0.50], [0.80, 0.40, 0.90]], np.float32)
NV = np.array([8, 16, 4])
VOX = ((BOUNDS[1] - BOUNDS[0]) / NV).astype(np.float32)
NV2 = NV // 2
SCENE_PT_MLP = [3, 128, 256]
CLS_FC = [2057, 1024, 256]
OBJ_NPOINTS = [256, 64, None]
OBJ_RADII = [0.02, 0.04, None]
OBJ_NSAMPLES = [64, 128, None]
OBJ_MLPS = [[0, 64, 128], [128, 128, 256], [256, 256, 512]]


def fc(i, o, act=True):
    l = nn.Linear(i, o)
    nn.init.kaiming_normal_(l.weight); nn.init.constant_(l.bias, 0)
    return nn.Sequential(l, nn.ReLU(inplace=True)) if act else l


def conv1d(i, o):
    c = nn.Conv1d(i, o, 1)
    nn.init.kaiming_normal_(c.weight); nn.init.constant_(c.bias, 0)
    return nn.Sequential(c, nn.ReLU(inplace=True))


def conv3d(i, o, k, p):
    c = nn.Conv3d(i, o, k, padding=p)
    nn.init.kaiming_normal_(c.weight); nn.init.constant_(c.bias, 0)
    return nn.Sequential(c, nn.ReLU(inplace=True))


def convt3d(i, o):
    c = nn.ConvTranspose3d(i, o, 2, stride=2)
    nn.init.kaiming_normal_(c.weight); nn.init.constant_(c.bias, 0)
    return nn.Sequential(c, nn.ReLU(inplace=True))


def fps(x, n):
    idx = torch.zeros(n, dtype=torch.long, device=x.device)
    d = torch.full((x.shape[0],), 1e10, device=x.device)
    far = 0
    for i in range(n):
        idx[i] = far
        d = torch.minimum(d, ((x - x[far]) ** 2).sum(-1))
        far = int(d.argmax())
    return idx


def ball_indices(x, centers, r, ns):
    """exact pointnet2 ball-query semantics: first `ns` in-radius points in index order, padded with the first in-radius one"""
    d = torch.cdist(centers, x)
    inr = d < r
    first = inr.float().argmax(-1)
    ar = torch.arange(x.shape[0], device=x.device).expand_as(d)
    key = torch.where(inr, ar, torch.full_like(ar, x.shape[0]))
    take = key.sort(-1).values[:, :ns]
    return torch.where(take >= x.shape[0], first.unsqueeze(-1).expand_as(take), take)


class SCN(nn.Module):
    def __init__(self, gripper_pts):
        super().__init__()
        self.scene_pt_mlp = nn.Sequential(conv1d(SCENE_PT_MLP[0], SCENE_PT_MLP[1]), conv1d(SCENE_PT_MLP[1], SCENE_PT_MLP[2]))
        self.vox = nn.ModuleList([nn.Sequential(conv3d(256, 512, 3, 1), nn.MaxPool3d(2, 2)),
                                  nn.Sequential(conv3d(512, 1024, 3, 1), nn.MaxPool3d(2, 2)),
                                  convt3d(1024, 512)])
        dims = [[m[0] + 3] + m[1:] for m in OBJ_MLPS]                       # use_xyz appends 3 channels
        self.sa_mlp = nn.ModuleList([nn.Sequential(*[conv1d(d[i], d[i + 1]) for i in range(len(d) - 1)]) for d in dims])
        self.obj_fc = nn.Sequential(fc(512, 1024), fc(1024, 1024))
        self.cls = nn.Sequential(fc(CLS_FC[0], CLS_FC[1]), fc(CLS_FC[1], CLS_FC[2]), fc(CLS_FC[2], 1, act=False))
        for name, v in (("b0", BOUNDS[0]), ("b1", BOUNDS[1]), ("vs", VOX)):
            self.register_buffer(name, torch.tensor(v))
        self.register_buffer("nv", torch.tensor(NV)); self.register_buffer("nv2", torch.tensor(NV2))
        self.register_buffer("strides", torch.tensor([int(NV[1] * NV[2]), int(NV[2]), 1]))
        self.register_buffer("strides2", torch.tensor([int(NV2[1] * NV2[2]), int(NV2[2]), 1]))
        self._static = []
        self.set_gripper(gripper_pts)

    @torch.no_grad()
    def set_gripper(self, pts):
        """the query body is constant: FPS / ball-query indices depend on coordinates only, compute once (as in Sajid's port)"""
        x = torch.as_tensor(pts, dtype=torch.float32)
        st = []
        for npoint, r, ns in zip(OBJ_NPOINTS, OBJ_RADII, OBJ_NSAMPLES):
            if npoint is None:
                st.append((x, None, None))
                break
            c = x[fps(x, npoint)]
            st.append((x, c, ball_indices(x, c, r, ns)))
            x = c
        self._static = st

    def obj_features(self):
        dev = self.b0.device
        f = None
        for sa, (x, c, take) in zip(self.sa_mlp, self._static):
            x = x.to(dev)
            if c is None:                                                    # group-all
                g = x if f is None else torch.cat([x, f], -1)
                return self.obj_fc(sa(g.t().unsqueeze(0)).squeeze(0).max(-1).values.unsqueeze(0)).squeeze(0)
            c, take = c.to(dev), take.to(dev)
            gx = x[take] - c.unsqueeze(1)
            g = gx if f is None else torch.cat([gx, f[take]], -1)
            h = sa(g.reshape(-1, g.shape[-1]).t().unsqueeze(0)).squeeze(0)
            f = h.reshape(-1, c.shape[0], take.shape[1]).max(-1).values.t()

    def scene_features(self, pts):
        """pts (B, N, 3) world coordinates -> (B, 1024, #scale-2 voxels)"""
        B = pts.shape[0]
        inds = torch.minimum(((pts - self.b0) // self.vs).long().clamp(min=0), self.nv - 1)
        flat = (inds * self.strides).sum(-1)                                 # (B, N)
        centers = inds.float() * self.vs + self.vs / 2 + self.b0
        h = self.scene_pt_mlp((pts - centers).transpose(1, 2))               # (B, 256, N)
        vox = torch.zeros(B, 256, int(self.nv.prod()), device=pts.device, dtype=h.dtype)
        vox.scatter_reduce_(2, flat.unsqueeze(1).expand(-1, 256, -1), h, "amax", include_self=False)
        l = [vox.reshape(B, 256, *NV.tolist())]
        for m in self.vox:
            l.append(m(l[-1]))
        return torch.cat([l[1], l[-1]], 1).reshape(B, 1024, -1)

    def classify(self, objf, scenef, trans, rot6):
        """trans (B, Q, 3), rot6 (B, Q, 6) -> logits (B, Q)"""
        trans = torch.minimum(torch.maximum(trans, self.b0 + 1e-3), self.b1 - 1e-3)
        i2 = torch.minimum(((trans - self.b0) // (2 * self.vs)).long().clamp(min=0), self.nv2 - 1)
        flat2 = (i2 * self.strides2).sum(-1)                                 # (B, Q)
        vf = torch.gather(scenef, 2, flat2.unsqueeze(1).expand(-1, 1024, -1)).transpose(1, 2)    # (B, Q, 1024)
        off = trans - (i2.float() * self.vs * 2 + self.vs / 2 + self.b0)     # the authors' centre formula (vox_size / 2)
        x = torch.cat([objf.expand(trans.shape[0], trans.shape[1], -1), vf, off, rot6], -1)
        return self.cls(x).squeeze(-1)

    def forward(self, pts, pos, R, shift):
        """pts (B, N, 3); pos (B, G, 3) pad centres, R (B, G, 3, 3), shift (B, 3) start -> goal.  -> pick, place logits (B, G)"""
        scenef = self.scene_features(pts)
        objf = self.obj_features()
        rot6 = torch.cat([R[..., :, 0], R[..., :, 1]], -1)
        G = pos.shape[1]
        q = self.classify(objf, scenef, torch.cat([pos, pos + shift[:, None, :]], 1), torch.cat([rot6, rot6], 1))
        return q[:, :G], q[:, G:]


def log1mexp(x):
    """log(1 - exp(x)) for x < 0"""
    x = torch.clamp(x, max=-1e-6)
    return torch.where(x > -0.6931, torch.log(-torch.expm1(x)), torch.log1p(-torch.exp(x)))


def feasible_logprob(z_pick, z_place):
    """log P(move works with this grasp) = log(1 - p_pick) + log(1 - p_place), where p = sigmoid(z) is a collision probability"""
    return torch.nn.functional.logsigmoid(-z_pick) + torch.nn.functional.logsigmoid(-z_place)


def hard_neg_bce(loss, frac=0.1):
    """NVlabs' trainer (loss_pct 0.1): average of the top-10% hardest elements and a random 10%, weighted 0.5 / 0.5"""
    n = loss.numel()
    if n == 0:
        return loss.sum()
    k = max(1, int(frac * n))
    top = loss.topk(k).values.mean()
    rnd = loss[torch.randperm(n, device=loss.device)[:k]].mean()
    return 0.5 * (top + rnd)
