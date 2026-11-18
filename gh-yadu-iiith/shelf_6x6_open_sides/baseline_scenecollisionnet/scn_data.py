"""Dataset of SceneCollisionNet states: a scene point cloud sampled from the layout + the 32 grasp poses of the target + labels."""
import pickle
from pathlib import Path

import numpy as np
import torch

import scn_geometry as G

HERE = Path(__file__).resolve().parent


def load():
    return pickle.load(open(HERE / "data" / "states.pkl", "rb")), pickle.load(open(HERE / "data" / "roots_scn.pkl", "rb"))


class StateSet(torch.utils.data.Dataset):
    def __init__(self, states, roots, n_points=8192, train=False):
        self.s, self.r, self.n, self.train = states, roots, n_points, train

    def __len__(self):
        return len(self.s)

    def __getitem__(self, i):
        s = self.s[i]
        r = self.r[s["rid"]]
        seed = None if self.train else i
        pts = G.scene_cloud(r["objects"], s["cells"], n_total=self.n, seed=seed if seed is not None else np.random.randint(1 << 30))
        return dict(pts=torch.from_numpy(pts), pos=torch.from_numpy(r["pos"]), R=torch.from_numpy(r["R"]), shift=torch.from_numpy(r["shift"]),
                    fy=torch.from_numpy(s["fy"].astype(np.int64)), pick=torch.from_numpy(s["pick"].astype(np.int64)),
                    place=torch.from_numpy(s["place"].astype(np.int64)), y=torch.tensor(float(s["y"])), idx=i)


def collate(items):
    return {k: torch.stack([it[k] for it in items]) if torch.is_tensor(items[0][k]) else torch.tensor([it[k] for it in items]) for k in items[0]}
