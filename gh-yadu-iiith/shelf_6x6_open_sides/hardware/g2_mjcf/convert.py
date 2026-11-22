"""Convert UFactory xArm Gripper G2 COLLADA meshes (src_ufactory/meshes/*_g2.dae) to binary STL (assets/).

Stdlib + numpy only.  Run isolated (the DAEs are untrusted data):
    <python> -I convert.py
Keeps metres (applies the DAE <unit meter=..> scale) and the mesh frame (the URDF link frame);
refuses any DAE whose geometry node has a non-identity transform or non-triangle primitives.
Re-runnable: an existing STL is left alone if identical, otherwise the script stops (never overwrites).
"""
import struct
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SRC, DST = HERE / "src_ufactory" / "meshes", HERE / "assets"
NS = {"c": "http://www.collada.org/2005/11/COLLADASchema"}
PARTS = ["base_link", "left_finger", "right_finger", "left_inner_knuckle", "right_inner_knuckle",
         "left_outer_knuckle", "right_outer_knuckle"]


def load_dae(path):
    root = ET.parse(path).getroot()
    unit = root.find("c:asset/c:unit", NS)
    scale = float(unit.get("meter")) if unit is not None else 1.0
    assert (root.findtext("c:asset/c:up_axis", "Z_UP", NS)) == "Z_UP", "only Z_UP handled"
    geoms = root.findall("c:library_geometries/c:geometry", NS)
    assert len(geoms) == 1, f"{path.name}: expected 1 geometry, got {len(geoms)}"
    mesh = geoms[0].find("c:mesh", NS)
    assert not [e for e in mesh if e.tag.split("}")[1] in ("polylist", "polygons", "tristrips", "trifans")], \
        f"{path.name}: non-triangle primitives"
    # the node that instances the geometry must not move it
    for node in root.iter("{%s}node" % NS["c"]):
        if node.find("c:instance_geometry", NS) is not None:
            for m in node.findall("c:matrix", NS):
                M = np.array(m.text.split(), float).reshape(4, 4)
                assert np.allclose(M, np.eye(4)), f"{path.name}: non-identity node transform"
            assert not node.findall("c:translate", NS) + node.findall("c:rotate", NS) + node.findall("c:scale", NS)
    verts_src = mesh.find("c:vertices/c:input[@semantic='POSITION']", NS).get("source")[1:]
    arr = mesh.find(f"c:source[@id='{verts_src}']/c:float_array", NS)
    P = np.array(arr.text.split(), float).reshape(-1, 3) * scale
    tris = []
    for t in mesh.findall("c:triangles", NS):
        inputs = t.findall("c:input", NS)
        stride = max(int(i.get("offset")) for i in inputs) + 1
        voff = int(next(i for i in inputs if i.get("semantic") == "VERTEX").get("offset"))
        p = np.array(t.findtext("c:p", "", NS).split(), int).reshape(-1, stride)
        tris.append(p[:, voff].reshape(-1, 3))
        assert len(tris[-1]) == int(t.get("count"))
    return P, np.vstack(tris), scale


def stl_bytes(P, F):
    v = P[F]                                                    # (n, 3, 3)
    n = np.cross(v[:, 1] - v[:, 0], v[:, 2] - v[:, 0])
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-20)
    rec = np.zeros(len(F), dtype=[("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")])
    rec["n"], rec["v"] = n, v
    return b"g2 converted from UFactory DAE".ljust(80, b"\0") + struct.pack("<I", len(F)) + rec.tobytes()


def main():
    DST.mkdir(exist_ok=True)
    for part in PARTS:
        P, F, scale = load_dae(SRC / f"{part}_g2.dae")
        # base_link_g2.dae carries a 0.1 mm stray cluster (300 tris) at (-0.25, 0.47, 0.10) m: drop anything > 0.2 m out
        keep = (np.linalg.norm(P[F], axis=2) < 0.2).all(1)
        dropped, F = int((~keep).sum()), F[keep]
        data = stl_bytes(P, F)
        out = DST / f"{part}_g2.stl"
        if out.exists():
            if out.read_bytes() != data:
                sys.exit(f"{out} exists and differs; not overwriting")
            status = "unchanged"
        else:
            out.write_bytes(data)
            status = "written"
        lo, hi = P[F].reshape(-1, 3).min(0), P[F].reshape(-1, 3).max(0)
        print(f"{part:22s} unit={scale:g} tris={len(F):6d} dropped={dropped:3d} bbox_m min={np.round(lo, 4)} max={np.round(hi, 4)} {status}")


if __name__ == "__main__":
    main()
