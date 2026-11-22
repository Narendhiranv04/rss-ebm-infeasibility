# xArm7 + xArm Gripper G2 (MuJoCo)

`xarm7_g2.xml` is the menagerie `xarm7.xml` with the gripper meshes swapped for UFactory's G2 meshes.
Same body / joint / actuator / geom / site names, same linkage (equality constraints + `split` tendon + one
`gripper` actuator, ctrl 0 = open .. 255 = closed). To use it in the sim: `XARM7_XML=<this folder>/xarm7_g2.xml`.

## Bottom line

In UFactory's own robot description, **the G2 changes only the meshes, and of those only the base housing really
changes.** Joint origins, joint limits, inertials and the TCP (`joint_tcp` 0.172 m) are the same as the G1. The G2
finger and knuckle meshes are the G1 shapes re-exported (same vertex count; every vertex matches within 0.01 mm, the
rounding). So in simulation the **gap range, pad size and pad position are identical**; only the housing is wider.

## Provenance

Repo `xArm-Developer/xarm_ros2`, commit `d837f2e17f7f36043103b367fca3af8172626245` (branch jazzy), fetched 2026-11-22
with `gh api .../contents/<path>?ref=<commit>` into `src_ufactory/` (untrusted data; only parsed, never executed).

| file (in repo `xarm_description/`) | sha256 |
|---|---|
| urdf/gripper/gripper_macro.xacro | 28fbba5fe5691681d571afbcc489ce6cc7c042cd1a83490c1db4770e0ab71678 |
| urdf/gripper/gripper.urdf.xacro | 6bb72e75b3b32094669973dfbeb3b61beb6ce0d7ca9604f03681a8f8df025e57 |
| urdf/gripper/xarm_gripper_macro.xacro | 60b681379a59957c31ef7b0e2b66ffa1c42c244d13f49aeff3690f4282e67a06 |
| urdf/gripper/xarm_gripper.urdf.xacro | e1987b755501636b7ba32518d98ecb5fc8c6d2c4b1c45a49d332d6672f22c8cb |
| meshes/gripper/xarm/base_link_g2.dae | d46f7e336755a7fd58ebee4b3eab2c6eca7eec611358fc6bf5f683e63efd7575 |
| meshes/gripper/xarm/left_finger_g2.dae | 69c888529def394ddf00edfb11db061d300c586e6df681a33c894af5be14a59b |
| meshes/gripper/xarm/right_finger_g2.dae | c2914ea791f520253e5536b0078f4e7e3b7b16112eb3dab21db2f5029c150b36 |
| meshes/gripper/xarm/left_inner_knuckle_g2.dae | f12e642f44ce4a5b0c9bb2ed5c2cf4f9ea0a22b5e87839c29490e8eee6171b86 |
| meshes/gripper/xarm/right_inner_knuckle_g2.dae | 9e73e1fb4e65bc79dbaf2d8fe86fa835c67faa1e5ab08a8a55bec3b9023a1aa3 |
| meshes/gripper/xarm/left_outer_knuckle_g2.dae | b791ec24aae545a5dd70aefd12ddac6bdbecf74388ca2f655b34b140ae7da80b |
| meshes/gripper/xarm/right_outer_knuckle_g2.dae | d9ed7ed551136ecd57fe733c5e9c7e7fa70e10345e26441452122b6bba061f55 |

Where the switch lives: `gripper_version` ('G1' default) is passed from `xarm_device_macro.xacro` ->
`xarm_gripper_macro.xacro` -> `xarm_gripper.urdf.xacro`, where `G2` only sets `name_suffix=_g2` and the `.dae` extension.
(`gripper_macro.xacro`/`gripper.urdf.xacro` do not pass `gripper_version` at all.)

## Conversion (`convert.py`)

`<env3d python> -I convert.py`: stdlib XML + numpy, no new packages. DAE header: `<unit meter="1"/>`, `Z_UP`, mesh
node transform = identity, so vertices are used as-is (metres, URDF link frame). One fix: `base_link_g2.dae` holds a
stray 0.1 mm cluster of 300 triangles at (-0.25, 0.47, 0.10) m (an export artifact). It would inflate the collision
convex hull, so the script drops any triangle > 0.2 m from the origin (only these 300). Output: binary STL in `assets/`.
The script never overwrites a differing file.

## Validation (`check_g2.py`, MuJoCo 3.16.0, 1 process, CPU)

Both models loaded the way `shelf_scene.py` does (`MjSpec.from_file` + `attach`); all 37 names the sim uses are found.
Gripper swept 0 -> 255 with the arm at `home`. Hand frame = `xarm_gripper_base_link` (= flange), z = approach.

| | standard (menagerie) | G2 | UFactory G2 spec |
|---|---|---|---|
| pad gap, open / closed | 88.8 / 4.6 mm | 88.8 / 4.6 mm | stroke 84 ± 1 mm (model: 84.2) |
| pad size (2 boxes per finger) | 30 wide x 19 long x 4 thick mm each; 37 mm span | same | not published |
| flange -> pad centre (open .. closed) | 142.2 .. 154.5 mm | same | URDF TCP 172 (fingertip) for both |
| flange -> fingertip (open / closed) | 162.2 / 174.5 mm | same | L ≈ 165.2 at d = 85, ≈ 175.9 at d = 0 |
| housing width (closing axis y) x depth x | 102.2 x 75.0 mm | **109.6** x 75.0 mm | not published |
| housing z range | -9.0 .. 103.1 mm | -5.0 .. 104.8 mm | |
| whole gripper width y (open / closed) | 172.9 / 110.1 mm | same (knuckles are widest) | |
| self-contacts in gripper during sweep | 0 | 0 | |

Spec source: UFactory, xArm Gripper G2 (AG1200) Technical Specifications,
https://docs.accessories.ufactory.cc/xArm_Gripper_G2/6.Technical_Specifications.html (also weight 820 g, payload 5 kg,
10-50 N).

## Caveats

- The G2 in this model is only as accurate as UFactory's URDF, which reuses the G1 kinematics and finger shapes. If the
  real G2 fingers or linkage differ, this model will not show it. A caliper check on the real gripper (max opening, pad
  size, flange-to-pad distance) would settle it.
- Inertials are the G1 values (the URDF does not change them); the real G2 is 820 g.
- `shelf_scene.py` `_HAND_EY[5] = 0.052` (hand half-width > 5 cm above the pads) is below the G2 housing
  half-width (54.8 mm; G1 51.1 mm), so the carry footprint rule would under-count the G2 housing by about 3 mm per side.
- The collision geom for the housing is MuJoCo's convex hull of the mesh, as for the standard model.
