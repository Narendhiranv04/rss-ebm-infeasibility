# topdown_check summary (generated, EXPLORATORY, sim kinematics only)

Per cell: R reach / C neighbour clearance >= 5 mm / yaw that is clear (0, 9 = 90, B = both, - none) / slide directions that work (f = -x toward robot, b = +x, l = +y, r = -y). Row 5 back, row 0 next to the robot; col 0 = robot's right (-y).


## neighbours: tall_box {'shape': 'box', 'size': 0.018, 'height': 0.18}  (target {'shape': 'cylinder', 'size': 0.018, 'height': 0.14})
```
r5  Rx-fblr  Rx-fblr  Rx-fblr  Rx-fblr  Rx-fblr  Rx-fblr 
r4  Rx-fblr  Rx-fblr  Rx-fblr  Rx-fblr  Rx-fblr  Rx-fblr 
r3  Rx-fblr  Rx-fblr  Rx-fblr  Rx-fblr  Rx-fblr  Rx-fblr 
r2  Rx-fblr  Rx-fblr  Rx-fblr  Rx-fblr  Rx-fblr  Rx-fblr 
r1  Rx-fblr  Rx-fblr  Rx-fblr  Rx-fblr  Rx-fblr  Rx-fblr 
r0  Rx-fblr  Rx-fblr  Rx-fblr  Rx-fblr  Rx-fblr  Rx-fblr 
```
reach 36/36; clear 0/36; best clearance mm min/median -16.1 / -16.0; slide per direction: +x 36/36, -x 36/36, +y 36/36, -y 36/36; all 4 slides 36/36; grasp level used {'top': 12, 'middle': 24}; slides from reach-only grasps 36

## neighbours: median_cyl {'shape': 'cylinder', 'size': 0.0147, 'height': 0.147}  (target {'shape': 'cylinder', 'size': 0.018, 'height': 0.14})
```
r5  RCBfblr  RCBfblr  RCBfblr  RCBfblr  RCBfblr  RCBfblr 
r4  RCBfblr  RCBfblr  RCBfblr  RCBfblr  RCBfblr  RCBfblr 
r3  RCBfblr  RCBfblr  RCBfblr  RCBfblr  RCBfblr  RCBfblr 
r2  RCBfblr  RCBfblr  RCBfblr  RCBfblr  RCBfblr  RCBfblr 
r1  RCBfblr  RCBfblr  RCBfblr  RCBfblr  RCBfblr  RCBfblr 
r0  RCBfblr  RCBfblr  RCBfblr  RCBfblr  RCBfblr  RCBfblr 
```
reach 36/36; clear 36/36; best clearance mm min/median 7.2 / 7.3; slide per direction: +x 36/36, -x 36/36, +y 36/36, -y 36/36; all 4 slides 36/36; grasp level used {'top': 36}; slides from reach-only grasps 0

## VAL front-column rule
blocked 231/364 = 0.635; by K: {'0': {'blocked': 13, 'n': 86}, '1': {'blocked': 33, 'n': 53}, '2': {'blocked': 77, 'n': 97}, '3': {'blocked': 54, 'n': 70}, 'NONE': {'blocked': 54, 'n': 58}}
