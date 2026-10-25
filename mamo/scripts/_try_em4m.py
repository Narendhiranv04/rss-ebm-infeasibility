import sys,time; sys.path.insert(0,'.')
from mamo.scene import parse_scene_file
from mamo.context import MAMOContext
from mamo.em4m.planner import em4m
ids = [int(a) for a in sys.argv[1:]] or [200003, 200006, 200017]
for sid in ids:
    lvl = {1:5, 2:10, 3:15}[sid // 100000]
    sc=parse_scene_file(rf"D:/RRC/New folder/m4m_em4m_project/archive/ros_src/em4m-main/dat/clutter_scenes/{lvl}/plan_{sid}_SCENE.txt")
    ctx=MAMOContext(sc,seed=0)
    sol=em4m(ctx)
    print(sid,'solved',sol.solved,sol.reason,'pushes',len(sol.pushes),{k:(round(v,1) if isinstance(v,float) else v) for k,v in sol.stats.items()},flush=True)
    print('   push stats',ctx.push_stats,flush=True)
