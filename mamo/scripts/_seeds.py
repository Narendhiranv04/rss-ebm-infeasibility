import sys,time; sys.path.insert(0,'.')
from mamo.scene import parse_scene_file
from mamo.context import MAMOContext
from mamo.em4m.planner import em4m
sid=int(sys.argv[1]); lvl={1:5,2:10,3:15}[sid//100000]
sc=parse_scene_file(rf"D:/RRC/New folder/m4m_em4m_project/archive/ros_src/em4m-main/dat/clutter_scenes/{lvl}/plan_{sid}_SCENE.txt")
for seed in map(int, sys.argv[2:]):
    ctx=MAMOContext(sc,seed=seed); sol=em4m(ctx)
    print(sid,'seed',seed,'solved',sol.solved,sol.reason,{k:(round(v,1) if isinstance(v,float) else v) for k,v in sol.stats.items()},flush=True)
    print('   push stats',ctx.push_stats,flush=True)
