import sys,time; sys.path.insert(0,'.')
from mamo.scene import parse_scene_file
from mamo.context import MAMOContext
from mamo.m4m.planner import m4m
for lvl,sid in [(10,200003),(10,200006),(10,200001),(10,200017)]:
    sc=parse_scene_file(rf"D:/RRC/New folder/m4m_em4m_project/archive/ros_src/em4m-main/dat/clutter_scenes/{lvl}/plan_{sid}_SCENE.txt")
    ctx=MAMOContext(sc,seed=0)
    sol=m4m(ctx)
    print(sid,'solved',sol.solved,sol.reason,'pushes',len(sol.pushes),{k:(round(v,1) if isinstance(v,float) else v) for k,v in sol.stats.items()},flush=True)
    print('   retrieval fail stats',ctx.retrieval_stats,'push stats',ctx.push_stats,flush=True)
