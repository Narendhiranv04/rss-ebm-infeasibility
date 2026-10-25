import sys,time; sys.path.insert(0,'.')
import mamo.mapf.cbs as C
import mamo.context as X
orig=C.cbs
def wrapped(agents,tl=30.0,max_t=400,stats=None):
    t=time.perf_counter(); st={}
    r=orig(agents,tl,max_t,st)
    dt=time.perf_counter()-t
    if dt>1.0:
        ninv=sum(len(a.invalid_goals) for a in agents.values()); pen=sum(a.goal_penalty is not None for a in agents.values())
        print(f'CBS {dt:.1f}s ok={r is not None} CTexp={None if r is None else r.expanded} ll={st.get("ll_expanded")} agents={len(agents)} invalid_goals={ninv} penalised={pen}',flush=True)
    return r
X.cbs=wrapped
from mamo.scene import parse_scene_file
from mamo.context import MAMOContext
from mamo.em4m.planner import em4m
sc=parse_scene_file(r"D:/RRC/New folder/m4m_em4m_project/archive/ros_src/em4m-main/dat/clutter_scenes/10/plan_200003_SCENE.txt")
ctx=MAMOContext(sc,seed=2); sol=em4m(ctx)
print('solved',sol.solved,{k:(round(v,1) if isinstance(v,float) else v) for k,v in sol.stats.items() if 'time' in k})
