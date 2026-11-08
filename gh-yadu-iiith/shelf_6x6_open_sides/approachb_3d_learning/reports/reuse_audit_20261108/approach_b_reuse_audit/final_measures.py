import json, os, sys, glob, collections as C
B3, CQ, XY = sys.argv[1], sys.argv[2], sys.argv[3]
RAW = B3 + '/dataset_bd_classquota_without_home_position_no_overpass/raw'
split = json.load(open(B3 + '/dataset_placement_extension_v2/production_v3/frozen/split_manifest_first_cohort.json'))['split']
def ek(c0, ed): return tuple(sorted({**{k: tuple(v) for k, v in c0.items()}, **{k: tuple(v) for k, v in ed.items()}}.items()))
need = C.Counter(); steps = 0; trie = 0
for rid in split:
    r = json.load(open(f'{RAW}/roots/{rid}.json')); L = r['labels']; c0 = r['initial_cells']
    park = {o: tuple(v[0]) for o, v in r['parking'].items() if v}
    yk = {ek(c0, {})} | {ek(c0, d['edit']) for d in L['D']}
    bl = L['blocker']; mins = [set(s) for s in bl.get('minimal_sets', [])]
    pos = {frozenset(s) for s in bl.get('minimal_sets', [])} | {frozenset(s) for v in (bl.get('clearing_sets_reachable') or {}).values() for s in v}
    ends = {}; pref = set()
    for b in L['B']:
        m = len(b['sequence'])
        if not m: continue
        fs = b['failed_step']; ok = m if fs in (None, 'target') else fs
        sq = tuple((o, tuple(d)) for o, d in b['sequence'])
        for j in range(1, ok + 1):
            if sq[:j] not in pref:
                pref.add(sq[:j]); trie += 1
            e = ek(c0, dict(sq[:j]))
            if e in yk: src = 'stored'
            elif all(d == park.get(o) for o, d in sq[:j]) and not L['original']['y'] and bl.get('full_lattice'):
                S = {o for o, _ in sq[:j]}
                src = 'lattice' if (frozenset(S) in pos or (S <= set(bl['scope']) and not any(mm <= S for mm in mins))) else 'unknown'
            else: src = 'unknown'
            ends[e] = src
    for s in ends.values(): need[s] += 1
print('3D unique X-candidate endpoints by Y source', dict(need), 'unique prefixes (trie nodes = relocation-step replays needed)', trie)
sel = json.load(open(CQ + '/selected.json'))['selected']
cpu = calls = 0
for k, v in sel.items():
    for rid in v:
        r = json.load(open(f'{CQ}/raw/roots/{rid}.json')); cpu += r['teacher_cpu_s']; tc = r['teacher_calls']; calls += tc['goal'] + tc['sequence']
print('2D corpus teacher CPU-s', round(cpu), 'calls', calls, 'CPU-s/call', round(cpu / calls, 2))
cq = {rid for v in sel.values() for rid in v}
xy = {os.path.basename(p) for p in glob.glob(XY + '/*/results/*') + glob.glob(XY + '/*/*/results/*')}
print('2D X/Y roots', len(xy), 'overlap with class-quota selected', len(xy & cq), 'namespaces', sorted({'-'.join(x.split('-')[:2]) for x in xy}))
