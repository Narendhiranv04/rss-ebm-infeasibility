import json, os, sys, collections as C
RAW, PV3 = sys.argv[1], sys.argv[2]
split = json.load(open(PV3 + '/frozen/split_manifest_first_cohort.json'))['split']
T = C.Counter()
def ekey(c0, ed): return tuple(sorted({**{k: tuple(v) for k, v in c0.items()}, **{k: tuple(v) for k, v in ed.items()}}.items()))
for fn in sorted(os.listdir(RAW + '/roots')):
    rid = fn[:-5]; sp = split.get(rid)
    if sp is None: continue
    r = json.load(open(RAW + '/roots/' + fn)); L = r['labels']; c0 = r['initial_cells']
    park = {o: tuple(v[0]) for o, v in r['parking'].items() if v}
    yk = {ekey(c0, {}): L['original']['y']}
    for d in L['D']: yk[ekey(c0, d['edit'])] = d['y']
    bl = L['blocker']; mins = [set(s) for s in bl.get('minimal_sets', [])]
    pos = {frozenset(s) for s in bl.get('minimal_sets', [])} | {frozenset(s) for v in (bl.get('clearing_sets_reachable') or {}).values() for s in v}
    def Y(seq):
        k = ekey(c0, dict(seq))
        if k in yk: return 'stored%d' % yk[k]
        if not all(tuple(d) == park.get(o) for o, d in seq) or L['original']['y'] or not bl.get('full_lattice'): return 'unknown'
        S = {o for o, _ in seq}
        if frozenset(S) in pos: return 'lattice1'
        if S <= set(bl['scope']) and not any(m <= S for m in mins): return 'lattice0'
        return 'unknown'
    g = C.defaultdict(list)
    for b in L['B']:
        m = len(b['sequence'])
        if not m: continue
        fs = b['failed_step']; xs = 'X0_reloc_failed' if fs not in (None, 'target') else 'Xpending_connector'
        seq = [(o, tuple(d)) for o, d in b['sequence']]
        T[(sp, b['role'], xs, Y(seq))] += 1
        g[ekey(c0, dict(seq))].append((tuple(seq), xs, Y(seq)))
    for e, v in g.items():
        if len({s for s, _, _ in v}) >= 2:
            T[(sp, 'ABBA_group', '|'.join(sorted(x for _, x, _ in v)), v[0][2])] += 1
agg = C.Counter()
for (sp, role, xs, y), n in T.items():
    agg[(role if role == 'ABBA_group' else 'ALL_B', xs, y)] += n
for k, v in sorted(agg.items()): print(k, v)
print('--- by role (all splits)')
rr = C.Counter()
for (sp, role, xs, y), n in T.items():
    if role != 'ABBA_group': rr[(role, xs)] += n
for k, v in sorted(rr.items()): print(k, v)
json.dump({'|'.join(k): v for k, v in T.items()}, open('xtab3d.json', 'w'), indent=0)
