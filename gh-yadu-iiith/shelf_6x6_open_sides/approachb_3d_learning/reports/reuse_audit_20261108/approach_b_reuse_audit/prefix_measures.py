import json, sys, collections as C
B3, CQ, SPL = sys.argv[1:4]
RAW = B3 + '/dataset_bd_classquota_without_home_position_no_overpass/raw'
split = json.load(open(B3 + '/dataset_placement_extension_v2/production_v3/frozen/split_manifest_first_cohort.json'))['split']
d3 = C.Counter()
for rid, sp in split.items():
    pref = set()
    for b in json.load(open(f'{RAW}/roots/{rid}.json'))['labels']['B']:
        m = len(b['sequence']); fs = b['failed_step']; ok = m if fs in (None, 'target') else (fs if m else 0)
        sq = tuple((o, tuple(d)) for o, d in b['sequence'])
        for j in range(1, ok + 1): pref.add(sq[:j])
    d3[sp + '|n'] += len(pref); d3[sp + '|depth_sum'] += sum(len(p) for p in pref)
print('3D unique connector-pending prefixes / sum of depths', dict(d3))
sp2 = {r['root_id']: r['split'] for r in json.load(open(SPL))['roots']}
sel = json.load(open(CQ + '/selected.json'))['selected']
d2 = C.Counter()
for rid in (x for v in sel.values() for x in v):
    sp = sp2[rid]; rec_pref, norec_pref = set(), set()
    for b in json.load(open(f'{CQ}/raw/roots/{rid}.json'))['labels']['B']:
        m = len(b['sequence'])
        if not m: continue
        fs = b['failed_step']; ok = m if fs in (None, 'target') else fs
        sq = tuple((o, int(d)) for o, d in b['sequence'])
        for j in range(1, ok + 1):
            (rec_pref if fs is None else norec_pref).add(sq[:j])
    d2[sp + '|prefix_with_records'] += len(rec_pref); d2[sp + '|prefix_relocs_ok_no_records'] += len(norec_pref - rec_pref)
print('2D corpus unique relocation-complete prefixes', dict(sorted(d2.items())))
