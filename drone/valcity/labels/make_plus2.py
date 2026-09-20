"""v1_plus2 labels: v1_full with the objects verified by eye on Sat 19 Sep (drone/valcity/crops.py sheet) promoted to
sure: spacecraft clusters 2, 34, 41, 73, 77, 85, 107; cluster 182 = a fourth red-nosed small_plane (the votes said
spacecraft/medium_plane); manual jammer m1 (frame 2). Still unsure (left out of training as ignore regions): 12
(spacecraft vs jet_plane), 364 (medium_plane?), m2 (large_tower?).
Usage: python make_plus2.py v1_full.json OUT.json"""
import json
import sys

d = json.load(open(sys.argv[1]))
SURE = {'2': 'spacecraft', '34': 'spacecraft', '41': 'spacecraft', '73': 'spacecraft', '77': 'spacecraft',
        '85': 'spacecraft', '107': 'spacecraft', '182': 'small_plane', 'm1': 'jammer'}
for k, cls in SURE.items():
    o = d['objects'][k]
    o['cls'] = cls
    o['unsure'] = False
n_sure = sum(1 for o in d['objects'].values() if not o.get('unsure'))
json.dump(d, open(sys.argv[2], 'w'), indent=1)
print('objects', len(d['objects']), 'sure', n_sure, 'unsure', len(d['objects']) - n_sure)
