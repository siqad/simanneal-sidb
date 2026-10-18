"""Auto pk- budgets must equal the frozen SC2s policy file at 10-62 sites.

Usage: pkm_auto_budget.py SIMANNEAL_BINARY SOURCE_DIR. Standard library only.
tests/data/PART6_pkm_policies.json is the frozen sa-evo part-6 policy file
(sha256 a7c96639...). form() below is copied from sa-evo scaling/scripts/policy.py
v2 (sha256 275c5abb...), which produced every SC2s test result.
"""
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET

binary, source = map(Path, sys.argv[1:3])
policy_file = source / 'tests/data/PART6_pkm_policies.json'
FROZEN_SHA256 = 'a7c96639a0a899b1bfa8f2dc1ef7fc4efb3e531f30c49863cc9611a5f1b4fc36'
assert hashlib.sha256(policy_file.read_bytes()).hexdigest() == FROZEN_SHA256, 'policy file changed'
SC2S = json.loads(policy_file.read_text())['SC2s']['pk-']

RG = [2 ** k for k in range(10)]
def form(p, n, pf):
    xs = math.log2(n) if p['form'] == 'POW' else n / 8
    c0 = p.get('c0', 4); nc = p.get('nc', 9)
    ci = min(max(int(round(p['a'] + p['b'] * xs + p['k'] * pf - c0)), 0), nc - 1)
    ri = min(max(int(round(p['r0'] + p['r1'] * math.log2(n) + p['m'] * pf)), 0), p.get('nr', 10) - 1)
    h = p['hop'][0] if p.get('nb') is None or n <= p['nb'] else p['hop'][1]
    return 2 ** (c0 + ci), RG[ri], h

def step(n):
    # Qualified tiers retained for pk- at 2-9 sites (no part-6 test there).
    return 256, 8, 2

def historical(n):
    return 10000, 16 if n <= 9 else 32 if n <= 25 else 128, 5

base = source / 'sample_problems/or_00_problem.xml'
def problem(path, count, physics):
    tree = ET.parse(base)
    params = tree.find('sim_params')
    for key in ['anneal_cycles', 'num_instances', 'hop_attempt_factor']:
        node = params.find(key)
        if node is not None:
            params.remove(node)  # omitted budgets request Auto
    for key, value in dict(physics, num_workers=1, random_seed=731,
                           population_backend='portable').items():
        node = params.find(key)
        if node is None:
            node = ET.SubElement(params, key)
        node.text = str(value)
    layer = tree.find("design/layer[@type='DB']")
    template = list(layer)[0]
    for site in list(layer):
        layer.remove(site)
    for i in range(count):
        site = ET.fromstring(ET.tostring(template))
        # 3000 lattice columns apart (~1.15 um): every site excludes positive charge.
        site.find('latcoord').attrib.update(n=str(3000 * i), m='0', l='0')
        site.find('physloc').attrib.update(x=str(3000 * i * 3.84), y='0')
        layer.append(site)
    tree.write(path)

physics_points = [
    dict(muzm=-0.32, eps_r=5.6, debye_length=5),
    dict(muzm=-0.45, eps_r=3.0, debye_length=8),
    dict(muzm=-0.5, eps_r=1.0, debye_length=10),
    dict(muzm=-0.1, eps_r=10.0, debye_length=1),
]
checked = 0
with tempfile.TemporaryDirectory() as directory:
    inp, out = Path(directory) / 'in.xml', Path(directory) / 'out.xml'
    for physics in physics_points:
        for count in [1] + list(range(2, 63)) + [63]:
            problem(inp, count, physics)
            result = subprocess.run([str(binary), str(inp), str(out)], capture_output=True,
                                    text=True, timeout=120)
            assert result.returncode == 0, (physics, count, result.stderr)
            misc = ET.parse(out)
            got = tuple(int(misc.findtext('misc/' + k)) for k in
                        ['anneal_cycles', 'requested_restarts', 'hop_attempt_factor'])
            eligible = 2 <= count <= 62
            expected = (form(SC2S, count, 0.0) if count >= 10 else step(count)) if eligible \
                else historical(count)
            assert got == expected, (physics, count, got, expected)
            assert misc.findtext('misc/pkm_auto_selected') == ('true' if eligible else 'false'), (physics, count)
            assert misc.findtext('misc/pkp_large_auto_selected') == 'false', (physics, count)
            assert misc.findtext('misc/search_budget') == ('scoped_auto' if eligible else 'stock_or_explicit')
            checked += 1
print(f'pk- Auto budgets match frozen SC2s policy {FROZEN_SHA256[:16]} at {checked} size/physics points')
