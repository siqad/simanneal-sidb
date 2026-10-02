"""Production XML/CLI regression checks. Uses only the Python standard library."""
import itertools
import json
import math
from pathlib import Path
import subprocess
import sys
import struct
import tempfile
import xml.etree.ElementTree as ET

binary, source = map(Path, sys.argv[1:3])
base = source / 'sample_problems/or_00_problem.xml'

def problem(path, options, singleton=False):
    tree = ET.parse(base)
    params = tree.find('sim_params')
    defaults = dict(anneal_cycles=64, num_instances=8, num_workers=2,
                    random_seed=731, population_backend='portable')
    defaults.update(options)
    for key, value in defaults.items():
        node = params.find(key)
        if node is None:
            node = ET.SubElement(params, key)
        node.text = str(value)
    if singleton:
        layer = tree.find("design/layer[@type='DB']")
        for site in list(layer)[1:]:
            layer.remove(site)
    tree.write(path)
    return tree

def oracle(tree, q):
    sites = tree.findall("design/layer[@type='DB']/dbdot/latcoord")
    f32 = lambda v: struct.unpack('f', struct.pack('f', v))[0]
    # SiQADConnector stores lattice vectors as float; retain its input contract.
    points = [(f32(int(p.get('n')) * f32(3.84)), f32(f32(int(p.get('m')) * f32(7.68)) + int(p.get('l')) * 2.25)) for p in sites]
    a = [[0.0] * len(q) for _ in q]
    for i in range(len(q)):
        for j in range(i):
            r = math.dist(points[i], points[j]) * 1e-10
            a[i][j] = a[j][i] = 1.602e-19 / (4 * 3.14159 * 8.854e-12 * 5.6) * math.exp(-r / 5e-9) / r
    energy = lambda c: sum(c[i] * a[i][j] * c[j] for i in range(len(c)) for j in range(i))
    v = [-sum(a[i][j] * q[j] for j in range(len(q))) for i in range(len(q))]
    mu, eps = float(tree.findtext('sim_params/muzm')), 1e-6
    pop = all((c == -1 and v[i]+mu < eps) or
              (c == 1 and v[i]+mu-.59 > -eps) or
              (c == 0 and v[i]+mu > -eps and v[i]+mu-.59 < eps)
              for i, c in enumerate(q))
    # Compute hop deltas independently by changing both charges in the Hamiltonian.
    stable = pop
    for i, qi in enumerate(q):
        for j, qj in enumerate(q):
            if qi < qj:
                moved = list(q)
                moved[i] += 1
                moved[j] -= 1
                stable &= energy(moved) - energy(q) >= -eps
    return stable, energy(q)

with tempfile.TemporaryDirectory() as directory:
    directory = Path(directory)
    inp, out = directory/'in.xml', directory/'out.xml'
    def run(options, singleton=False, args=(), succeeds=True):
        tree = problem(inp, options, singleton)
        out.unlink(missing_ok=True)
        result = subprocess.run([str(binary), str(inp), str(out), *args], capture_output=True, text=True, timeout=30)
        assert (result.returncode == 0) == succeeds, (options, result.stderr, result.stdout[-2000:])
        if not succeeds:
            # SiQADConnector writes an empty document during destruction.
            # Failure must not export any successful charge result.
            if out.exists():
                assert not ET.parse(out).findall('.//dist'), options
            return
        xml = ET.parse(out)
        distributions = xml.findall('.//dist')
        assert distributions, options
        for node in distributions:
            q = [{'0':0, '-':-1, '+':1}[c] for c in node.text.strip()]
            valid, energy = oracle(tree, q)
            assert math.isclose(float(node.get('energy')), energy, rel_tol=0, abs_tol=1e-12), (node.attrib,energy,q)
            if node.get('physically_valid') == '1':
                assert valid, (options, q)
        return xml

    original = run({})
    assert sum(int(x.get('count')) for x in original.findall('.//dist')) == 8
    for refinement in ['none', 'k6', 'k10', 'shared']:
        result = run(dict(search_profile='optimized', refinement=refinement, refinement_rounds=2))
        assert result.findtext('misc/search_profile') == 'optimized'
        assert result.findtext('misc/random_backend') == 'pcg32'
        assert int(result.findtext('misc/refinement_geometry_count')) >= 0
        assert result.findtext('misc/refinement_shared_single_cache_fallback') == ('true' if refinement == 'shared' else 'false')
        assert any(x.get('physically_valid') == '1' for x in result.findall('.//dist'))
    singleton = run(dict(search_profile='optimized'), singleton=True)
    assert singleton.findtext('misc/executed_restarts') == '0'
    assert singleton.findtext('misc/singleton_used') == 'true'
    for key, value in [('search_profile','typo'), ('random_backend','typo'),
                       ('repair','perhaps'), ('transient_domain_mask','perhaps'),
                       ('refinement','typo'), ('refinement_rounds','2junk'),
                       ('refinement_candidates','0'), ('anneal_cycles','0'),
                       ('population_backend','typo'), ('random_seed','-1')]:
        run({key:value}, succeeds=False)
    pots = directory/'pots.json'
    pots.write_text(json.dumps({'pots': [[0.0]]}))
    run({}, args=['--ext-pots', str(pots)], succeeds=False)
    run({}, args=['--ext-pots'], succeeds=False)
    run({}, args=['--ext-pots-step'], succeeds=False)
print('CLI profile, physical output, singleton, metadata, and input checks passed')
