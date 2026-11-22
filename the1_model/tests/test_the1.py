import sys
from pathlib import Path

import numpy as np
import pytest
import sympy as sp

from the1.benchmark_suite import PhysicsBenchmarkSuite
from the1.core import The1WorldModel
from the1.lean4 import (
    LeanTranslationError, LeanVerifier, lean_symbol_map, sympy_to_lean,
)
from the1.math_suite import MathBenchmarkSuite


def _discover(name, n=12):
    law = PhysicsBenchmarkSuite.get_all_laws()[name]
    X, y = law.sample_data(n, seed=7)
    model = The1WorldModel(random_state=1)
    return model.fit(X, y, var_names=law.var_names), X, law


def test_translation_basics():
    x, y = sp.symbols('x y')
    expr = sp.sqrt(x ** 2 + y ** 2) + sp.Abs(x) + sp.pi * sp.exp(-y)
    text = sympy_to_lean(expr, ['x', 'y'])
    assert 'Real.sqrt' in text
    assert '|x|' in text
    assert 'Real.pi' in text
    assert 'Real.exp' in text


def test_translation_rejects_complex_and_unknown():
    x = sp.Symbol('x')
    with pytest.raises(LeanTranslationError):
        sympy_to_lean(sp.I * x, ['x'])
    with pytest.raises(LeanTranslationError):
        sympy_to_lean(sp.Function('mystery')(x), ['x'])


def test_identifier_collisions_and_keywords():
    mapping = lean_symbol_map(['x-1', 'x_1', 'theorem', '2v'])
    assert len(set(mapping.values())) == 4
    assert mapping['theorem'].endswith('_')
    assert mapping['2v'].startswith('v_')


def test_kinetic_energy_hypotheses():
    law, X, _ = _discover('kinetic_energy')
    hyps = LeanVerifier().generate_hypotheses(law, X)
    kinds = {h.kind for h in hyps}
    assert {'symmetry', 'homogeneity', 'positivity'} <= kinds
    even = next(h for h in hyps if h.name == 'even_v')
    assert even.statement == 'law m (-v) = law m v'
    hom = next(h for h in hyps if h.kind == 'homogeneity')
    assert '(s ^ 3)' in hom.statement


def test_log_domain_does_not_create_false_symmetry():
    law = MathBenchmarkSuite.get_all_laws()['log_natural']
    X, y = law.sample_data(12, seed=7)
    found = The1WorldModel(random_state=1).fit(X, y, var_names=law.var_names)
    hyps = LeanVerifier().generate_hypotheses(found, X)
    assert all(h.kind != 'symmetry' for h in hyps)


def test_build_source_matches_law_statement():
    law, X, _ = _discover('newton_second')
    verifier = LeanVerifier()
    hyps = verifier.generate_hypotheses(law, X)
    source, placed = verifier.build_source(law, hyps)
    body = sympy_to_lean(law.sympy_expr, law.var_names)
    assert f'def law ({law.var_names[0]} {law.var_names[1]} : ℝ) : ℝ := {body}' in source
    assert placed[0].line_start > 0
    assert placed[-1].line_end <= len(source.splitlines())


def test_unavailable_toolchain_is_reported(tmp_path, monkeypatch):
    monkeypatch.setattr('shutil.which', lambda name: None)
    law, X, _ = _discover('kinetic_energy')
    report = LeanVerifier().verify_law(law, X)
    assert report['status'] == 'unavailable'
    assert report['hypotheses'] and all(
        h['status'] == 'unverified' for h in report['hypotheses'])
    assert 'import Mathlib' in report['source']


def test_fake_lean_diagnostics_are_mapped(tmp_path):
    script = tmp_path / 'fake_lean.py'
    script.write_text(
        'import sys\n'
        'path = sys.argv[1]\n'
        'lines = open(path, encoding="utf-8").read().splitlines()\n'
        'for i, line in enumerate(lines, 1):\n'
        '    if line.startswith("theorem homogeneous"):\n'
        '        print(f"{path}:{i}:0: error: tactic failed")\n'
        '        sys.exit(1)\n',
        encoding='utf-8',
    )
    law, X, _ = _discover('kinetic_energy')
    verifier = LeanVerifier(lean_cmd=[sys.executable, str(script)], timeout=30)
    report = verifier.verify_law(law, X)
    assert report['status'] == 'partial'
    statuses = {h['name']: h['status'] for h in report['hypotheses']}
    assert statuses['homogeneous'] == 'unproven'
    assert statuses['positive'] == 'proven'


def test_quadratic_discovery_and_prediction():
    x = np.linspace(-3, 3, 20)
    y = 3.0 * x ** 2 + 2.0 * x + 1.0
    model = The1WorldModel(random_state=1)
    law = model.fit(x.reshape(-1, 1), y, var_names=['x'])
    assert law.train_r2 > 0.999
    assert np.allclose(law.predict(np.array([[2.0], [-2.0]])), [17.0, 9.0], atol=1e-6)


def test_explorer_reads_csv_and_verification_field(tmp_path):
    import pandas as pd
    from the1.explorer import DatasetExplorer
    path = tmp_path / 'kinetic.csv'
    X, y = PhysicsBenchmarkSuite.get_all_laws()['kinetic_energy'].sample_data(40, seed=3)
    pd.DataFrame({'m': X[:, 0], 'v': X[:, 1], 'E': y}).to_csv(path, index=False)
    report = DatasetExplorer(random_state=1).discover(path, target='E',
                                                      features=['m', 'v'], split='random')
    assert report.metrics['train_r2'] > 0.99
    assert report.equation.startswith('E = ')
    assert report.to_dict()['metrics']['source']
