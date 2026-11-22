"""Behavioral physics/math benchmarks. Run from any directory; no external services."""
import argparse
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp

SCRIPT = Path(__file__).resolve().parents[1]/"skills/the-one-equation/scripts/discover_equations.py"
spec = importlib.util.spec_from_file_location("discovery",SCRIPT)
engine = importlib.util.module_from_spec(spec)
spec.loader.exec_module(engine)


def options(mode="regression",**changes):
    values=dict(mode=mode,degree=2,order=1,functions="none",frequencies="1",term=[],
                max_terms=12,max_features=400,smooth_window=0,include_time=False,ordered=False,
                selection_atol=1e-10,selection_rtol=1e-7)
    values.update(changes)
    return argparse.Namespace(**values)


class DiscoveryTests(unittest.TestCase):
    def setUp(self):
        self.rng=np.random.default_rng(83)

    def run_model(self,data,opts):
        report=engine.discover(data,opts)
        json.dumps(report,allow_nan=False)
        return report

    def test_newton_force(self):
        mass=self.rng.uniform(.2,10,200)
        acceleration=self.rng.uniform(-10,10,200)
        report=self.run_model({"variables":{"mass":mass,"acceleration":acceleration},
                               "targets":{"force":mass*acceleration}},options())
        self.assertLess(report["models"][0]["validation_rmse"],1e-9)
        self.assertEqual(report["models"][0]["selected_terms"],1)

    def test_ideal_gas_rational_term(self):
        n=self.rng.uniform(.1,2,300)
        temperature=self.rng.uniform(250,400,300)
        volume=self.rng.uniform(1,10,300)
        report=self.run_model({"variables":{"n":n,"T":temperature,"V":volume},
                               "targets":{"P":8.314*n*temperature/volume}},
                              options(degree=1,term=["n*T/V"]))
        self.assertLess(report["models"][0]["validation_rmse"],1e-8)
        self.assertAlmostEqual(report["models"][0]["terms"][0]["coefficient"],8.314,places=8)

    def test_inverse_square_gravity(self):
        radius=self.rng.uniform(1,10,300)
        report=self.run_model({"variables":{"r":radius},"targets":{"g":12/radius**2}},
                              options(term=["r**(-2)"]))
        self.assertLess(report["models"][0]["validation_rmse"],1e-9)

    def test_polynomial_degree_five(self):
        x=self.rng.uniform(-2,2,300)
        report=self.run_model({"variables":{"x":x},"targets":{"y":2*x**5-3*x**2+1}},
                              options(degree=5))
        self.assertLess(report["models"][0]["validation_rmse"],1e-9)
        self.assertEqual(report["models"][0]["selected_terms"],3)

    def test_composite_fractional_and_periodic(self):
        x=self.rng.uniform(.3,3,400)
        y=2*np.sin(3*x)+.7*np.exp(-x*x)+3*x**1.5
        report=self.run_model({"variables":{"x":x},"targets":{"y":y}},
                              options(degree=1,functions="sin,cos",frequencies="1,2,3",
                                      term=["exp(-x*x)","x**1.5"]))
        self.assertLess(report["models"][0]["validation_rmse"],1e-8)

    def test_noise(self):
        x=self.rng.normal(size=400)
        y=3*x+1+self.rng.normal(0,.05,400)
        report=self.run_model({"variables":{"x":x},"targets":{"y":y}},options())
        self.assertLess(report["models"][0]["validation_rmse"],.08)
        self.assertGreater(report["models"][0]["validation_rmse"],.02)

    def test_exponential_decay_ode(self):
        t=np.linspace(0,3,1200)
        x=3*np.exp(-.7*t)
        report=self.run_model({"t":t,"states":{"x":x}},options("ode"))
        model=report["models"][0]
        self.assertEqual(model["selected_terms"],1)
        self.assertAlmostEqual(model["terms"][0]["coefficient"],-.7,places=5)
        self.assertLess(model["validation_rmse"],1e-5)

    def test_second_order_oscillator(self):
        t=np.linspace(0,12,2400)
        report=self.run_model({"t":t,"states":{"x":np.cos(2*t)}},options("ode",order=2))
        coefficient=report["models"][0]["terms"][0]["coefficient"]
        self.assertAlmostEqual(coefficient,-4,places=3)
        self.assertLess(report["models"][0]["validation_rmse"],1e-3)

    def test_fourth_order_oscillator(self):
        t=np.linspace(0,8,1000)
        report=self.run_model({"t":t,"states":{"x":np.cos(1.3*t)}},
                              options("ode",order=4,degree=1))
        # Lower derivatives of one sinusoid are correlated, so a unique law is not identifiable.
        self.assertLess(report["models"][0]["validation_rmse"],.002)
        self.assertTrue(any("rank deficient" in note for note in report["notes"]))

    def test_independent_ode_rollout(self):
        t=np.linspace(0,2,800)
        report=self.run_model({"t":t,"states":{"x":3*np.exp(-.7*t)}},options("ode"))
        terms=report["models"][0]["terms"]
        def learned_rhs(t,state):
            env={"x":np.array([state[0]])}
            return [sum(term["coefficient"]*engine.expression(term["expression"],env)[0] for term in terms)]
        unseen_t=np.linspace(0,4,200)
        prediction=solve_ivp(learned_rhs,(0,4),[7.],t_eval=unseen_t,rtol=1e-9,atol=1e-10).y[0]
        self.assertLess(np.max(np.abs(prediction-7*np.exp(-.7*unseen_t))),1e-5)

    def test_lorenz_coupled_ode(self):
        t=np.linspace(0,8,4000)
        def rhs(t,state):
            x,y,z=state
            return [10*(y-x),x*(28-z)-y,x*y-(8/3)*z]
        trajectories=[]
        for initial in ([1,2,3],[-4,3,8]):
            states=solve_ivp(rhs,(0,8),initial,t_eval=t,rtol=1e-11,atol=1e-12).y
            trajectories.append({"t":t,"states":dict(zip(["x","y","z"],states))})
        report=self.run_model({"trajectories":trajectories},options("ode"))
        expected=[{"x":-10,"y":10},{"x":28,"y":-1,"x*z":-1},{"x*y":1,"z":-8/3}]
        for model,wanted in zip(report["models"],expected):
            found={term["expression"]:term["coefficient"] for term in model["terms"]}
            self.assertEqual(set(found),set(wanted))
            for name,value in wanted.items():
                self.assertAlmostEqual(found[name],value,delta=.03)
            self.assertLess(model["validation_rmse"],.15)
        self.assertEqual(report["validation_strategy"],"last trajectory held out")

    def test_heat_pde(self):
        t=np.linspace(0,1,100)
        x=np.linspace(-np.pi,np.pi,150)
        nu=.3
        u=sum(amplitude*np.exp(-nu*k*k*t[:,None])*np.sin(k*x[None,:])
              for k,amplitude in [(1,1),(2,.4),(3,.2)])
        report=self.run_model({"t":t,"axes":[x],"u":u},options("pde",order=2,degree=1))
        model=report["models"][0]
        found={term["expression"]:term["coefficient"] for term in model["terms"]}
        self.assertIn("u_x0_x0",found)
        self.assertAlmostEqual(found["u_x0_x0"],nu,delta=.006)
        self.assertLess(model["validation_rmse"],.002)

    def test_implicit_circle(self):
        theta=self.rng.uniform(0,2*np.pi,400)
        report=self.run_model({"variables":{"x":2*np.cos(theta),"y":2*np.sin(theta)}},
                              options("implicit"))
        self.assertLess(report["models"][0]["normalized_validation_residual"],1e-8)
        self.assertEqual(report["models"][0]["near_null_directions"],1)

    def test_two_dimensional_mixed_pde(self):
        t=np.linspace(0,1,40)
        x=np.linspace(-np.pi,np.pi,55)
        y=np.linspace(-np.pi,np.pi,55)
        u=sum(amplitude*np.exp(-(.2*(k*k+l*l)+.1*k*l)*t[:,None,None])
              *np.sin(k*x[None,:,None]+l*y[None,None,:])
              for k,l,amplitude in [(1,1,1),(2,1,.7),(1,-2,.4),(2,2,.3)])
        report=self.run_model({"t":t,"axes":[x,y],"u":u},options("pde",order=2,degree=1))
        model=report["models"][0]
        found={term["expression"]:term["coefficient"] for term in model["terms"]}
        self.assertAlmostEqual(found.get("u_x0_x1",0),.1,delta=.005)
        self.assertAlmostEqual(found.get("u_x0_x0",0),.2,delta=.005)
        self.assertAlmostEqual(found.get("u_x1_x1",0),.2,delta=.005)
        self.assertLess(model["validation_rmse"],.001)

    def test_original_nine_families(self):
        spec=importlib.util.spec_from_file_location("original",SCRIPT.with_name("fit_equation.py"))
        original=importlib.util.module_from_spec(spec)
        spec.loader.exec_module(original)
        x=np.linspace(1,5,25)
        cases={"constant":np.full_like(x,7),"linear":2*x+1,"quadratic":3*x*x-2*x+4,
               "cubic":x**3-2*x+1,"exponential":2*np.exp(.7*x),"power":4*x**1.7,
               "logarithmic":3+2*np.log(x),"reciprocal":2+3/x,"square-root":1+4*np.sqrt(x)}
        for family,values in cases.items():
            with self.subTest(family=family):
                report=original.analyze(x,values)
                self.assertEqual(report["recommended"]["family"],family)
                self.assertLess(report["recommended"]["rmse"],1e-8)

    def test_csv_and_advanced_cli(self):
        with tempfile.TemporaryDirectory() as folder:
            csv=Path(folder)/"data.csv"
            csv.write_text("x,y\n1,3\n2,5\n3,7\n4,9\n",encoding="utf-8-sig")
            run=subprocess.run([sys.executable,str(SCRIPT.with_name("fit_equation.py")),"--csv",str(csv)],capture_output=True,text=True)
            self.assertEqual(run.returncode,0,run.stderr)
            self.assertEqual(json.loads(run.stdout)["recommended"]["family"],"linear")
            data=Path(folder)/"data.json"
            x=list(range(40))
            data.write_text(json.dumps({"variables":{"x":x},"targets":{"y":[2*v+1 for v in x]}}))
            run=subprocess.run([sys.executable,str(SCRIPT),"--data",str(data),"--functions","none"],capture_output=True,text=True)
            self.assertEqual(run.returncode,0,run.stderr)
            self.assertLess(json.loads(run.stdout)["models"][0]["validation_rmse"],1e-8)

    def test_security_no_python_execution(self):
        env={"x":np.arange(1.,20.)}
        for expr in ["__import__('os').system('echo bad')","x[0]","(lambda: 1)()", "open('secret')", "x.__class__"]:
            with self.assertRaises(ValueError):
                engine.expression(expr,env)

    def test_domain_skip_not_row_removal(self):
        x=np.linspace(-2,2,100)
        report=self.run_model({"variables":{"x":x},"targets":{"y":2*x}},
                              options(functions="log,sqrt"))
        self.assertEqual(len(report["skipped_domain_terms"]),2)
        model=report["models"][0]
        self.assertEqual(model["training_rows"]+model["validation_rows"],len(x))

    def test_invalid_data_and_time(self):
        with self.assertRaises(ValueError):
            engine.variables({"x":[1,None,3]})
        with self.assertRaises(ValueError):
            engine.time_axis([0,1,1,2],minimum=3)
        with self.assertRaises(ValueError):
            self.run_model({"variables":{"x":[1]*100},"targets":{"y":list(range(100))}},options())

    def test_duplicate_input_groups(self):
        x=np.repeat(np.arange(100.),3)
        train,valid,_=engine.split_rows({"x":x})
        self.assertEqual(len(set(x[train]) & set(x[valid])),0)

    def test_nonuniform_time_smoothing_rejected(self):
        t=np.linspace(0,3,100)**1.1
        with self.assertRaises(ValueError):
            self.run_model({"t":t,"states":{"x":np.exp(-t)}},options("ode",smooth_window=7))


if __name__=="__main__":
    unittest.main(verbosity=2)
