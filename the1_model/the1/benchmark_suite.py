"""
PhysicsBenchmarkSuite: Comprehensive benchmark of natural laws for Symbolic World Models.
Includes classical mechanics, gravitation, thermodynamics, electromagnetism,
relativity, quantum phenomena, and optics.
"""

import numpy as np
from typing import Dict, Any, List, Tuple

class NaturalLawBenchmark:
    def __init__(self,
                 name: str,
                 category: str,
                 target_name: str,
                 var_names: List[str],
                 formula_str: str,
                 eval_fn,
                 domain: List[Tuple[float, float]],
                 ood_domain: List[Tuple[float, float]],
                 description: str = ""):
        self.name = name
        self.category = category
        self.target_name = target_name
        self.var_names = var_names
        self.formula_str = formula_str
        self.eval_fn = eval_fn
        self.domain = domain
        self.ood_domain = ood_domain
        self.description = description

    def sample_data(self,
                    n_samples: int,
                    ood: bool = False,
                    noise_std: float = 0.0,
                    seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
        rng = np.random.RandomState(seed)
        domain = self.ood_domain if ood else self.domain
        
        # Draw samples uniformly within bounds
        cols = []
        for low, high in domain:
            cols.append(rng.uniform(low, high, size=n_samples))
        X = np.column_stack(cols)
        
        y = self.eval_fn(X)
        if noise_std > 0.0:
            scale = np.std(y) if np.std(y) > 1e-6 else 1.0
            y += rng.normal(0, noise_std * scale, size=y.shape)
            
        return X, y


class PhysicsBenchmarkSuite:
    """Standardized suite of fundamental physics laws."""
    
    @staticmethod
    def get_all_laws() -> Dict[str, NaturalLawBenchmark]:
        laws = {}

        # 1. Newton's Second Law: F = m * a
        laws["newton_second"] = NaturalLawBenchmark(
            name="Newton's Second Law",
            category="Classical Mechanics",
            target_name="F",
            var_names=["m", "a"],
            formula_str="m * a",
            eval_fn=lambda X: X[:, 0] * X[:, 1],
            domain=[(0.5, 5.0), (0.5, 10.0)],
            ood_domain=[(5.0, 15.0), (10.0, 25.0)],
            description="Force equals mass times acceleration."
        )

        # 2. Universal Gravitation: F = G * (m1 * m2) / r^2 (setting G=1 or scaled)
        laws["gravitation"] = NaturalLawBenchmark(
            name="Universal Gravitation",
            category="Astrophysics",
            target_name="F_g",
            var_names=["m1", "m2", "r"],
            formula_str="6.674e-11 * m1 * m2 / (r^2)",
            eval_fn=lambda X: 6.674e-11 * X[:, 0] * X[:, 1] / (X[:, 2]**2),
            domain=[(1e3, 1e4), (1e3, 1e4), (10.0, 50.0)],
            ood_domain=[(1e4, 5e4), (1e4, 5e4), (50.0, 150.0)],
            description="Gravitational force between two massive bodies."
        )
        
        # Standard Normalized Gravitation (c=1 scale for pure functional discovery)
        laws["gravitation_normalized"] = NaturalLawBenchmark(
            name="Normalized Gravitation",
            category="Classical Mechanics",
            target_name="F",
            var_names=["m1", "m2", "r"],
            formula_str="m1 * m2 / (r^2)",
            eval_fn=lambda X: X[:, 0] * X[:, 1] / (X[:, 2]**2),
            domain=[(1.0, 10.0), (1.0, 10.0), (1.0, 5.0)],
            ood_domain=[(10.0, 20.0), (10.0, 20.0), (5.0, 10.0)],
            description="Gravitational inverse-square interaction."
        )

        # 3. Kinetic Energy: E = 0.5 * m * v^2
        laws["kinetic_energy"] = NaturalLawBenchmark(
            name="Kinetic Energy",
            category="Classical Mechanics",
            target_name="E_k",
            var_names=["m", "v"],
            formula_str="0.5 * m * (v^2)",
            eval_fn=lambda X: 0.5 * X[:, 0] * (X[:, 1]**2),
            domain=[(0.5, 10.0), (0.5, 8.0)],
            ood_domain=[(10.0, 25.0), (8.0, 20.0)],
            description="Energy of a moving mass."
        )

        # 4. Ideal Gas Law: P = n * R * T / V (with R=8.314)
        laws["ideal_gas"] = NaturalLawBenchmark(
            name="Ideal Gas Law",
            category="Thermodynamics",
            target_name="P",
            var_names=["n", "T", "V"],
            formula_str="8.314 * n * T / V",
            eval_fn=lambda X: 8.314 * X[:, 0] * X[:, 1] / X[:, 2],
            domain=[(1.0, 5.0), (273.0, 373.0), (0.1, 1.0)],
            ood_domain=[(5.0, 10.0), (373.0, 500.0), (1.0, 3.0)],
            description="Equation of state of a hypothetical ideal gas."
        )

        # 5. Coulomb's Law: F = k_e * q1 * q2 / r^2 (k_e = 8.9875e9 or normalized)
        laws["coulomb"] = NaturalLawBenchmark(
            name="Coulomb's Electrostatic Law",
            category="Electromagnetism",
            target_name="F_e",
            var_names=["q1", "q2", "r"],
            formula_str="q1 * q2 / (r^2)",
            eval_fn=lambda X: X[:, 0] * X[:, 1] / (X[:, 2]**2),
            domain=[(0.5, 5.0), (0.5, 5.0), (1.0, 4.0)],
            ood_domain=[(5.0, 10.0), (5.0, 10.0), (4.0, 8.0)],
            description="Electrostatic force between charged particles."
        )

        # 6. Einstein's Mass-Energy Equivalence: E = m * c^2 (normalized c=3 or c=1)
        laws["mass_energy"] = NaturalLawBenchmark(
            name="Mass-Energy Equivalence",
            category="Relativity",
            target_name="E",
            var_names=["m"],
            formula_str="m * 9",  # with c=3
            eval_fn=lambda X: X[:, 0] * 9.0,
            domain=[(0.1, 10.0)],
            ood_domain=[(10.0, 50.0)],
            description="Einstein's mass-energy conversion with normalized c=3."
        )

        # 7. Relativistic Lorentz Factor: gamma = 1 / sqrt(1 - v^2)
        laws["lorentz_factor"] = NaturalLawBenchmark(
            name="Lorentz Factor (Time Dilation)",
            category="Relativity",
            target_name="gamma",
            var_names=["v"],
            formula_str="1 / sqrt(1 - v^2)",
            eval_fn=lambda X: 1.0 / np.sqrt(np.maximum(1.0 - (X[:, 0]**2), 1e-9)),
            domain=[(0.05, 0.75)],
            ood_domain=[(0.75, 0.95)],
            description="Relativistic factor determining time dilation and length contraction."
        )

        # 8. Kepler's Third Law: T = k * a^(3/2) (T^2 proportional to a^3)
        laws["kepler_third"] = NaturalLawBenchmark(
            name="Kepler's Third Law",
            category="Astrophysics",
            target_name="T",
            var_names=["a"],
            formula_str="2 * a^(1.5)",
            eval_fn=lambda X: 2.0 * np.power(X[:, 0], 1.5),
            domain=[(1.0, 8.0)],
            ood_domain=[(8.0, 25.0)],
            description="Orbital period scaling with semi-major axis."
        )

        # 9. Simple Pendulum: T = 2 * pi * sqrt(L / g) (with g=9.81 => ~2.007 * sqrt(L))
        laws["pendulum_period"] = NaturalLawBenchmark(
            name="Simple Pendulum Period",
            category="Classical Mechanics",
            target_name="T",
            var_names=["L"],
            formula_str="2.006 * sqrt(L)",
            eval_fn=lambda X: (2 * np.pi / np.sqrt(9.81)) * np.sqrt(X[:, 0]),
            domain=[(0.2, 3.0)],
            ood_domain=[(3.0, 10.0)],
            description="Harmonic period of a simple pendulum."
        )

        # 10. Snell's Law: n2 = n1 * sin(theta1) / sin(theta2)
        laws["snells_law"] = NaturalLawBenchmark(
            name="Snell's Law of Refraction",
            category="Optics",
            target_name="n2",
            var_names=["n1", "theta1", "theta2"],
            formula_str="n1 * sin(theta1) / sin(theta2)",
            eval_fn=lambda X: X[:, 0] * np.sin(X[:, 1]) / np.sin(X[:, 2]),
            domain=[(1.0, 2.5), (0.2, 1.2), (0.2, 1.2)],
            ood_domain=[(2.5, 4.0), (1.2, 1.5), (1.2, 1.5)],
            description="Law of refraction for light passing between media."
        )

        # 11. Stefan-Boltzmann Law: j = sigma * T^4 (normalized sigma=5.67e-8 or normalized)
        laws["stefan_boltzmann"] = NaturalLawBenchmark(
            name="Stefan-Boltzmann Radiation Law",
            category="Thermodynamics",
            target_name="j",
            var_names=["T"],
            formula_str="0.5 * (T^4)",
            eval_fn=lambda X: 0.5 * (X[:, 0]**4),
            domain=[(0.5, 4.0)],
            ood_domain=[(4.0, 8.0)],
            description="Total blackbody emissive power proportional to absolute temperature."
        )

        # 12. Radioactive Exponential Decay: N(t) = N0 * exp(-lambda * t)
        laws["radioactive_decay"] = NaturalLawBenchmark(
            name="Radioactive Decay",
            category="Nuclear Physics",
            target_name="N",
            var_names=["N0", "lambda_const", "t"],
            formula_str="N0 * exp(-lambda_const * t)",
            eval_fn=lambda X: X[:, 0] * np.exp(-X[:, 1] * X[:, 2]),
            domain=[(50.0, 100.0), (0.1, 0.5), (0.5, 4.0)],
            ood_domain=[(100.0, 200.0), (0.5, 1.0), (4.0, 8.0)],
            description="Exponential decay of radioactive isotopes over time."
        )

        # 13. Hooke's Law: F = -k * x
        laws["hooke_law"] = NaturalLawBenchmark(
            name="Hooke's Law of Elasticity",
            category="Classical Mechanics",
            target_name="F",
            var_names=["k", "x"],
            formula_str="-k * x",
            eval_fn=lambda X: -X[:, 0] * X[:, 1],
            domain=[(10.0, 50.0), (0.01, 0.5)],
            ood_domain=[(50.0, 100.0), (0.5, 1.5)],
            description="Restoring force exerted by a spring."
        )

        # 14. Bernoulli's Dynamic Pressure: q = 0.5 * rho * v^2
        laws["bernoulli_pressure"] = NaturalLawBenchmark(
            name="Bernoulli's Dynamic Pressure",
            category="Fluid Dynamics",
            target_name="q",
            var_names=["rho", "v"],
            formula_str="0.5 * rho * (v^2)",
            eval_fn=lambda X: 0.5 * X[:, 0] * (X[:, 1]**2),
            domain=[(1.0, 5.0), (2.0, 10.0)],
            ood_domain=[(5.0, 15.0), (10.0, 30.0)],
            description="Kinetic energy per unit volume of fluid."
        )

        # 15. Wave Speed: v = f * lambda
        laws["wave_speed"] = NaturalLawBenchmark(
            name="Wave Dispersion Velocity",
            category="Wave Mechanics",
            target_name="v",
            var_names=["f", "lambda"],
            formula_str="f * lambda",
            eval_fn=lambda X: X[:, 0] * X[:, 1],
            domain=[(10.0, 100.0), (0.1, 2.0)],
            ood_domain=[(100.0, 500.0), (2.0, 10.0)],
            description="Velocity of propagation of periodic waves."
        )

        return laws
