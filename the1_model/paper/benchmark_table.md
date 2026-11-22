# The1 Empirical Physics Benchmark

| category            | law_name                       | ground_truth                        | discovered                        |   train_r2 |   val_r2 |   complexity |   latency_ms |
|:--------------------|:-------------------------------|:------------------------------------|:----------------------------------|-----------:|---------:|-------------:|-------------:|
| Classical Mechanics | Newton's Second Law            | F = m * a                           | F = a*m                           |          1 |        1 |            3 |     351.922  |
| Astrophysics        | Universal Gravitation          | F_g = 6.674e-11 * m1 * m2 / (r^2)   | F_g = 6.674e-11*m1*m2/r**2        |          1 |        1 |            7 |     301.895  |
| Classical Mechanics | Normalized Gravitation         | F = m1 * m2 / (r^2)                 | F = m1*m2/r**2                    |          1 |        1 |            6 |     655.708  |
| Classical Mechanics | Kinetic Energy                 | E_k = 0.5 * m * (v^2)               | E_k = 0.5*m*v**2                  |          1 |        1 |            5 |     342.96   |
| Thermodynamics      | Ideal Gas Law                  | P = 8.314 * n * T / V               | P = 8.314*T*n/V                   |          1 |        1 |            6 |     500.678  |
| Electromagnetism    | Coulomb's Electrostatic Law    | F_e = q1 * q2 / (r^2)               | F_e = q1*q2/r**2                  |          1 |        1 |            6 |     682.778  |
| Relativity          | Mass-Energy Equivalence        | E = m * 9                           | E = 9*m                           |          1 |        1 |            2 |      97.2648 |
| Relativity          | Lorentz Factor (Time Dilation) | gamma = 1 / sqrt(1 - v^2)           | gamma = 1/sqrt(1 - v**2)          |          1 |        1 |            6 |     597.667  |
| Astrophysics        | Kepler's Third Law             | T = 2 * a^(1.5)                     | T = 2*a**(3/2)                    |          1 |        1 |            4 |      99.9041 |
| Classical Mechanics | Simple Pendulum Period         | T = 2.006 * sqrt(L)                 | T = 2.00606668071065*sqrt(Abs(L)) |          1 |        1 |            5 |     105.242  |
| Optics              | Snell's Law of Refraction      | n2 = n1 * sin(theta1) / sin(theta2) | n2 = n1*sin(theta1)/sin(theta2)   |          1 |        1 |            7 |     672.438  |
| Thermodynamics      | Stefan-Boltzmann Radiation Law | j = 0.5 * (T^4)                     | j = 0.5*T**4                      |          1 |        1 |            3 |     295.565  |
| Nuclear Physics     | Radioactive Decay              | N = N0 * exp(-lambda_const * t)     | N = N0*exp(-lambda_const*t)       |          1 |        1 |            6 |     445.503  |
| Classical Mechanics | Hooke's Law of Elasticity      | F = -k * x                          | F = -k*x                          |          1 |        1 |            4 |     160.61   |
| Fluid Dynamics      | Bernoulli's Dynamic Pressure   | q = 0.5 * rho * (v^2)               | q = 0.5*rho*v**2                  |          1 |        1 |            5 |     318.329  |
| Wave Mechanics      | Wave Dispersion Velocity       | v = f * lambda                      | v = f*lambda                      |          1 |        1 |            3 |     172.359  |