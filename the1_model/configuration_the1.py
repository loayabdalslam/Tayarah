"""
Configuration class for The1 Inductive Symbolic World Model.
Inherits from Hugging Face PretrainedConfig.
"""

from transformers import PretrainedConfig

class The1Config(PretrainedConfig):
    model_type = "the1"

    def __init__(self,
                 max_complexity: int = 25,
                 population_size: int = 60,
                 generations: int = 20,
                 tournament_size: int = 3,
                 mutation_rate: float = 0.35,
                 crossover_rate: float = 0.65,
                 occam_weight: float = 0.01,
                 rational_snap_tol: float = 0.04,
                 **kwargs):
        super().__init__(**kwargs)
        self.max_complexity = max_complexity
        self.population_size = population_size
        self.generations = generations
        self.tournament_size = tournament_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.occam_weight = occam_weight
        self.rational_snap_tol = rational_snap_tol
