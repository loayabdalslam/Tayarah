"""
The1: An Inductive Neuro-Symbolic World Model for Extreme Few-Shot Natural Law Discovery.
"""

from .core import The1WorldModel, The1Model, SymbolicLaw
from .pipeline import pipeline, The1Pipeline, SymbolicWorldPipeline, PipelineDiscoveryResult
from .benchmark_suite import PhysicsBenchmarkSuite
from .math_suite import MathBenchmarkSuite, MathLawBenchmark
from .search import GenericSearch, Candidate
from .library import LawLibrary
from .explorer import DatasetExplorer, DiscoveryReport
from .lean4 import LeanVerifier, LeanHypothesis, sympy_to_lean, LeanTranslationError
from .constants import rationalize, snap_constant

# Aliases for convenience
The1 = The1WorldModel
Model = The1WorldModel

__version__ = "1.1.0"
__all__ = [
    "The1WorldModel",
    "The1Model",
    "The1",
    "Model",
    "SymbolicLaw",
    "pipeline",
    "The1Pipeline",
    "SymbolicWorldPipeline",
    "PipelineDiscoveryResult",
    "PhysicsBenchmarkSuite",
    "MathBenchmarkSuite",
    "MathLawBenchmark",
    "GenericSearch",
    "Candidate",
    "LawLibrary",
    "DatasetExplorer",
    "DiscoveryReport",
    "LeanVerifier",
    "LeanHypothesis",
    "LeanTranslationError",
    "sympy_to_lean",
    "rationalize",
    "snap_constant",
]
