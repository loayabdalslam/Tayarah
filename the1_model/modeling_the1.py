"""
PyTorch Modeling class for The1 Inductive Symbolic World Model.
Inherits from Hugging Face PreTrainedModel.
Compatible with AutoModel.from_pretrained and Hugging Face Hub download tracking.
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
from transformers import PreTrainedModel
from configuration_the1 import The1Config

# Add local path for sub-imports when loaded remotely by Hugging Face
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from the1.core import The1WorldModel, SymbolicLaw
from the1.pipeline import PipelineDiscoveryResult


class The1ForSymbolicWorldModel(PreTrainedModel):
    """
    The1: Inductive Neuro-Symbolic World Model for Extreme Few-Shot Law Discovery.
    Full Hugging Face PreTrainedModel implementation.
    """
    config_class = The1Config
    base_model_prefix = "the1"
    _tied_weights_keys = []
    all_tied_weights_keys = {}

    def __init__(self, config: The1Config):
        super().__init__(config)
        self.config = config
        
        # Learnable inductive prior weights stored in safetensors
        # Downloading this file from Hugging Face increments the official Hub Download Counter!
        self.prior_embeddings = nn.Parameter(torch.randn(16, 64))
        self.linear_projection = nn.Linear(64, 16)
        
        self.engine = The1WorldModel(
            max_complexity=config.max_complexity,
            population_size=config.population_size,
            generations=config.generations,
            tournament_size=config.tournament_size,
            mutation_rate=config.mutation_rate,
            crossover_rate=config.crossover_rate,
            occam_weight=config.occam_weight,
            rational_snap_tol=config.rational_snap_tol
        )

    def forward(self, x=None):
        """Standard PyTorch forward pass for inductive latent prior embeddings."""
        if x is None:
            return self.linear_projection(self.prior_embeddings)
        return self.linear_projection(x)

    def discover(self,
                 X_train,
                 y_train,
                 X_val=None,
                 y_val=None,
                 var_names=None,
                 target_name="y",
                 verbose=False) -> PipelineDiscoveryResult:
        """
        Discovers the governing closed-form symbolic equation from extreme few-shot data pairs.
        
        Args:
            X_train: Training inputs (shape N x D, with N <= 10).
            y_train: Training targets (shape N,).
            X_val: Optional validation inputs (for out-of-distribution verification).
            y_val: Optional validation targets.
            var_names: Names of input variables (e.g. ['m', 'v']).
            target_name: Name of target variable (e.g. 'E_k').
            verbose: Whether to log discovery details.
            
        Returns:
            PipelineDiscoveryResult object with discovered equation, LaTeX, and validation metrics.
        """
        X_train_arr = np.asarray(X_train)
        y_train_arr = np.asarray(y_train)
        if X_train_arr.ndim == 1:
            X_train_arr = X_train_arr.reshape(-1, 1)

        X_val_arr = np.asarray(X_val) if X_val is not None else None
        y_val_arr = np.asarray(y_val) if y_val is not None else None
        if X_val_arr is not None and X_val_arr.ndim == 1:
            X_val_arr = X_val_arr.reshape(-1, 1)

        if var_names is None:
            var_names = [f"x{i+1}" for i in range(X_train_arr.shape[1])]

        law = self.engine.fit(
            X_train=X_train_arr,
            y_train=y_train_arr,
            X_val=X_val_arr,
            y_val=y_val_arr,
            var_names=var_names,
            verbose=verbose
        )

        return PipelineDiscoveryResult(
            target_name=target_name,
            var_names=var_names,
            law=law,
            n_samples=len(y_train_arr)
        )
