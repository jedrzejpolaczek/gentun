#!/usr/bin/env python
"""
NSGANet class.
"""

import random
import functools
import numpy as np
from loguru import logger

try:
    from .nsga_2 import NSGA2
except ImportError:
    pass
try:
    from ..utils import bayesian_optimization_algorithm
except ImportError:
    pass


class NSGANet(NSGA2):  # TODO: add typing and docstring
    """
    Genetic algorithm used in the Neural Architecture Search using Multi-Objective Genetic Algorithm paper.
    """

    def __init__(self, population, crossover_probability: float=0.2, mutation_probability: float=0.8):
        super(NSGANet, self).__init__(population)
        self.crossover_probability = crossover_probability
        self.mutation_probability = mutation_probability
    
    def run(self, max_generations):  # TODO: add typing and docstring
        logger.info("Start exploration phase.")
        while self.generation <= max_generations:
            self.evolve_population()
            self.generation += 1

        logger.info("Start exploitation phase.")
        self.population = bayesian_optimization_algorithm(self.population)
