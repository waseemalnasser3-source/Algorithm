"""
Evolutionary Optimization Engine
Bio-inspired algorithmic implementation for intelligent attribute selection
Developed by: Rania Hassan (Evolution Engine Developer)
"""

from .chromosome import Chromosome
from .fitness import FitnessEvaluator
from .operators import GeneticOperators
from .ga_engine import GeneticAlgorithm

__all__ = ['Chromosome', 'FitnessEvaluator', 'GeneticOperators', 'GeneticAlgorithm']

