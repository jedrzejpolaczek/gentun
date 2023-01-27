# Make public APIs available at top-level import
from .populations import Population, GridPopulation
from .server import DistributedPopulation, DistributedGridPopulation
from .client import GentunClient

# Genetic Algorithms
try:
    from .genetic_algorithms.genetic_algorithm import GeneticAlgorithm
    from .genetic_algorithms.russian_roulette_genetic_algorithm import RussianRouletteGA
    from .genetic_algorithms.nsga_2 import NSGA2
    from .genetic_algorithms.nsga_net import NSGANet
except ImportError:
    print("Warning: install genetic algorithms to use GeneticAlgorithm, RussianRouletteGA, NSGA2 and NSGANet.")

# xgboost individuals and models
try:
    from .individuals.xgboost_individual import XgboostIndividual
    from .models.xgboost_models import XgboostModel
except ImportError:
    print("Warning: install xgboost to use XgboostIndividual and XgboostModel.")

# Keras individuals and models
try:
    from .individuals.genetic_cnn_individual import GeneticCnnIndividual
    from .models.keras_models import GeneticCnnModel
except ImportError:
    print("Warning: install Keras and TensorFlow to use GeneticCnnIndividual and GeneticCnnModel.")

# Keras X0 individuals and models
try:
    from .individuals.binary_string_network_representation_with_skip_bit_individual import BinaryStringNetworkRepresentationWithSkipBitIndividual
    from .models.binary_string_network_representation_with_skip_bit_model import BinaryStringNetworkRepresentationWithSkipBitModel
except ImportError:
    print("Warning: install Keras and TensorFlow to use GeneticCnnIndividual and GeneticCnnModel.")

# Utils functions
try:
    from .utils import bayesian_optimization_algorithm
except ImportError:
    print("Warning: install utils to use bayesian_optimization_algorithm.")