#!/usr/bin/env python
"""
NSGANet class.
"""

import random
import functools
import numpy as np
from loguru import logger

try:
    from .genetic_algorithm import GeneticAlgorithm
except ImportError:
    pass
try:
    from ..utils import bayesian_optimization_algorithm
except ImportError:
    pass


class NSGANet(GeneticAlgorithm):
    """
    Genetic algorithm used in the Neural Architecture Search using Multi-Objective Genetic Algorithm paper.
    """

    def __init__(self, population, crossover_probability: float=0.2, mutation_probability: float=0.8):
        super(NSGANet, self).__init__(population)
        self.crossover_probability = crossover_probability
        self.mutation_probability = mutation_probability
        self.fitnesses: np.array = None
        self.fronts: list = None
        self.ranks: dict = None
        self.crowding_distance_metrics = None
        self.non_domiated_sorted_indicies = None
    
    def run(self, max_generations):  # TODO: add typing and docstring
        logger.info("Starting genetic algorithm.")
        logger.info("Start exploration phase.")
        while self.generation <= max_generations:
            self.evolve_population()
            self.generation += 1

        logger.info("Start exploitation phase.")
        self.population = bayesian_optimization_algorithm(self.population)

    def evolve_population(self):  # TODO: add typing and docstring
        if self.population.get_size() < self.tournament_size:
            msg = "Population size is smaller than tournament size."
            logger.error(msg)
            raise ValueError(msg)
        
        logger.info("Evaluating generation #{}".format(self.generation))
        logger.info("Population size: {}".format(self.population.get_size()))
        logger.info("Fittest individual is: {}".format(self.population.get_fittest()))
        logger.info("Fitness value is: {}".format((self.population.get_fittest().get_fitness())))

        logger.debug("Calculate fitnesses.")
        self.get_fitnesses()

        logger.debug("Calculate the pareto fronts.")
        self.get_fronts()

        logger.debug("Calculate ranks.")
        self.get_ranks()
        
        logger.debug("Calculate crowding distance.")
        self.get_crowding_distance_metrics()

        logger.debug("Sorting population indexes using nondominated sort.")
        self.non_domiated_sorted_indicies = self.nondominated_sort()
        
        logger.debug("Evolving population.")
        new_population = self.create_new_population()
        
        logger.debug("Saving new population in place of current population.")
        self.population.individuals = new_population

    def get_fitnesses(self) -> np.array:  # TODO: add typing and docstring
        if self.fitnesses == None:
            self.fitnesses = self.calculate_fitnesses()
        
        return self.fitnesses
    
    def get_fronts(self) -> list:  # TODO: add typing and docstring
        if self.fronts == None:
            self.fronts = self.fast_nondominated_sort()
        
        return self.fronts
    
    def get_ranks(self) -> list:  # TODO: add typing and docstring
        if self.ranks == None:
            self.ranks = self.fronts_to_nondomination_rank()
        
        return self.ranks
    
    def get_crowding_distance_metrics(self) -> list:  # TODO: add typing and docstring
        if self.crowding_distance_metrics == None:
            self.crowding_distance_metrics = self.calculate_crowding_distance_metrics()
        
        return self.crowding_distance_metrics
    
    def calculate_fitnesses(self) -> np.array:  # TODO: add typing and docstring
        fitnesses = []
        for individual in self.population.individuals:
            fitnesses.append(individual.get_fitness())
        
        assert self.population.get_size() == len(fitnesses)

        logger.debug("Fitnesses (type {} and size {}): \n{}".format(type(fitnesses), len(fitnesses), fitnesses))
        return np.array(fitnesses)
    
    def normalize_fitnesses(self) -> np.array:  # TODO: add typing and docstring
        self.fitnesses = (self.fitnesses - np.min(self.fitnesses)) / (np.max(self.fitnesses) - np.min(self.fitnesses))
        logger.debug("Normalize fitnesses (type {} and size {}): {}".format(type(self.fitnesses), len(self.fitnesses), self.fitnesses))
    
    def fast_nondominated_sort(self) :  # TODO: add typing and docstring
        """
        Calculate Pareto fronts
        The population is sorted and partitioned into fronts (F1, F2, etc.), 
        where F1 (first front) indicates the approximated Pareto front.

        :return list: fronts is a list of fronts, each front contain index of each individual for self.population.individuals
        """
        logger.debug("Calculate dominated set for each individual.")
        domination_sets = []
        domination_counts = []
        for individual in self.population.individuals:
            current_domination_set = set()
            domination_counts.append(0)
            for i, other_individual in enumerate(self.population.individuals):
                if self.dominates(individual, other_individual):
                    current_domination_set.add(i)
                elif self.dominates(other_individual, individual):
                    domination_counts[-1] += 1

            domination_sets.append(current_domination_set)
        domination_counts = np.array(domination_counts)

        fronts = []
        while True:
            current_front = np.where(domination_counts==0)[0]
            if len(current_front) == 0:
                logger.debug("Current front have no individuals. Therefore we stop looking for new fronts.")
                break
            logger.debug("Front (type {} and size {}): {}".format(type(current_front), len(current_front), current_front))
            fronts.append(current_front)

            for individual in current_front:
                # this individual is already accounted for, make it -1 so  ==0 will not find it anymore
                domination_counts[individual] = -1 
                dominated_by_current_set = domination_sets[individual]
                for dominated_by_current in dominated_by_current_set:
                    domination_counts[dominated_by_current] -= 1
        
        logger.debug("Front (type {} and size {}): {}".format(type(fronts), len(fronts), fronts))
        return fronts
    
    def dominates(self, individual, other_individual):  # TODO: add typing and docstring
        larger_or_equal = individual.get_fitness() >= other_individual.get_fitness()
        larger = individual.get_fitness() > other_individual.get_fitness()

        return np.all(larger_or_equal) and np.any(larger)  # We are using np.all in case fitness would be array
    
    def fronts_to_nondomination_rank(self) -> dict:  # TODO: add typing and docstring
        """
        
        :return dict: dictinary of indexes of each individual for self.population.individuals
        """
        nondomination_rank_dict = {}
        for i, front in enumerate(self.fronts):
            for x in front:   
                nondomination_rank_dict[x] = i

        return nondomination_rank_dict

    def calculate_crowding_distance_metrics(self):  # TODO: add typing and docstring
        """
        Crowding Distance is a mechanism of ranking among members of a front, 
        which are dominating or dominated by each other.
        """
        # Prepare necessary data
        number_of_individuals = self.population.get_size()
        crowding_distance_metrics = np.zeros(number_of_individuals)
        number_of_objectives = len(self.fitnesses[0])
        
        # Normalise each objectives, so they are in the range [0,1]
        # This is necessary, so each objective's contribution have the same magnitude to the crowding metric.
        self.normalize_fitnesses()

        # Calculate crowding distance metrics
        for front in self.fronts:
            for objective_i in range(number_of_objectives):
                # logger.debug("fitnesses (type: {}): {}".format(type(self.fitnesses), self.fitnesses))
                # logger.debug("front (type: {}): {}".format(type(front), front))
                # logger.debug("objective_i (type: {}): {}".format(type(objective_i), objective_i))
                
                sorted_front = sorted(front, key = lambda x : self.fitnesses[x, objective_i])
                # logger.debug("sorted_front (type: {}): {}".format(type(sorted_front), sorted_front))
                # logger.debug("sorted_front[0] (type: {}): {}".format(type(sorted_front[0]), sorted_front[0]))
                # logger.debug("sorted_front[-1] (type: {}): {}".format(type(sorted_front[-1]), sorted_front[-1]))

                crowding_distance_metrics[sorted_front[0]] = np.inf
                crowding_distance_metrics[sorted_front[-1]] = np.inf
                if len(sorted_front) > 2:
                    for i in range(1,len(sorted_front)-1):
                        crowding_distance_metrics[sorted_front[i]] += self.fitnesses[sorted_front[i+1], objective_i] - self.fitnesses[sorted_front[i-1], objective_i]

        assert len(crowding_distance_metrics) == number_of_individuals
        logger.debug("crowding_distance_metrics (type: {}): {}".format(type(crowding_distance_metrics), crowding_distance_metrics))
        return crowding_distance_metrics
    
    def nondominated_sort(self):  # TODO: add typing and docstring
        
        number_of_individuals = len(self.crowding_distance_metrics)
        indicies = list(range(number_of_individuals))

        def nondominated_compare(a, b):
            # returns 1 if a dominates b, or if they equal, but a is less crowded
            # return -1 if b dominates a, or if they equal, but b is less crowded
            # returns 0 if they are equal in every sense
            
            
            if self.ranks[a] > self.ranks[b]:  # domination rank, smaller better
                return -1
            elif self.ranks[a] < self.ranks[b]:
                return 1
            else:
                # crowding metrics, larger better
                if self.crowding_distance_metrics[a] < self.crowding_distance_metrics[b]:   
                    return -1
                elif self.crowding_distance_metrics[a] > self.crowding_distance_metrics[b]:
                    return 1
                else:
                    return 0

        # decreasing order, the best is the first
        non_domiated_sorted_indicies = sorted(indicies, key = functools.cmp_to_key(nondominated_compare), reverse=True)

        logger.debug("non_domiated_sorted_indicies (type: {}): {}".format(type(non_domiated_sorted_indicies), non_domiated_sorted_indicies))
        return non_domiated_sorted_indicies

    def create_new_population(self):  # TODO: add typing and docstring
        """Survive current population, tournament, crossover them and mutate their children."""

        # logger.debug("Adjust survival wall for current population.")
        surviving_individuals = []
        for i in range(int(self.population.get_size()/2)):
            surviving_individuals.append(self.population.individuals[self.non_domiated_sorted_indicies[i]])

        offsprings = []
        for _ in range(self.population.get_size()/2):
            parent_first = self.tournament_select()
            parent_second = self.tournament_select()
            offspring = parent_first.crossover(parent_second)
            offspring.mutate()
            offsprings.append(offspring)

        new_population = surviving_individuals + offsprings
        
        assert len(new_population) == self.population.get_size()
        logger.debug("new_population (type: {}): {}".format(type(new_population), new_population))
        return new_population
    
    def tournament_select(self):  # TODO: add typing and docstring
        tournament = self.get_population_type()(
            self.population.get_species(), 
            self.x_train, 
            self.y_train, 
            individual_list=[
                self.population[i] for i in random.sample(range(self.population.get_size()), self.tournament_size)
            ], 
            maximize=self.population.get_fitness_criteria()
        )

        fittest_individual = tournament.get_fittest()

        logger.debug("fittest_individual (type: {}): {}".format(type(fittest_individual), fittest_individual))
        return tournament.get_fittest()

    # def tournament_select(self):  # TODO: add typing and docstring
    #     """


    #     :return self.population.__class__: fittest 
    #     """
    #     current_tournament_size = self.tournament_size
    #     if current_tournament_size > self.population.get_size():
    #         current_tournament_size = self.population.get_size()
        
    #     if current_tournament_size < 2:
    #         # Number of partners defined by tournament_size value
    #         print("Front type: {}".format(type(self.population.get_front(front_index))))
    #         print("Front size: {}".format(len(self.population.get_front(front_index))))
    #         print("Front size for population: {}".format(self.population.get_front_size(front_index)))
    #         print("Front: {}".format(self.population.get_front(front_index)))
    #         for i in random.sample(range(self.population.get_front_size(front_index)), self.population.get_front_size(front_index)):
    #             print("Type of i in for: {}".format(type(i)))
    #             break
    #         random_crossover_partners=[
    #                 self.population[i] for i in random.sample(range(self.population.get_size()), current_tournament_size)
    #             ]

    #         tournament_participants = self.get_population_type()(
    #             self.population.get_species(), 
    #             self.x_train, 
    #             self.y_train, 
    #             individual_list=random_crossover_partners,
    #             maximize=self.population.get_fitness_criteria()
    #         )

    #         # fittest_individual_from_tournament = tournament_participants.get_fittest()  # TODO: rethink overwright population class
    #         fittest_individual = None
    #         for participant in tournament_participants.individuals:
    #             if fittest_individual is None or (self.crowding_operator(participant, fittest_individual)):
    #                 fittest_individual = participant
    #     else:
    #         fittest_individual = self.population.individuals[0]

    #     return fittest_individual
    
    # def crowding_operator(self, individual, other_individual):  # TODO: add typing and docstring
    #     first_individual_higher_rank = individual.rank < other_individual.rank
    #     individuals_equal_rank = individual.rank == other_individual.rank
    #     first_individual_crowding_distance = individual.crowding_distance > other_individual.crowding_distance

    #     return first_individual_higher_rank or (individuals_equal_rank and first_individual_crowding_distance)
