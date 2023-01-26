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

    def __init__(self, population, crossover_probability=0.2, mutation_probability=0.8):
        super(NSGANet, self).__init__(population)
        self.crossover_probability = crossover_probability
        self.mutation_probability = mutation_probability
        self.fitnesses = None
        self.fronts = None
        self.nondomination_ranks_dict = None
        self.crowding_distance_metrics = None
        self.non_domiated_sorted_indicies = None
    
    def run(self, max_generations):  # TODO: add typing and docstring
        print("Starting genetic algorithm...\n")
        print("Start exploration phase...\n")
        while self.generation <= max_generations:
            self.evolve_population()
            self.generation += 1

        print("Start exploitation phase...\n")
        self.population = bayesian_optimization_algorithm(self.population)

    def evolve_population(self):  # TODO: add typing and docstring
        if self.population.get_size() < self.tournament_size:
            msg = "Population size is smaller than tournament size."
            logger.error(msg)
            raise ValueError(msg)
        
        logger.info("Evaluating generation #{}...".format(self.generation))
        logger.info("Population size: {}".format(self.population.get_size()))
        logger.info("Fittest individual is: {}".format(self.population.get_fittest()))
        logger.info("Fitness value is: {}\n".format((self.population.get_fittest().get_fitness())))

        self.fitnesses = self.get_ftinesses()

        logger.debug("Calculate the pareto fronts.")
        self.fronts = self.fast_nondominated_sort()
        self.nondomination_ranks_dict = self.fronts_to_nondomination_rank()
        
        logger.debug("Calculate crowding distance.")
        self.crowding_distance_metrics = self.calculate_crowding_distance_metrics()

        logger.debug("Sorting population indexes using nondominated sort.")
        self.non_domiated_sorted_indicies = self.nondominated_sort()
        
        logger.debug("Evolving population.")
        new_population = self.create_new_population()
        
        logger.debug("Saving new population in place of current population.")
        self.population.individuals = new_population

    def get_ftinesses(self) -> list:  # TODO: add typing and docstring
        ftinesses = []
        for individual in self.population.individuals:
            ftinesses.append(individual.get_fitness())
        
        logger.debug("Population lenght/size: {}".format(self.population.get_size()))
        logger.debug("Fronts lenght/size: {}".format(len(ftinesses)))
        logger.debug("Fronts: {}".format(ftinesses))
        return ftinesses

    def fast_nondominated_sort(self) -> list[list]:  # TODO: add typing and docstring
        """
        Calculate Pareto fronts
        The population is sorted and partitioned into fronts (F1, F2, etc.), 
        where F1 (first front) indicates the approximated Pareto front.

        :return list: fronts is a list of fronts, each front contain index of each individual for self.population.individuals
        """
        # Calculate dominated set for each individual
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
            logger.debug("Front: {}".format(current_front))
            fronts.append(current_front)

            for individual in current_front:
                # this individual is already accounted for, make it -1 so  ==0 will not find it anymore
                domination_counts[individual] = -1 
                dominated_by_current_set = domination_sets[individual]
                for dominated_by_current in dominated_by_current_set:
                    domination_counts[dominated_by_current] -= 1
                
        logger.debug("Fronts lenght/size: {}".format(len(fronts)))
        logger.debug("Fronts: {}".format(fronts))
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
        num_objectives = self.fitnesses.shape[1]
        num_individuals = self.fitnesses.shape[0]
        
        # Normalise each objectives, so they are in the range [0,1]
        # This is necessary, so each objective's contribution have the same magnitude to the crowding metric.
        normalized_fitnesses = np.zeros_like(self.fitnesses)
        for objective_i in range(num_objectives):
            min_val = np.min(self.fitnesses[:,objective_i])
            max_val = np.max(self.fitnesses[:,objective_i])
            val_range = max_val - min_val
            normalized_fitnesses[:,objective_i] = (self.fitnesses[:,objective_i] - min_val) / val_range
        
        self.fitnesses = normalized_fitnesses
        crowding_distance_metrics = np.zeros(num_individuals)

        for front in self.fronts:
            for objective_i in range(num_objectives):
                
                sorted_front = sorted(front,key = lambda x : self.fitnesses[x,objective_i])
                
                crowding_distance_metrics[sorted_front[0]] = np.inf
                crowding_distance_metrics[sorted_front[-1]] = np.inf
                if len(sorted_front) > 2:
                    for i in range(1,len(sorted_front)-1):
                        crowding_distance_metrics[sorted_front[i]] += self.fitnesses[sorted_front[i+1],objective_i] - self.fitnesses[sorted_front[i-1],objective_i]

        return crowding_distance_metrics

    def nondominated_sort(self):  # TODO: add typing and docstring
        
        num_individuals = len(self.calculate_crowding_distance_metrics)
        indicies = list(range(num_individuals))

        def nondominated_compare(a, b):
            # returns 1 if a dominates b, or if they equal, but a is less crowded
            # return -1 if b dominates a, or if they equal, but b is less crowded
            # returns 0 if they are equal in every sense
            
            
            if self.nondomination_ranks_dict[a] > self.nondomination_ranks_dict[b]:  # domination rank, smaller better
                return -1
            elif self.nondomination_ranks_dict[a] < self.nondomination_ranks_dict[b]:
                return 1
            else:
                # crowding metrics, larger better
                if self.calculate_crowding_distance_metrics[a] < self.calculate_crowding_distance_metrics[b]:   
                    return -1
                elif self.calculate_crowding_distance_metrics[a] > self.calculate_crowding_distance_metrics[b]:
                    return 1
                else:
                    return 0

        # decreasing order, the best is the first
        non_domiated_sorted_indicies = sorted(indicies,key = functools.cmp_to_key(nondominated_compare),reverse=True)

        return non_domiated_sorted_indicies

    def create_new_population(self):  # TODO: add typing and docstring
        """Survive current population, tournament, crossover them and mutate their children."""

        logger.debug("adjust survival wall for current population.")
        surviving_individuals = []
        for i in range(int(self.population.get_size()/2)):
            surviving_individuals.append(self.population.individuals[self.non_domiated_sorted_indicies[i]])

        offsprings = []
        for _ in range(self.population.get_size()):
            logger.debug("Tournament current population to get fittest parents from tournament individuals set.")
            offspring_parents = self.tournament_select()

            logger.debug("Crossover tournament winner parrents.")
            offspring = offspring_parents[0].crossover(offspring_parents[0])

            logger.debug("Mutate offspring.")
            offspring.mutate()

            logger.debug("Save offspring.")
            offsprings.append(offspring)

        logger.debug("Create new population from survivals and offsprings.")
        new_population = surviving_individuals + offsprings
        
        return new_population
    
    # def touranment_selection(num_parents, num_offspring):  # TODO: add typing and docstring
    #     offspring_parents = []
    #     for _ in range(num_offspring):
    #         contestants = np.random.randint(0,num_parents,2) # generate 2 random numbers, take the smaller (parent list is already sorted, smaller index, better)
    #         winner = np.min(contestants)
    #         offspring_parents.append(winner)
        
    #     return offspring_parents
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
