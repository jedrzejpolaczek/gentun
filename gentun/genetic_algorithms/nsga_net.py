#!/usr/bin/env python
"""
NSGANet class.
"""

import random

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
            raise ValueError("Population size is smaller than tournament size.")
        
        print("Evaluating generation #{}...".format(self.generation))
        fittest = self.population.get_fittest()
        print("Fittest individual is:")
        print(fittest)
        print("Fitness value is: {}\n".format(round(fittest.get_fitness(), 4)))

        print("Perform: Non-dominated sorting...")
        self.fast_nondominated_sort()

        print("Perform: Calculating crowding distance...")
        self.crowding_distance()
        
        print("Perform: Evolving population...")
        current_population_copy = self.get_population_type()(
            self.population.get_species(), 
            self.x_train, 
            self.y_train, 
            individual_list=[],
            maximize=self.population.get_fitness_criteria()
        )
        new_population = self.create_new_population(current_population_copy)
        
        print("Perform: Saving new population as current population...")
        self.population = new_population

    def fast_nondominated_sort(self):  # TODO: add typing and docstring
        """
        The population is sorted and partitioned into fronts (F1, F2, etc.), 
        where F1 (first front) indicates the approximated Pareto front.
        """
        self.population.fronts = [[]]

        # Calculate Pareto front
        for individual in self.population.individual_list:
            individual.domination_count = 0
            individual.dominated_solutions = []
            for other_individual in self.population.individual_list:
                if individual.dominates(other_individual):
                    individual.dominated_solutions.append(other_individual)
                elif other_individual.dominates(individual):
                    individual.domination_count += 1
            if individual.domination_count == 0:
                individual.rank = 0
                self.population.fronts[0].append(individual)

        # Calculate rest of the fronts
        i = 0
        while len(self.population.fronts[i]) > 0:
            temp = []
            for individual in self.population.fronts[i]:
                for other_individual in individual.dominated_solutions:
                    other_individual.domination_count -= 1
                    if other_individual.domination_count == 0:
                        other_individual.rank = i + 1
                        temp.append(other_individual)
            i = i + 1
            self.population.fronts.append(temp)
    
    def crowding_distance(self):  # TODO: add typing and docstring
        """
        Crowding Distance is a mechanism of ranking among members of a front, 
        which are dominating or dominated by each other.
        """
        for front in self.population.fronts:
            if len(front) > 0:
                solutions_number = len(front)

                for individual in front:
                    individual.crowding_distance = 0

                for m in range(len(front[0].get_fitness())):
                    front.sort(key=lambda individual: individual.get_fitness())
                    front[0].crowding_distance = 10 ** 9
                    front[solutions_number - 1].crowding_distance = 10 ** 9

                    fitness_values = [individual.get_fitness() for individual in front]
                    scale = max(fitness_values) - min(fitness_values)
                    if scale == 0: 
                        scale = 1

                    for i in range(1, solutions_number - 1):
                        front[i].crowding_distance += (front[i + 1].get_fitness() - front[i - 1].get_fitness()) / scale

    def create_new_population(self, new_population):  # TODO: add typing and docstring
        """ """
        while new_population.get_size() < self.population.get_size():
            print("Perform: Reproduction...")
            child = self.reproduction()

            print("Perform: Mutation...")
            child.mutate()

            new_population.add_individual(child)
        
        return new_population

    def reproduction(self):  # TODO: add typing and docstring
        """Tournament, crossover and mutation"""
        partner_one = self.tournament_select()
        partner_two = partner_one
        while partner_one == partner_two:
                partner_two = self.tournament_select()

        child = partner_one.crossover(partner_two)

        return child
    
    def tournament_select(self):  # TODO: add typing and docstring
        """


        :return self.population.__class__: fittest 
        """
        # Number of partners defined by tournament_size value
        random_crossover_partners=[
                self.population[i] for i in random.sample(range(self.population.get_size()), self.tournament_size)
            ]

        tournament_participants = self.get_population_type()(
            self.population.get_species(), 
            self.x_train, 
            self.y_train, 
            individual_list=random_crossover_partners,  # Remember! There be only tournament_size number of individuals
            maximize=self.population.get_fitness_criteria()
        )

        # fittest_individual_from_tournament = tournament_participants.get_fittest()  # TODO: rethink overwright population class
        fittest_individual = None
        for participant in tournament_participants.individuals:
            if fittest_individual is None or (self.crowding_operator(participant, fittest_individual)):
                fittest_individual = participant

        return fittest_individual

    def crowding_operator(self, individual, other_individual):  # TODO: add typing and docstring
        first_individual_higher_rank = individual.rank < other_individual.rank
        individuals_equal_rank = individual.rank == other_individual.rank
        first_individual_crowding_distance = individual.crowding_distance > other_individual.crowding_distance

        return first_individual_higher_rank or (individuals_equal_rank and first_individual_crowding_distance)
