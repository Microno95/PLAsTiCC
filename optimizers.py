import numpy as np
import typing
import multiprocessing as mp
import tqdm
import gc
from functools import partial

class RandomGenerator(object):
    def __init__(self, return_type, return_space, choice=False):
        self.is_choice = choice
        self.return_space = return_space
        self.return_type = return_type
        if return_type in (int, float, complex) and not self.is_choice:
            if return_type is int:
                def _gen():
                    return return_type(np.random.randint(self.return_space[0], self.return_space[1] + 1))
            elif return_type is float:
                def _gen():
                    return return_type(np.random.random() * (self.return_space[1] - self.return_space[0]) + self.return_space[0])
            else:
                def _gen():
                    return return_type(np.random.random() * (self.return_space[1].real - self.return_space[0].real) + np.random.random() * (self.return_space[1].imag - self.return_space[0].imag) * 1.j + self.return_space[0])
        else:
            def _gen():
                return return_type(np.random.choice(self.return_space))
        self._gen = _gen
    
    def __str__(self):
        return "{} with type: {} and return space: {}; is_choice={}".format(self.__class__.__name__, self.return_type, self.return_space, self.is_choice)
    
    def __iter__(self):
        return self
    
    def __next__(self):
        return self._gen()
    
class Individual(object):
    parameter_names = ['current']
    
    def __init__(self, current):
        self.is_fitness_computed = False
        self.parameters = {'current': current}
        self.fitness = 0
        
    def get_fitness(self) -> float:
        if not self.is_fitness_computed:
            self.fitness = np.random.random() * self.parameters['current']
            self.is_fitness_computed = True
        return self.fitness
    
    def get_parameters(self):
        return self.parameters

    def __str__(self):
        pnum = len(self.parameters)
        parameters_str = ",".join(["{}={}".format(*i) for i in self.parameters.items()])
        return "{} with {} parameter{}: ".format(self.__class__.__name__, pnum, "s" if pnum > 1 else "") + parameters_str + " & fitness:{}".format(self.get_fitness() if self.is_fitness_computed else "N/A")

    def __repr__(self):
        repr_str = ",".join("{key}={value}".format(key=key, value=repr(value)) for key, value in self.parameters.items())
        return "{}({})".format(self.__class__.__name__, repr_str)

    @classmethod
    def get_parameter_names(cls):
        return cls.parameter_names

class GAOptimizer(object):
    
    def __init__(self, population_type: Individual, population_count: int, search_space_gens: dict, surviving_count: float, mutation_rate: float) -> None:
        for i in search_space_gens.items():
            if not isinstance(i[1], RandomGenerator):
                raise TypeError("Search space generators must be of type RandomGenerator or a subclass!")
        self._gens = search_space_gens
        if population_count <= 1:
            raise ValueError("You need more than one individual in the population")
        if not (1 < surviving_count < population_count):
            raise ValueError("Surviving member count must be between 1 and {} (the population count)".format(population_count))
        if not (0 < mutation_rate < 1):
            raise ValueError("Mutation rate must be between 0 and 1")
        self.surviving_count = surviving_count
        self.population = []
        self.pop_type = population_type
        self.pop_count = population_count
        self.mut_rate = mutation_rate
        self.population = self._create_population(self._gens, self.pop_count)
            
    def evolve(self, pool=None) -> None:
        if pool is not None:
            pool.map(self.pop_type.get_fitness, self.population)
            self.population = [x for _,x in sorted(zip(fitness_values, self.population))]
        else:
            self.population.sort(key=lambda x: x.get_fitness())
        self.population = self.population[:self.surviving_count - 1]
        self.current_genome_generators = dict((pname, []) for pname in self.pop_type.get_parameter_names())
        for i in self.population:
            individual_genome = i.get_parameters()
            for j in self.pop_type.get_parameter_names():
                self.current_genome_generators[j].append(individual_genome[j])
        self.genome_selectors = dict((pname, RandomGenerator(pgen.return_type, self.current_genome_generators[pname], True)) for pname, pgen in self._gens.items())
        self.population.extend(self._create_population(self.genome_selectors, self.pop_count-self.surviving_count+1))
        gc.collect()
        
    def evolve_multigeneration(self, num_generations:int, verbose=0, pool=None) -> None:
        population_mask = [True] * len(self.population)
        with tqdm.tqdm_notebook(total=self.pop_count + (self.pop_count - self.surviving_count)*(num_generations - 1)) as population_iter:
            for ngen in range(num_generations):
                population_iter.clear()
                if pool is not None:
                    fitness_values = pool.map(self.pop_type.get_fitness, self.population)
                    self.population = [x for _,x in sorted(zip(fitness_values, self.population), key=lambda t:t[0])]
                    if ngen == 0:
                        population_iter.update(self.pop_count)
                    else:
                        population_iter.update(self.pop_count - self.surviving_count)
                else:
                    for idx,individual in enumerate((i for i,v in zip(self.population, population_mask) if v)):
                        individual.get_fitness()
                        population_mask[idx] = False
                        population_iter.update()
                    self.population.sort(key=lambda x: x.get_fitness())
                if verbose == 1:
                    population_iter.set_description("Generation: {}; Best Fitness: {}".format(ngen + 1, self.population[0]))
                elif verbose == 2:
                    population_iter.write("In decreasing order of optimality, the current population is\n")
                    for individual in self.population:
                        population_iter.write(str(individual))
                if (ngen < num_generations - 1):
                    for i in range(self.surviving_count, self.pop_count):
                        parent1,parent2 = tuple(np.random.choice(len(self.population), 2))
                        parent1_genome,parent2_genome = self.population[parent1].get_parameters(), self.population[parent2].get_parameters()
                        self.genome_selectors = dict((pname, RandomGenerator(pgen.return_type, (parent1_genome[pname], parent2_genome[pname]), True)) for pname, pgen in self._gens.items())
                        self.population[i:i+1] = self._create_population(self.genome_selectors, 1)
                        population_mask[i] = True
                gc.collect()
            
    def _create_population(self, pop_gens, num_gens):
        return [self.pop_type(**dict((pname, next(pop_gens[pname] if np.random.random() > self.mut_rate else self._gens[pname])) for pname in pop_gens.keys())) for i in range(num_gens)]
    
    def __getitem__(self, index):
        return self.population[index]
    
    def __len__(self):
        return len(self.population)
    
class ComplexIndividual(Individual):
    parameter_names = ['x', 'y']

    def __init__(self, x, y):
        self.is_fitness_computed = False
        self.parameters = {'x': x, 'y': y}
        self.fitness = 0

    def get_fitness(self) -> float:
        if not self.is_fitness_computed:
            self.fitness = self.parameters['x']**2 - 2*self.parameters['x'] + 1 + self.parameters['y']**2
            self.is_fitness_computed = True
        return self.fitness

    @classmethod
    def get_parameter_names(cls):
        return cls.parameter_names
