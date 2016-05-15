import numpy as np
import copy
import matplotlib.pyplot as plt

class Combinatory_GA(object):

    def __init__(self,nGen=100, N=50,pC=0.8,pM=0.05,elitism=True,by_product=False,verbose=True):
        '''
        Parameters:
        -------------------

        nGen: integer, default 100
            Number of generations of the GA

        N: integer, default 50
         Number of individuals in the population of the GA

        pC: float, default 0.8
          Probability of crossover

        pM: float, default 0.05
          Probability of mutation

        elitism: boolean, default True
               If the elitism should be applied, i.e., the best individual should remain in the next generations.

        verbose: boolen, default True
               If messages of progrees should be printed.

        '''

        self.MAX_FITNESS = 1e100
        self.best_fitness = self.MAX_FITNESS
        self.stats_best = []
        self.stats_avg = []

        self.best_id = None

        self.N = N
        self.pC = pC
        self.pM = pM
        self.nGen= nGen
        self.pop = None
        self.pop_fitness = np.zeros(self.N)
        self.current_gen = 0

        self.equal_gen = 0

        self.to_evaluate = np.array([True for _ in range(self.N)])
        self.n_evals = 0

        self.elitism = elitism
        self.verbose = verbose

        self.hall_of_fame = []

    def restart(self):
        '''
        Restarts the population, i.e., generates a new population and stores the best individual and best fitness so far. The 
        method is indicated when is detected that the algorithm has a premature convergence.
        '''

        self.hall_of_fame.append((self.best_fitness,self.best_id))

        self.best_fitness = None
        self.generate_initial_population()

        self.to_evaluate = np.array([True for _ in range(self.N)])
        self.evaluate_population()
        self.update_stats()

        if self.verbose:
            print('Restarted....')

    def generate_initial_population(self):
        '''
        Generates an initial population of individuals.
        '''
        print('TODO')
        raise NotImplementedError


    def evaluate_population(self):
        '''
        Evaluates the fitness of each individual in the population. Note that remaining individuals from the last generation are
        not evaluated since it is not necessary.
        '''
        self.n_evals += sum(self.to_evaluate)
        for i,id_ in enumerate(self.pop):
            if self.to_evaluate[i]:
                self.pop_fitness[i] = self.eval_id(id_)
        self.to_evaluate = np.array([False for _ in range(self.N)])

    def update_stats(self):
        '''
        Update the generation stats such as average solution and best solution.
        '''

        arg_min_ = np.argmin(self.pop_fitness)
        self.best_fitness = self.pop_fitness[arg_min_]
        self.best_id = self.pop[arg_min_]

        self.stats_best.append(self.best_fitness)
        self.stats_avg.append(np.mean(self.pop_fitness))

    def select(self):
        '''
        Selection operator. The choice was a slection by tournament, where two random individuals are choosen, with reposition, and 
        the one that has the better fitness is copied to the new generation
        '''

        # select by tournament
        new_pop = []
        new_fitness = []

        while len(new_pop) != self.N:

            id_1 = np.random.randint(self.N)
            id_2 = id_1
            while id_1 == id_2:
                id_2 = np.random.randint(self.N)

            new_pop.append(self.pop[id_1] if self.pop_fitness[id_1] <= self.pop_fitness[id_2] else self.pop[id_2])
            new_fitness.append(self.pop_fitness[id_1] if self.pop_fitness[id_1] <= self.pop_fitness[id_2] else self.pop_fitness[id_2])

        self.pop = copy.copy(np.array(new_pop))
        self.pop_fitness = copy.copy(np.array(new_fitness))


    def cross(self):
        '''
        Crossover operator by random crossover point
        '''
        new_pop = []
        new_fitness = []
        for i in range(self.N):
            if np.random.rand() < self.pC:
                id_1 = i
                id_2 = id_1
                while id_2 == id_1:
                    id_2 = np.random.randint(self.N)

                pc = np.random.randint(1,self.n_vector-1)
                new_id = np.hstack((self.pop[id_1][:pc],self.pop[id_2][pc:]))
                new_pop.append(new_id)
                new_fitness.append(self.MAX_FITNESS)
                self.to_evaluate[i] = True
            else:
                new_pop.append(self.pop[i])
                new_fitness.append(self.pop_fitness[i])
        self.pop = copy.copy(np.array(new_pop))
        self.pop_fitness = copy.copy(np.array(new_fitness))

    def mutate(self):
        '''
        Mutate operator. When it is applied, two elements of an individual are swaped.
        '''

        i = 0
        for id_ in self.pop:
            if np.random.rand() < self.pM:
                pos_1 = np.random.randint(self.n_vector)
                swap_1 = id_[pos_1]
                pos_2 = pos_1
                swap_2 = swap_1
                while pos_2 == pos_1 or swap_1 == swap_2:
                    pos_2 = np.random.randint(self.n_vector)
                    swap_2 = id_[pos_2]
                    if len(np.unique(id_)) <= 1:
                        break

                id_[pos_1] = swap_2
                id_[pos_2] = swap_1
                self.pop[i] = id_
                self.to_evaluate[i] = True
            i += 1

    def optimize(self,plot_stats=False):
        '''
        Runs the optimization of the fitness function.

        Parameters: plot_stats, boolean (default=False)
                    Plots a graphic of the evalution of the best an average fitness of the population

        Returns: tuple
                 Tuple containing the fitness of the best individual and the best individual.
        '''

        self.generate_initial_population()
        self.evaluate_population()
        self.update_stats()
        for gen in range(self.nGen):

            self.select()
            self.cross()
            self.mutate()

            self.evaluate_population()

            self.elitize()

            self.update_stats()

            if self.verbose:
                print('Gen[' + str(gen) + '/' + str(self.nGen)  + ']')

            if len(np.unique(self.pop_fitness)) <= 0.05*self.N:
                self.equal_gen += 1
                if self.equal_gen > 20:
                    self.restart()
                    self.equal_gen = 0
            else:
                self.equal_gen = 0

        if plot_stats:
            self.plot_stats()

        return self.get_solution()

    def plot_stats(self,plot_best=True,plot_avg=True):
        '''
        Plot the average and best fitness of the population in each generation.

        Parameters:
        ------------------

        plot_best: boolean, default True
                 If the best individual should be plotted.

        plot_avg: boolean, default True
                If the average individual should be plotted.
        '''

        plt.figure(1)
        if plot_avg:
            plt.plot(self.stats_avg[1:],'b')
        if plot_best:
            plt.plot(self.stats_best[1:],'r')

    def get_solution(self):
        '''
        After the optimization returns the solution of the optimization problem.

        Returns: tuple
                 Containing the best fitness and best individual found.
        '''

        best_fitness = self.best_fitness
        best_id = self.best_id

        for fit,id_ in self.hall_of_fame:
            if fit < best_fitness:
                best_fitness = fit
                best_id = id_

        return (best_fitness,best_id)

    def print_solution(self):
        '''
        Prints the solution in a human readable manner.
        '''

        best_fitness,best_id = self.get_solution()

        print('Best solution: ' + str(best_id))

        print ('---------------------------------------')

        for i,backpack in enumerate(self.max_weight_vector):
            print('Backpack ' + str(i) + ' ---> max weight: ' + str(backpack))
            objects = np.where(best_id==2*i+1)[0]
            for object_ in objects:
                print('         Object ' + str(object_) + ': {weight:' + str(self.weight_vector[object_]) + ',value:' + str(self.value_vector[object_]) + '}')

            print('Total weight: ' + str(np.sum(self.weight_vector[objects])))
            print('Total value: ' + str(np.sum(self.value_vector[objects])))

            print('---------------------------------------------')
                


    def elitize(self):
        '''
        Makes elitism to ensure that the best individual remains in the next generation.
        '''
        if self.elitism:

            for id_ in self.pop:
                if (id_ == self.best_id).all():
                    return

            if self.verbose:
                print('Elitized...')
            pos_max = np.argmax(self.pop_fitness)
            self.pop[pos_max] = self.best_id
            self.pop_fitness[pos_max] = self.best_fitness


    def eval_id(self,id_):
        '''
        Method that should be overrided to evaluate the fitness of an individual.

        Returns: float
                 Value representing the fitness of an individual where smaller values are considered better fitness, 
                 since the optimization procedure is a minimization problem.
        '''
        print('TODO')
        raise NotImplementedError
