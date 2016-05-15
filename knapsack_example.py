import numpy as np
from combinatory_GA import Combinatory_GA

class Knapsack_problem(Combinatory_GA):

    def __init__(self,max_weight_vector, weight_vector, value_vector,nGen=100, N=50,pC=0.8,pM=0.05,elitism=True,verbose=True):
        '''
        Resolves the knapsack problem. It is allowed multiple backpacks.

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

        max_weight_vector: array of positive integers
                         An array containing the maximum capacity of each backpack in the problem. The length of this array is the number of backpacks of
                         the problem.

        weight_vector: array of positive floats [n_objects,]
                     A vector of the weight of each object.

        value_vector: array of positive floats [n_objects,]
                    A vector of the value of each object. The length of this vector is the same of the vector weight_vector.
        '''

        super(Knapsack_problem,self).__init__(nGen=nGen, N=N,pC=pC,pM=pM,elitism=elitism,verbose=verbose)

        self.max_weight_vector = max_weight_vector

        if weight_vector.shape != value_vector.shape:
            raise ValueError('Unequal shape for weight and value vector...')

        self.weight_vector = weight_vector
        self.value_vector = value_vector

        self.n_vector = self.weight_vector.shape[0]

    def eval_id(self,id_):
        '''
        Evaluates the fitness of an individual. Since it is a minimization problem, when one of the back packs is no full, its total weight is inverted, but when 
        it is exceeded its capacity, her weight is mutiplied by 10 and summed as a penality. Then all the weights are summed to obtain the total weight.

        Parameters: id_ array[n_weigths,]
                    An array of integers represanting a possible solution
        '''

        if id_.shape != self.weight_vector.shape:
            raise ValueError('Unequal shape for weight and individual vector...')

        s_ = 0
        total_value = np.sum(self.value_vector)
        for i,max_weight in enumerate(self.max_weight_vector):
            sum_weight = np.sum(self.weight_vector[id_==(2*i+1)])
            sum_value = np.sum(self.value_vector[id_==(2*i+1)])
            dist = 0 if max_weight > sum_weight else sum_weight-max_weight
            diff = min(max_weight,abs(sum_weight-max_weight))
            s_ += 1/sum_value + sum_value*(dist**2/diff)

        ret = s_

        return ret

    def generate_initial_population(self):
        '''
        Generates an initial population. An individual is composed only by integer values that area between 0 and the number of
        backpack minus 1. Since, in this representation, odd numbers represent an object inserted in a backpack. In that way, a '1'
        in the 'n' position of an individual represent that the object at position 'n' was inserted in the first backpack, as a '5' in the 'x' position 
        represents that the object at position 'x' was inserted in the third backpack, and so on.
        '''

        size = len(self.weight_vector)
        population = []
        n_sacks = 2*len(self.max_weight_vector)
        for i in range(self.N):
            population.append(np.random.randint(0,n_sacks,size))

        self.pop = np.array(population)
