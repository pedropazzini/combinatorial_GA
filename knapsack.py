from knapsack_example import Knapsack_problem
import sys
import numpy as np


def main ():

    n_objects = 100
    weights = np.random.normal(10,5,n_objects)
    values = np.random.uniform(10,100,n_objects)

    max_weight_backpack = np.array([30,40,50])

    kp = Knapsack_problem(max_weight_backpack,weights,values,elitism=False)
    kp.optimize()

    kp.print_solution()

if __name__ == "__main__":
    sys.exit(main())
