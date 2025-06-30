"""
This file defines the baselines for other selection methods,
such as fitness proportionate selection, tournament selection, etc.
"""
import numpy as np


def tournament_selection(fitnesses, tournament_size=7, num_to_select=None):
    """
    Tournament selection algorithm
    :param fitnesses (np.ndarray): fitnesses of the population. This should be a
      numpy array of shape (num_individuals, num_objectives)
    :param tournament_size (int): (optional) size of the tournament. Default is 7
    :param num_to_select (int): (optional) number of individuals to select.
      Default is the same as the population size
    :return (np.ndarray): indices of selected individuals (shape (num_to_select,))
    """
    if num_to_select is None:
        num_to_select = fitnesses.shape[0]

    selected = np.zeros(num_to_select, dtype=int)

    for itr in range(num_to_select):
        # Get a random set of individuals
        tournament = np.random.choice(
            fitnesses.shape[0], size=tournament_size, replace=False
        )
        # Get the best individual from the tournament
        selected[itr] = tournament[np.argmax(fitnesses[tournament])]

    return selected


def fitness_proportionate_selection(fitnesses, num_to_select=None):
    """
    Fitness proportionate selection algorithm
    :param fitnesses (np.ndarray): fitnesses of the population. This should be a
      numpy array of shape (num_individuals, num_objectives)
    :param num_to_select (int): (optional) number of individuals to select.
      Default is the same as the population size
    :return (np.ndarray): indices of selected individuals (shape (num_to_select,))
    """
    if num_to_select is None:
        num_to_select = fitnesses.shape[0]

    # Get the total fitness of the population
    total_fitness = np.sum(fitnesses, axis=0)
    # Get the probability of each individual being selected
    probabilities = fitnesses / total_fitness
    # Select the individuals
    selected = np.random.choice(
        fitnesses.shape[0], size=num_to_select, replace=True, p=probabilities
    )

    return selected