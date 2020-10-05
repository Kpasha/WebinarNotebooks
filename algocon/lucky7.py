import numpy as np 
import pandas as pd 
class Dice:

    def __init__(self):
        self.numbers = [1, 2, 3, 4, 5, 6]

    def roll(self):
        return np.random.choice(self.numbers)

dice1 = Dice()
dice2 = Dice()

# Monte Carlo Experiment

num_of_iterations = 1000000

list_of_outcomes = []

for _ in range(num_of_iterations):
    list_of_outcomes.append(dice1.roll() + dice2.roll())

from collections import Counter

x = pd.Series(Counter(list_of_outcomes)).sort_index()

print((x/num_of_iterations)*100)
