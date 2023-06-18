#!/usr/bin/env python

"""
Entrega #4 - Dynamic Programming - Value iteration Gambler's Problem

@author: Leonardo Pezenatto da Silva
@email: leonardo.pezenatto@posgrad.ufsc.br
@date: Jun 20, 2023
"""

import numpy as np
import matplotlib.pyplot as plt

class Gambler:
    def __init__(self, prob, iterations, theta=0.00000001):
        self.value = np.zeros(101)
        self.reward = np.zeros(101)
        self.reward[100] = 1
        self.values_recorded = []
        self.pi = []
        self.prob = prob
        self.iterations = iterations
        self.theta = theta
        

    def value_iteration(self):
        delta = 0
        p = np.zeros(101)
        while delta < self.theta:
            for capital in range(1,100):
                previous_value = self.value[capital]
                for bet in range(1, min(capital, 100 - capital)+1): # Bet value with minimum of 1 and maximum of 100 - capital
                    p[bet] = self.prob*(self.reward[capital + bet] + self.value[capital + bet]) + (1-self.prob)*(self.reward[capital - bet] + self.value[capital - bet])
                self.value[capital] = max(p)
                delta = max(delta, abs(previous_value - self.value[capital]))
        self.values_recorded.append(self.value.copy())
                
    def policy(self):
        self.pi = []
        p = np.zeros(101)
        for capital in range(1,100):
            for bet in range(1, min(capital, 100 - capital)+1): # Bet value with minimum of 1 and maximum of 100 - capital
                p[bet] = self.prob*(self.reward[capital + bet] + self.value[capital + bet]) + (1-self.prob)*(self.reward[capital - bet] + self.value[capital - bet])
            self.pi.append(np.argmax(p))

    def compute(self):
        for i in range(self.iterations):
            self.value_iteration()
            self.policy()
        self.plot_results()

    def plot_results(self):
        plt.subplot(2, 1, 1)
        for data in self.values_recorded:
            plt.plot(data[:99])
        labels = ['Iteration {}'.format(i+1) for i in range(len(self.values_recorded))]
        plt.legend(labels)
        plt.xlabel('Capital')
        plt.ylabel('Value Estimates')
        plt.subplot(2, 1, 2)
        plt.bar(range(99), self.pi, align='center', alpha=0.5)
        plt.xlabel('Capital')
        plt.ylabel('Final Policy')
        plt.show()


if __name__ == "__main__":
    g = Gambler(0.4, 5) 
    g.compute()