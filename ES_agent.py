import numpy as np
import pandas as pd

import sys
from sklearn.metrics import accuracy_score

import time
import random

import math
from tqdm.auto import tqdm

def feature_preprocessing(input):
  player = input
  player = np.array(player)
  if player[26] == 1:
    min_0 = 12
    max_0 = 24
    player[min_0:max_0] = 0
  else:
    min_0 = 0
    max_0 = 12
    player[min_0:max_0] = 0
  return player

def net_function(X, w, bias):
  return np.matmul(X, w) + bias

def sigmoid_function(x):
  y = 1 / (1 + np.exp(-x))
  return y

class MyNeuralNetwork(object):
  def __init__(self, sigma, num_population, num_games, num_feature, num_output, num_hidden_neurons):
    self.num_feature = num_feature
    self.num_output = num_output
    self.num_hidden_neurons = num_hidden_neurons
    self.sigma = sigma
    self.num_population = num_population
    self.num_games = num_games
    self.w1 = np.random.normal(size=(self.num_feature, self.num_hidden_neurons))
    self.w2 = np.random.normal(size=(self.num_hidden_neurons, self.num_output))
    self.b1 =  0.5 * np.ones([1, self.num_hidden_neurons])
    self.b2 =  0.5 * np.ones([1, self.num_output])

    #initial parent
    parent_w1 = []
    parent_w2 = []
    parent_b1 = []
    parent_b2 = []
    for i in range(num_population):
      w1 = np.random.uniform(-0.2, 0.2, size=(num_feature, num_hidden_neurons))
      w2 = np.random.uniform(-0.2, 0.2, size=(num_hidden_neurons, num_output))
      b1 = np.random.uniform(-0.2, 0.2, size=num_hidden_neurons)
      b2 = np.random.uniform(-0.2, 0.2, size=num_output)

      parent_w1.append(w1)
      parent_w2.append(w2)
      parent_b1.append(b1)
      parent_b2.append(b2)

    self.parent_w1 = np.array(parent_w1)
    self.parent_w2 = np.array(parent_w2)
    self.parent_b1 = np.array(parent_b1)
    self.parent_b2 = np.array(parent_b2)
  
  def make_kids(self):
    Nw = 3000
    tau = 1 / np.sqrt(2 * np.sqrt(Nw))
    prime_w1 = []
    prime_w2 = []
    prime_b1 = []
    prime_b2 = []
    for i in range(len(self.parent_w1)):
      #σ′i(j) = σi(j)exp( τN(0,1) )
      sigma_prime_w1 = self.sigma * np.exp(tau * np.random.uniform(0, 1, size=(self.num_feature, self.num_hidden_neurons)))
      sigma_prime_w2 = self.sigma * np.exp(tau * np.random.uniform(0, 1, size=(self.num_hidden_neurons, self.num_output)))
      sigma_prime_b1 = self.sigma * np.exp(tau * np.random.uniform(0, 1, size=self.num_hidden_neurons))
      sigma_prime_b2 = self.sigma * np.exp(tau * np.random.uniform(0, 1, size=self.num_output))
      # σ′i(j)Nj(0,1)
      w1_prime = sigma_prime_w1 * np.random.normal(0, 1, size=(self.num_feature, self.num_hidden_neurons))
      w2_prime = sigma_prime_w2 * np.random.normal(0, 1, size=(self.num_hidden_neurons, self.num_output))
      b1_prime = sigma_prime_b1 * np.random.normal(0, 1, size=self.num_hidden_neurons)
      b2_prime = sigma_prime_b2 * np.random.normal(0, 1, size=self.num_output)
      prime_w1.append(w1_prime)
      prime_w2.append(w2_prime)
      prime_b1.append(b1_prime)
      prime_b2.append(b2_prime)

    #transform to array
    prime_w1 = np.array(prime_w1)
    prime_w2 = np.array(prime_w2)
    prime_b1 = np.array(prime_b1)
    prime_b2 = np.array(prime_b2)

    #create child, w′i(j) = wi(j) + σ′i(j)Nj(0,1), j = 1, ..., Nw
    child_w1 = self.parent_w1 + prime_w1
    child_w2 = self.parent_w2 + prime_w2
    child_b1 = self.parent_b1 + prime_b1
    child_b2 = self.parent_b2 + prime_b2

    return child_w1, child_w2, child_b1, child_b2

  def evaluate(self, w1, w2, b1, b2):
    whiteValue = 0
    blackValue = 0
    for i in range(self.num_games):
      #Initial Game to start
      backgammon_env = backgammon()
      backgammon_env.genarateBoard()
      while backgammon_env.get_winner() is None:
        #get all move
        move = backgammon_env.get_moves()
        for m in move:
          if m[26] == 1:
            score_i = self.predict_t(m, w1, w2, b1, b2)
          else:
            score_i = random.random()
          backgammon_env.score_move(m, score_i)
        if backgammon_env.winner == 'WHITE':
          whiteValue += 1
        if backgammon_env.winner == 'BLACK':
          blackValue += 1
    return whiteValue / self.num_games

  def get_fitness(self, child_w1, child_w2, child_b1, child_b2):
    fitness_total = []
    fitness_parent = []
    fitness_child = []
    for j in range(self.num_population):
      fitness = self.evaluate(self.parent_w1[j], self.parent_w2[j], self.parent_b1[j], self.parent_b2[j])
      fitness_total.append(fitness)
      fitness_parent.append(fitness)

    #got fitness of child
    for j in range(self.num_population):
      fitness = self.evaluate(child_w1[j], child_w2[j], child_b1[j], child_b2[j])
      fitness_total.append(fitness)
      fitness_child.append(fitness)

    print(np.mean(fitness_parent))
    return fitness_total

  def make_selection(self, fitness_total, child_w1, child_w2, child_b1, child_b2):
    #sort the fitness
    sorted_idx = sorted(range(len(fitness_total)), reverse=True, key=lambda k: fitness_total[k])
    sorted_idx = np.array(sorted_idx)

    #choose top half as the new parent
    new_parent_w1 = []
    new_parent_w2 = []
    new_parent_b1 = []
    new_parent_b2 = []
    for i in range(self.num_population):
      if sorted_idx[i] < self.num_population:
        new_parent_w1.append(self.parent_w1[sorted_idx[i]])
        new_parent_w2.append(self.parent_w2[sorted_idx[i]])
        new_parent_b1.append(self.parent_b1[sorted_idx[i]])
        new_parent_b2.append(self.parent_b2[sorted_idx[i]])
      else:
        idx = sorted_idx[i] - self.num_population
        new_parent_w1.append(child_w1[idx])
        new_parent_w2.append(child_w2[idx])
        new_parent_b1.append(child_b1[idx])
        new_parent_b2.append(child_b2[idx])

    #transform new parent as the array
    new_parent_w1 = np.array(new_parent_w1)
    new_parent_w2 = np.array(new_parent_w2)
    new_parent_b1 = np.array(new_parent_b1)
    new_parent_b2 = np.array(new_parent_b2)

    return new_parent_w1, new_parent_w2, new_parent_b1, new_parent_b2


  def predict_t(self, inputs, w1, w2, b1, b2):
    inputs = feature_preprocessing(inputs)
    z1 = net_function(inputs, w1, b1)
    a1 = sigmoid_function(z1)
    z2 = net_function(a1, w2, b2)
    a2 = sigmoid_function(z2)
    return a2

  def update_parent(self, new_parent_w1, new_parent_w2, new_parent_b1, new_parent_b2):
    self.parent_w1 = 0.7 * self.parent_w1 + 0.3 * new_parent_w1
    self.parent_w2 = 0.7 * self.parent_w2 + 0.3 * new_parent_w2
    self.parent_b1 = 0.7 * self.parent_b1 + 0.3 * new_parent_b1
    self.parent_b2 = 0.7 * self.parent_b2 + 0.3 * new_parent_b2

  def train(self):
    child_w1, child_w2, child_b1, child_b2 = self.make_kids()
    fitness_total = self.get_fitness(child_w1, child_w2, child_b1, child_b2)
    new_parent_w1, new_parent_w2, new_parent_b1, new_parent_b2 = self.make_selection(fitness_total, child_w1, child_w2, child_b1, child_b2)
    self.update_parent(new_parent_w1, new_parent_w2, new_parent_b1, new_parent_b2)
    
  def predict(self, inputs):
    inputs = feature_preprocessing(inputs)
    z1 = net_function(inputs, self.w1, self.b1)
    a1 = sigmoid_function(z1)
    z2 = net_function(a1, self.w2, self.b2)
    a2 = sigmoid_function(z2)
    return a2
