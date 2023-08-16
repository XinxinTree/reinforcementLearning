
import numpy as np
import pandas as pd
import pickle
import sys
from sklearn.metrics import accuracy_score

import time
import random

import math

#function of load data
def read(file_name):
  data = pd.read_csv(file_name)
  return data
  
#split data to X and y
def data_pre(data, num_features):
  nn = np.arange(num_features)
  col = data.columns[nn]
  y = data.drop(col, axis =1)
  X = data[col]
  return X, y

#split data to X_train, X_test, y_train, y_test
def split_data(X, y, train_size):
  df_shuffleX = X.sample(frac=1)
  df_shuffley = y.sample(frac=1)
  train_s = int(train_size * len(X))

  X_train = df_shuffleX[:train_s]
  X_test = df_shuffleX[train_s:]
  y_train = df_shuffley[:train_s]
  y_test = df_shuffley[train_s:]

  X_train = X_train.reset_index()
  X_test = X_test.reset_index()
  y_train = y_train.reset_index()
  y_test = y_test.reset_index()

  X_train = X_train.drop('index', axis = 1)
  X_test = X_test.drop('index', axis = 1)
  y_train = y_train.drop('index', axis = 1)
  y_test = y_test.drop('index', axis = 1)
  return X_train, X_test, y_train, y_test

#feature preprocessing
def feature_preprocessing(input):
  player = input
  if player[14] == 1:
    min_0 = 6
    max_0 = 12
    player[min_0:max_0] = 0
  else:
    min_0 = 0
    max_0 = 6
    player[min_0:max_0] = 0
  return player

#normalization function
def normalization(X_train, X_test):
  mean = np.mean(X_train, axis = 0)
  variance = np.var(X_train, axis = 0)

  z_score_train = (X_train - mean) / (variance++0.00001)
  z_score_test = (X_test - mean) / (variance++0.00001)
  return z_score_train, z_score_test

def net_function(X, w, bias):
  return np.matmul(X, w) + bias
  
def sigmoid_function(x):
  y = 1 / (1 + np.exp(-x))
  return y

def derivative_function(x):
  y = sigmoid_function(x) * (1 - sigmoid_function(x))
  return y

def error_fun(prediction, actual):
  err = (1/2) * (prediction - actual) ** 2
  return err

class MyNeuralNetwork(object):
  def __init__(self, num_feature, num_output, num_hidden_neurons):
    self.num_feature = num_feature
    self.num_output = num_output
    self.num_hidden_neurons = num_hidden_neurons
    self.w1 = np.random.normal(size=(self.num_feature, self.num_hidden_neurons))
    self.w2 = np.random.normal(size=(self.num_hidden_neurons, self.num_output))
    self.b1 =  0.5 * np.ones([1, self.num_hidden_neurons])
    self.b2 =  0.5 * np.ones([1, self.num_output])
    self.zt_out = 0
    self.zt_1_out = 0
    self.zt_hidden = 0
    self.zt_1_hidden = 0

  #forward do the net_function and sigmoid_function
  def forward(self, inputs):
    z1 = net_function(inputs, self.w1, self.b1)
    a1 = sigmoid_function(z1)
    z2 = net_function(a1, self.w2, self.b2)
    a2 = sigmoid_function(z2)
    return a1, a2, z1, z2

  #backward
  def backward(self, input, gamma, v_t, v_t1, a1, a2, z1, z2):

    #temporal difference error
    e = v_t1 - v_t

    #derivative sigmoid function
    de = derivative_function(z2)
    de_hidden = derivative_function(z1)

    #eligibility traces
    #out layer
    z_trace_out = np.matmul(a1.T, de)
    self.zt_1_out = gamma*self.zt_out + z_trace_out
    self.zt_out = self.zt_1_out

    #hidden layer
    me = self.w2.T * de_hidden
    z_trace_hidden = np.matmul(input.reshape(self.num_feature,1), me)
    self.zt_1_hidden = gamma*self.zt_hidden + z_trace_hidden
    self.zt_hidden = self.zt_1_hidden

    #weight changing
    wei_c2 = e * self.zt_1_out
    bias_c2 = e * de
    wei_c1 = e * self.zt_1_hidden
    bias_c1 = e * me

    return bias_c2, wei_c2, bias_c1, wei_c1

  #update
  def update_param(self, inputs, learning_rate, bias_c2,  weight_c2, bias_c1, weight_c1):
    self.w1 += learning_rate * weight_c1
    self.w2 += learning_rate * weight_c2
    self.b1 += learning_rate * bias_c1
    self.b2 += learning_rate * bias_c2
  
  def train(self, inputs, r, v_t, v_t1, learning_rate, gamma):
    inputs = feature_preprocessing(inputs)
    a1, a2, z1, z2 = self.forward(inputs)
    bias_c2, wei_c2, bias_c1, wei_c1 = self.backward(inputs, gamma, v_t, v_t1, a1, a2, z1, z2)
    self.update_param(inputs, learning_rate, bias_c2, wei_c2, bias_c1, wei_c1)

  def predict(self, inputs):
    z1 = net_function(inputs, self.w1, self.b1)
    a1 = sigmoid_function(z1)
    z2 = net_function(a1, self.w2, self.b2)
    a2 = sigmoid_function(z2)
    return a2

class two_d_set(object):
    def __init__(self):
        self.data = []

    def add_array(self, new_arr):
        temp = new_arr[15]
        new_arr[15] = new_arr[14]
        new_arr[14] = temp
        if new_arr not in self.data:
            self.data.append(new_arr)

    def remove_array(self, target_arr):
        self.data.remove(target_arr)

    def get_array(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

    def __str__(self):
        return str(self.data)
    
    def __iter__(self):
        return iter(self.data)
    
    def __getitem__(self, index):
        return self.data[index]

NUM_POINTS = 6

class backgammon:

  def gen_moves(self):
    self.moves = two_d_set()
    if self.dice1 != self.dice2:
      for i in range(0,6):
        first_board = self.move_check(i, self.dice1, self.board)
        if first_board == None:
          continue
        jx = 0
        for j in range(0,6):
          final_board = self.move_check(j, self.dice2, first_board)
          if final_board == None:
            jx += 1
            if jx == 6:
              self.moves.add_array(first_board)
            continue
          else:
            self.moves.add_array(final_board)

      for i in range(0,6):
        first_board = self.move_check(i, self.dice2, self.board)
        if first_board == None:
          continue
        jx = 0
        for j in range(0,6):
          final_board = self.move_check(j, self.dice1, first_board)
          if final_board == None:
            jx += 1
            if jx == 6:
              self.moves.add_array(first_board)
            continue
          else:
            self.moves.add_array(final_board)

    else:
      for i in range(0,6):
        first_board = self.move_check(i, self.dice1, self.board)
        if first_board == None:
          continue
        jx = 0
        for j in range(0,6):
          second_board = self.move_check(j, self.dice1, first_board)
          if second_board == None:
            jx += 1
            if jx == 6:
              self.moves.add_array(first_board)
            continue
          kx = 0
          for k in range(0,6):
            third_board = self.move_check(k, self.dice1, second_board)
            if third_board == None:
              kx += 1
              if kx == 6:
                self.moves.add_array(second_board)
              continue
            lx = 0
            for l in range(0,6):
              final_board = self.move_check(l, self.dice1, third_board)
              if final_board == None:
                lx += 1
                if lx == 6:
                  self.moves.add_array(third_board)
                continue
              else:
                self.moves.add_array(final_board)
    if len(self.moves) == 0: #PASS turn
      self.moves.add_array(self.board)
    return self.moves

  def __init__(self):
      self.moves_info = []
      self.genarateBoard()
      self.dice1 = self.roll()
      self.dice2 = self.roll()
      self.winner = None
      self.moves = self.gen_moves()

  def random_board(self):
    board = [0] * NUM_POINTS
    checkers = 15
    length = len(board)
    arr = [0, 1, 2, 3, 4, 5]

    # Shuffle the array
    random.shuffle(arr)

    for idx, point in enumerate(arr):
      if idx == 0:
        board[point] = self.rng(0,8)
      elif idx == length-1:
        board[point] = abs(sum(board)-15)
        continue
      else:
        board[point] = self.rng(0,checkers)
      checkers = checkers - board[point]
    return board

  def flip_coin(self):
    black = self.rng(0,1)
    if black == 0:
      return [0,1]
    if black == 1:
      return [1,0]

  def genarateBoard(self):
    off = [0,0]
    turn = self.flip_coin()
    white = self.random_board()
    black = self.random_board()
    self.board = white + black + off + turn

  def check_winner(self):
    self.winner = None
    if self.board[12] == 15:
      self.winner = "WHITE"
    elif self.board[13] == 15:
      self.winner = "BLACK"

  def get_winner(self):
    return self.winner

  def get_moves(self):
    return self.moves

  def get_board(self):
    return self.board

  def move_check(self,point, roll, board):
    player = board.copy()
    if player[14] == 1:
      min = 0
      max = 6
      new_move = player[min:max]
      off_idx = 12
    else:
      min = 6
      max = 12
      new_move = player[min:max]
      off_idx = 13

    highest_point = next((i for i, x in enumerate(new_move) if x != 0), None)
    if highest_point == None or new_move[point] == 0:
      return None
    if point < highest_point:
      return None
    if (point == highest_point) and (point+roll >= 6):
      new_move[point] -= 1
      player[off_idx] += 1
    elif (point+roll == 6):
      new_move[point] -= 1
      player[off_idx] += 1
    elif (point+roll < 6):
      new_move[point] -= 1
      new_move[point+roll] += 1
    else:
      return None
    
    player[min:max] = new_move

    return player

  def make_move(self, new_board):
    self.board = new_board
    self.moves_info = []
    self.dice1 = self.roll()
    self.dice2 = self.roll()
    self.check_winner()
    self.gen_moves()
  
  def roll(self):
    return self.rng(1,6)

  def rng(self, min, max):
    return random.randint(min,max)

  def get_next_move(moves):
    for move in moves:
        yield move

  def score_move(self, move_array, scalar_value):
    move_info = {
      "move": move_array,
      "score": scalar_value
    }
    self.moves_info.append(move_info)
    if len(self.moves_info) == len(self.moves): #confusing about that
      if self.board[14] == 1:
        self.make_move(self.get_best_move())
      else:
        self.make_move(self.get_best_move(highest_score=False))


  def get_best_move(self, highest_score=True):
    if len(self.moves_info) == 0:
      return None

    best_move = None
    if highest_score:
      best_move = max(self.moves_info, key=lambda x: x["score"])
      # print("Max",best_move)
    else:
      best_move = min(self.moves_info, key=lambda x: x["score"])
      # print("Min", best_move)

    return best_move["move"]

from tqdm.auto import tqdm

gamma = 0.1
learning_rate = 10
generations = 1000
mlp_agent = MyNeuralNetwork(16, 1, 80)
score_white = []
score_black = []
white_value = 0
black_value = 0

for _ in tqdm(range(generations)):
    # Initialize game parameters
    backgammon_env = backgammon()
    backgammon_env.genarateBoard()
    v_t = 0.5

    while backgammon_env.get_winner() is None:
        feature_vector = backgammon_env.get_board()

        # Get all moves
        moves = backgammon_env.get_moves()
        num_moves = len(moves)

        move_array = np.zeros((num_moves, 16))
        scalar_value = np.zeros(num_moves)

        for i, move in enumerate(moves):
            move_array[i] = move
            score_i = mlp_agent.predict(move)
            scalar_value[i] = score_i

        # Choose the best move based on the current player's perspective
        if backgammon_env.get_board()[14] == 1:
            index_white = np.argmax(scalar_value)
            vt_plus = scalar_value[index_white]
            score_white.append(np.max(scalar_value))
            best_move = moves[index_white]
        else:
            index_black = np.argmin(scalar_value)
            vt_plus = scalar_value[index_black]
            score_black.append(np.min(scalar_value))
            best_move = moves[index_black]

        # Make the best move
        backgammon_env.make_move(best_move)

        if backgammon_env.get_winner() is not None:
            if backgammon_env.winner == 'WHITE':
                white_value += 1
                r = 1
            else:
                black_value += 1
        else:
            r = 0

        feature_vector_plus = backgammon_env.get_board()
        feature_vector = np.array(feature_vector)
        mlp_agent.train(feature_vector, r, v_t, vt_plus, learning_rate, gamma)

        feature_vector = feature_vector_plus
        v_t = vt_plus

filename = 'mlp_agent.pkl'
with open(filename, 'wb') as file:
    pickle.dump(mlp_agent, file)
    
print(f"white wining rate {white_value / generations} black wining rate {black_value / generations} ")

#Test compare with random
whiteValue = 0
blackValue = 0
for i in tqdm(range(0, generations)):
    # Initialize game parameters
    backgammon_env = backgammon()
    backgammon_env.genarateBoard()
    while backgammon_env.get_winner() is None:
    
      #get all move
      move = backgammon_env.get_moves()
      for m in move:
        if m[14] == 1:
          score_i = mlp_agent.predict(m)
        else:
          score_i = random.random()
        backgammon_env.score_move(m, score_i)

      if backgammon_env.winner == 'WHITE':
        whiteValue += 1
      if backgammon_env.winner == 'BLACK':
        blackValue += 1

print(f"After trainning, white wining rate {whiteValue / generations} black wining rate {blackValue / generations} ")
