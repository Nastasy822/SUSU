import random 
import numpy as np
import os
import matplotlib.pyplot as plt
from IPython.display import clear_output

def sigmoid(z): # sigmoid function
  return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):# Derivative of the sigmoid function
  return sigmoid(z)*(1-sigmoid(z))

class Network(object):
  # sizes - list of sizes of neural network layers
  # num_layers - the number of layers of the neural network
  def __init__(self, sizes): 
    self.num_layers = len(sizes) 
    self.sizes = sizes 
    self.biases = [np.random.randn(y, 1) for y in sizes[1:]] 
    self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1],sizes[1:])]
    # initialization of connection weights with random numbers (y is the number of output layers, x is the number of input layers)


  def DataLoader(self,data,bs):
    n = len(data)
    return [data[k:k+bs] for k in range(0, n, bs)] 

  def forward(self, x):
    activation = x # layer output signals
    self.activations = [x] # list of output signals for all layers
    self.zs = [] # list of activation potentials for all layers

    for b, w in zip(self.biases, self.weights):
      z = np.dot(w, activation)+b # calculate the activation potentials of the current layer
      activation = sigmoid(z) # calculate the output signals of the current layer by applying a sigmoidal activation function to the activation potentials layer

      self.zs.append(z)
      self.activations.append(activation) 


  def backward(self,y):
    self.nabla_b = [np.zeros(b.shape) for b in self.biases] # list of dC/db gradients for each layer
    self.nabla_w = [np.zeros(w.shape) for w in self.weights] # list of dC/dw gradients for each layer
    
    delta = self.cost_derivative(self.activations[-1], y) *sigmoid_prime(self.zs[-1]) # calculate the measure of the influence of output layer neurons L on the error value
    self.nabla_b[-1] = delta # gradient dC/db for layer L
    self.nabla_w[-1] = np.dot(delta, self.activations[-2].transpose()) #gradient dC/dw for layer L

    for l in range(2, self.num_layers):
      delta = np.dot(self.weights[-l+1].transpose(), delta) * sigmoid_prime(self.zs[-l]) #calculate the measure of the influence of neurons of the l-th layer on the magnitude of the error

      self.nabla_b[-l] = delta # gradient dC/db for layer l
      self.nabla_w[-l] = np.dot(delta, self.activations[-l-1].transpose())# gradient dC/dw for layer l
 

  # Calculation of partial derivatives of the cost function from the output signals of the last layer
  def cost_derivative(self, output_activations, y):
    return (output_activations-y)

  def backprop(self, x , y ):
    self.forward(x) 
    self.backward(y)
    return (self.nabla_b, self.nabla_w)

  
  # Gradient Descent Step
  def update_mini_batch(self, batch, lr):
    
    nabla_b = [np.zeros(b.shape) for b in self.biases] # list of dC/db gradients for each layer
    nabla_w = [np.zeros(w.shape) for w in self.weights] # list of dC/dw gradients for each layer

    for x, y in batch:
      delta_nabla_b, delta_nabla_w = self.backprop(x, y)
      nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)] # sum the gradients dC/db for different cases of the current subsample
      nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)] #sum the gradients dC/dw for different cases of the current subsample
      self.weights = [w-(lr/len(batch))*nw for w, nw in zip(self.weights, nabla_w)] #update all weights w of the neural network
      self.biases = [b-(lr/len(batch))*nb for b, nb in zip(self.biases, nabla_b)] # update all offsets b of the neural network


  # Stochastic Gradient Descent
  def SGD(self, train_data, epochs , bs , lr , test_data):
    test_data = list(test_data) 
    n_test = len(test_data)
    train_data = list(train_data)

    self.log_accuracy=[]
    for epoch in range(epochs): 
      random.shuffle(train_data) 
      train_loader = self.DataLoader(train_data,bs) 
      for batch in train_loader: 
        self.update_mini_batch(batch, lr)

      self.log_accuracy.append((self.evaluate(test_data) / n_test) * 100)
      self.plot_summary()


  def plot_summary(self):
    clear_output(True)
    plt.plot([i for i in range(0,len(self.log_accuracy))], self.log_accuracy,'-go')
    plt.xlabel('epoch'),plt.ylabel('accuracy')
    plt.show()

  def evaluate(self, test_data): # Testing of learning progress
    test_results = [(np.argmax(self.feedforward(x)), y) for (x, y) in test_data]
    return sum(int(x == y) for (x, y) in test_results)

  def feedforward(self, a):
    for b, w in zip(self.biases, self.weights):
      a = sigmoid(np.dot(w, a)+b)
    return a

  def get_predict(self,image):
    image=np.reshape(image, (784, 1))
    return np.argmax(self.feedforward(image))

  def save_weights(self,save_path_dir=''):
    np.save(os.path.join(save_path_dir,'weights'),[self.weights,self.biases])

  def load_weights(self, load_path_dir=''):
    weights=np.load(load_path_dir,allow_pickle=True)
    self.weights, self.biases=weights[0],weights[1]
