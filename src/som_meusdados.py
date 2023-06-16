#https://towardsdatascience.com/understanding-self-organising-map-neural-network-with-python-code-7a77f501e985
import numpy as np
from numpy.ma.core import ceil
from scipy.spatial import distance #distance calculation
from sklearn.preprocessing import MinMaxScaler #normalisation
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score #scoring
from sklearn.datasets import load_iris
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from matplotlib import animation, colors
import ut

################# Dados - Ensaio de Baumann #################
data3 = ut.im_data(3)

c1, c2, c3 = data3[0:82, :], data3[82:164, :], data3[175:257, :]
data_file = np.concatenate((c1, c2, c3), axis=0)

#data_x = data_file[:, :24]
data_x = data_file[:, [0, 1, 2, 3, 4, 7, 8, 21]]

data_y = data_file[:, 24]
#data_y = data_file[:, 24].reshape(data_file.shape[0], 1)


################# Íris #################

iris = load_iris()
data_x = iris.data
data_y = iris.target


# train and test split
train_x, test_x, train_y, test_y = train_test_split(data_x, data_y, test_size=0.2, random_state=42)
print(train_x.shape, train_y.shape, test_x.shape, test_y.shape) # check the shapes
#train_x = test_x
#train_y = test_y


################################ Teste

#train_x = np.array([
#  [2, 0.5, 7],
#  [4, 0.7, 1],
#  [8, 0.2, 4],
#  [2, 0.8, 2]
#])

################################ Fim Teste

# Helper functions
# Data Normalisation
def minmax_scaler(data):
  scaler = MinMaxScaler()
  scaled = scaler.fit_transform(data)
  return scaled

# Euclidean distance
def e_distance(x,y):
  return distance.euclidean(x, y)

# Manhattan distance
def m_distance(x,y):
  return distance.cityblock(x,y)

# Best Matching Unit search
def winning_neuron(data, t, som, num_rows, num_cols):

  winner = [0, 0]
  shortest_distance = np.sqrt(data.shape[1]) # initialise with max distance
  #print('shortest_distance: ', shortest_distance)

  input_data = data[t]
  for row in range(num_rows):
    for col in range(num_cols):
      # Calcula a disntância euclidiana entre a amostra (escolhida aleatoreamente) e todos os neurônios.
      # O neurônio vencedor (menor distância) será retornado
      distance = e_distance(som[row][col], data[t])

      if distance < shortest_distance:
        shortest_distance = distance
        winner = [row, col]

  return winner

# Learning rate and neighbourhood range calculation
def decay(step, max_steps,max_learning_rate,max_m_dsitance):
  coefficient = 1.0 - (np.float64(step)/max_steps)
  learning_rate = coefficient*max_learning_rate
  neighbourhood_range = ceil(coefficient * max_m_dsitance)
  return learning_rate, neighbourhood_range



# hyperparameters
num_rows = 3
num_cols = 3
max_m_dsitance = 4
max_learning_rate = 0.5
max_steps = int(7.5*10e3)
max_steps = 200
# num_nurons = 5*np.sqrt(train_x.shape[0])
# grid_size = ceil(np.sqrt(num_nurons))
# print(grid_size)

#mian function

train_x_norm = minmax_scaler(train_x) # normalisation

# initialising self-organising map
num_dims = train_x_norm.shape[1] # numnber of dimensions in the input data
np.random.seed(40)

## Grid de números aleatórios de 0.0 a 1.0.
## Suas dimentçõs são, número de neurônios na linhas x números de neurônios colunas x quantidade de atributos
## dos dados de entrada (Também conhecida com matriz de peso)
som = np.random.random_sample(size=(num_rows, num_cols, num_dims)) # map construction

errors = []  # Lista para armazenar os valores de erro

# start training iterations
for step in range(max_steps):
  if (step+1) % 1000 == 0:
    print("Iteration: ", step+1) # print out the current iteration for every 1k
  learning_rate, neighbourhood_range = decay(step, max_steps, max_learning_rate, max_m_dsitance)

  # Gera um número aleatório de 0 a quantidade de amostras (120)
  t = np.random.randint(0, high=train_x_norm.shape[0]) # random index of traing data

  # recebe como parâmetro os dados de treinamento, o número aleatório t, a rede som,
  # o número de linhas e colunas da grid

  # recebe as coordenadas do neurônio vencedor (para determinada amostra t)
  winner = winning_neuron(train_x_norm, t, som, num_rows, num_cols)

  for row in range(num_rows):
    for col in range(num_cols):
      # Aqui é feito o cálculo de vizinhaça; por exemplo, para row, col = 0, 0 e winter (neurônio vencedor) = 1, 0,
      # É calculada a distância de Manhatan entre essas duas posições. O limite dessa distância é neighbourhood_range,
      # valor que vai sendo reduzido com o passar das iterações.

      if m_distance([row, col], winner) <= neighbourhood_range:
        som[row][col] += learning_rate*(train_x_norm[t]-som[row][col]) #update neighbour's weight

  error = e_distance(train_x_norm[t], som[winner[0]][winner[1]])
  errors.append(error)

print("SOM training completed")

# collecting labels

label_data = train_y
map = np.empty(shape=(num_rows, num_cols), dtype=object)

for row in range(num_rows):
  for col in range(num_cols):
    map[row][col] = [] # empty list to store the label


for t in range(train_x_norm.shape[0]):
  if (t+1) % 1000 == 0:
    print("sample data: ", t+1)
  winner = winning_neuron(train_x_norm, t, som, num_rows, num_cols)
  map[winner[0]][winner[1]].append(label_data[t]) # label of winning neuron



# construct label map
label_map = np.zeros(shape=(num_rows, num_cols), dtype=np.int64)

for row in range(num_rows):
  for col in range(num_cols):
    label_list = map[row][col]
    if len(label_list) == 0:
      #pass
      label = 3
    else:
      label = max(label_list, key=label_list.count)

    label_map[row][col] = label
print(label_map)
title = ('Iteration ' + str(max_steps))
cmap = colors.ListedColormap(['tab:green', 'tab:blue', 'tab:red', 'tab:purple'])
plt.imshow(label_map, cmap=cmap)
plt.colorbar()
plt.title(title)
plt.show()



# test data

# using the trained som, search the winning node of corresponding to the test data
# get the label of the winning node

data = minmax_scaler(test_x) # normalisation

winner_labels = []

for t in range(data.shape[0]):
 winner = winning_neuron(data, t, som, num_rows, num_cols)
 row = winner[0]
 col = winner[1]
 predicted = label_map[row][col]
 winner_labels.append(predicted)

print("Accuracy: ", accuracy_score(test_y, np.array(winner_labels)))


# Plotar gráfico de erro
#plt.plot(errors)
#plt.xlabel("Iteration")
#plt.ylabel("Error")
#plt.title("Training Error")
#plt.show()
