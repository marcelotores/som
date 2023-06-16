import ut
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler #normalisation
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from numpy.ma.core import ceil
from scipy.spatial import distance
from sklearn.metrics import accuracy_score #scoring

import pandas as pd
################# Dados - Ensaio de Baumann #################
data3 = ut.im_data(3)

c1, c2, c3 = data3[0:82, :], data3[82:164, :], data3[175:257, :]
data_file = np.concatenate((c1, c2, c3), axis=0)

#data_x = data_file[:, :24]
data_x = data_file[:, [0, 1, 2, 3, 4, 7, 8, 21]]

data_y = data_file[:, 24]

class SelfOrganizingMap:
    def __init__(self, input_dim, output_dim):
        self.weights = np.random.rand(output_dim[0], output_dim[1], input_dim)

    def train(self, data, labels, num_epochs, learning_rate):
        quantization_errors = []  # Lista para armazenar os valores de erro de quantização
        for epoch in range(num_epochs):
            data, labels = shuffle(data, labels)

            #np.random.shuffle(data)

            ## Definindo parâmetros dinamicamente
            learning_rate, neighbourhood_range = decay(epoch, num_epochs, learning_rate, max_m_dsitance=4)

            for sample in data:
                best_match = self._find_best_matching_unit(sample)

                ## Comentário temporário
                self._update_weights(sample, best_match, learning_rate)

                #### Nova Atualização

                #for row in range(self.weights.shape[0]):
                #  for col in range(self.weights.shape[1]):
                    # Aqui é feito o cálculo de vizinhaça; por exemplo, para row, col = 0, 0 e winter (neurônio vencedor) = 1, 0,
                    # É calculada a distância de Manhatan entre essas duas posições. O limite dessa distância é neighbourhood_range,
                    # valor que vai sendo reduzido com o passar das iterações.

                #    if self.m_distance([row, col], best_match) <= neighbourhood_range:
                #      print(epoch)
                #      self.weights[row][col] += learning_rate * (sample - self.weights[row][col]) #update neighbour's weight

                #### Fim

            quantization_error = self._calculate_quantization_error(data)
            quantization_errors.append(quantization_error)
        return quantization_errors

    def _find_best_matching_unit(self, sample):
        distances = np.linalg.norm(self.weights - sample, axis=2)
        return np.unravel_index(np.argmin(distances), distances.shape)

    def _update_weights(self, sample, best_match, learning_rate):
        x, y = best_match
        self.weights[x, y] += learning_rate * (sample - self.weights[x, y])

    def get_weights(self):
        return self.weights

    def m_distance(self, x, y):
      return distance.cityblock(x,y)

    def _calculate_quantization_error(self, data):
        quantization_error = 0

        for sample in data:
            best_match = self._find_best_matching_unit(sample)
            quantization_error += np.linalg.norm(sample - self.weights[best_match])

        quantization_error /= len(data)

        return quantization_error
    def calculate_accuracy(self, data, labels):
        num_correct = 0

        for i in range(len(data)):
            sample = data[i]
            best_match = self._find_best_matching_unit(sample)
            print(best_match)
            #predicted_label = labels[best_match[0], best_match[1]]
            #true_label = labels[i]

            #if predicted_label == true_label:
                #num_correct += 1

        #accuracy = num_correct / len(data)
        #return accuracy


# Normalização dos Dados
def minmax_scaler(data):
  scaler = MinMaxScaler()
  scaled = scaler.fit_transform(data)
  return scaled

def decay(epoch, num_epochs, max_learning_rate, max_m_dsitance):
  coefficient = 1.0 - (np.float64(epoch)/num_epochs)
  learning_rate = coefficient*max_learning_rate
  neighbourhood_range = ceil(coefficient * max_m_dsitance)
  return learning_rate, neighbourhood_range


train_x, test_x, train_y, test_y = train_test_split(data_x, data_y, test_size=0.2, random_state=42)
print(train_x.shape, train_y.shape, test_x.shape, test_y.shape) # check the shapes


# Normalizar os dados
data_x_normalized = minmax_scaler(data_x)
train_x_normalized = minmax_scaler(train_x)
test_x_normalized = minmax_scaler(test_x)

# Todos os Dados
#X = data_x_normalized
#y = data_y

# Dados de Treino
X = train_x_normalized
y = train_y

# Dados de Teste
#X = test_x_normalized
#y = test_y


# Criar e treinar o mapa auto-organizável
input_dim = X.shape[1]  # Dimensão das entradas
output_dim = (3, 3)  # Dimensão da grade do mapa
num_epochs = 2000  # Número de épocas de treinamento
learning_rate = 0.1  # Taxa de aprendizado


som = SelfOrganizingMap(input_dim, output_dim)

# Dados de treinamento
#quantization_errors = som.train(train_x_normalized, num_epochs, learning_rate)

# Dados de teste
quantization_errors = som.train(X, y, num_epochs, learning_rate)

weights = som.get_weights()

# Acurácia (Ai)
label_data = y
map = np.empty(shape=(output_dim[0], output_dim[1]), dtype=object)

for row in range(output_dim[0]):
  for col in range(output_dim[1]):
    map[row][col] = [] # empty list to store the label

for t in range(X.shape[0]):
  if (t+1) % 1000 == 0:
    print("sample data: ", t+1)

  winner = som._find_best_matching_unit(X[t])
  map[winner[0]][winner[1]].append(label_data[t]) # label of winning neuron

# construct label map
label_map = np.zeros(shape=(output_dim[0], output_dim[1]), dtype=np.int64)

for row in range(output_dim[0]):
  for col in range(output_dim[1]):
    label_list = map[row][col]
    if len(label_list)==0:
      label = 0
    else:
      label = max(label_list, key=label_list.count)
    label_map[row][col] = label

winner_labels = []

for t in range(test_x_normalized.shape[0]):
  winner = som._find_best_matching_unit(test_x_normalized[t])
  row = winner[0]
  col = winner[1]
  predicted = label_map[row][col]
  winner_labels.append(predicted)
print(label_map)
print("Accuracy: ",accuracy_score(test_y, np.array(winner_labels)))

########### Gráfico Dispersão dos Dados ##############

weights_flat = weights.reshape(-1, input_dim)

# Cada ponto no mapa representa uma amostra. São pegos apenas os dois primeiros atributos dos dados, mas podem ser pegos outros.

# Plotar o gráfico de dispersão
plt.scatter(X[:, 0], X[:, 1], c=y)

# plotando os neurônios em cima dos dados
plt.scatter(weights_flat[:, 0], weights_flat[:, 1], color='red', s=100)

plt.xlabel('Atributo 1')
plt.ylabel('Atributo 2')
plt.title('Gráfico de Dispersão dos Pesos dos Dados')
plt.show()



# Plotar o erro de quantização
plt.plot(quantization_errors)
plt.xlabel("Época")
plt.ylabel("Erro de Quantização")
plt.title("Erro de Quantização ao Longo das Épocas")
plt.show()