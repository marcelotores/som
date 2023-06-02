import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.utils import shuffle

class SelfOrganizingMap:
    def __init__(self, input_dim, output_dim):
        self.weights = np.random.rand(output_dim[0], output_dim[1], input_dim)

    def train(self, data, num_epochs, learning_rate):
        for epoch in range(num_epochs):
            np.random.shuffle(data)

            for sample in data:
                best_match = self._find_best_matching_unit(sample)
                self._update_weights(sample, best_match, learning_rate)

    def _find_best_matching_unit(self, sample):
        distances = np.linalg.norm(self.weights - sample, axis=2)
        return np.unravel_index(np.argmin(distances), distances.shape)

    def _update_weights(self, sample, best_match, learning_rate):
        x, y = best_match
        self.weights[x, y] += learning_rate * (sample - self.weights[x, y])

    def get_weights(self):
        return self.weights

# Carregar o conjunto de dados Iris
iris = load_iris()
data = iris.data

# Normalizar os dados
data_normalized = (data - data.mean(axis=0)) / data.std(axis=0)

# Criar e treinar o mapa auto-organizável
input_dim = data.shape[1]  # Dimensão das entradas
output_dim = (2, 2)  # Dimensão da grade do mapa
num_epochs = 1000  # Número de épocas de treinamento
learning_rate = 0.1  # Taxa de aprendizado


som = SelfOrganizingMap(input_dim, output_dim)
som.train(data_normalized, num_epochs, learning_rate)

# Obter os pesos treinados
weights = som.get_weights()

# Obter as classes dos dados Iris
target = iris.target

# Plotar o mapa auto-organizável como uma matriz
#plt.figure(figsize=(10, 10))
#plt.imshow(weights.reshape(output_dim[0]*output_dim[1], input_dim), aspect='auto', cmap='viridis')
#plt.colorbar()
#plt.title('Self-Organizing Map (Iris Dataset)')
#plt.xlabel('Feature Index')
#plt.ylabel('Neuron Index')
#plt.xticks(range(input_dim), iris.feature_names, rotation=45)
#plt.yticks(range(output_dim[0]*output_dim[1]), range(output_dim[0]*output_dim[1]))
#plt.tight_layout()
#plt.show()


# Plot
plt.figure(figsize=(8, 8))
for x in range(output_dim[0]):
    for y in range(output_dim[1]):
        print(weights[x, y, 0], weights[x, y, 1])
        plt.plot(weights[x, y, 0], weights[x, y, 1], 'bo', markersize=4)
for i in range(len(data)):
    plt.plot(data[i, 0], data[i, 1], 'ro', markersize=4)
plt.title('Self-Organizing Map')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()

#plt.figure(figsize=(8, 6))

#for i, (x, y) in enumerate(weights.reshape(-1, 2)):
#    plt.scatter(x, y, c=target[i], cmap='viridis')

#plt.colorbar(ticks=np.unique(target))
#plt.title('Self-Organizing Map (Iris Dataset)')
#plt.xlabel('Feature 1')
#plt.ylabel('Feature 2')
#plt.show()