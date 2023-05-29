import numpy as np
import matplotlib.pyplot as plt

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

# Exemplo de uso
# Gerar dados de treinamento
data = np.random.rand(100, 2)
print(data)
exit()

# Criar e treinar o mapa auto-organizável
input_dim = 2  # Dimensão das entradas
output_dim = (10, 10)  # Dimensão da grade do mapa
num_epochs = 100  # Número de épocas de treinamento
learning_rate = 0.1  # Taxa de aprendizado
som = SelfOrganizingMap(input_dim, output_dim)
som.train(data, num_epochs, learning_rate)

# Obter os pesos treinados
weights = som.get_weights()

# Plotar o mapa auto-organizável
plt.figure(figsize=(8, 8))

for x in range(output_dim[0]):
    for y in range(output_dim[1]):
        plt.scatter(weights[x, y, 0], weights[x, y, 1], color='b')
plt.scatter(data[:, 0], data[:, 1], color='r')
plt.title('Self-Organizing Map')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()
