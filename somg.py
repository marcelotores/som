import numpy as np
import matplotlib.pyplot as plt

class SelfOrganizingMap:
    def __init__(self, input_dim, output_dim):
        self.weights = np.random.rand(output_dim[0], output_dim[1], input_dim)
        # weights = 10 10 2

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
data = np.random.rand(10, 4)


# Criar e treinar o mapa auto-organizável
input_dim = 4  # Dimensão das entradas
output_dim = (2, 2)  # Dimensão da grade do mapa
num_epochs = 100  # Número de épocas de treinamento
learning_rate = 0.1  # Taxa de aprendizado


som = SelfOrganizingMap(input_dim, output_dim)
som.train(data, num_epochs, learning_rate)

# Obter os pesos treinados
weights = som.get_weights()

# Plotar o mapa auto-organizável (scatter)
# plt.figure(figsize=(8, 8))
#
# for x in range(output_dim[0]):
#     for y in range(output_dim[1]):
#         print(weights[x, y, 0], weights[x, y, 1])
#         # Nesse caso ele usa as os pesos para exibir as coordenadas?
#         plt.scatter(weights[x, y, 0], weights[x, y, 1],  color='b')
# plt.scatter(data[:, 0], data[:, 1], color='r')
# plt.title('Self-Organizing Map')
# plt.xlabel('Feature 1')
# plt.ylabel('Feature 2')
# plt.show()

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

# Matriz
# Plotar o mapa auto-organizável como uma matriz
# plt.figure(figsize=(8, 8))
# plt.imshow(weights.reshape(output_dim[0]*output_dim[1], input_dim), aspect='auto', cmap='viridis')
# plt.colorbar()
# plt.title('Self-Organizing Map')
# plt.xlabel('Feature Index')
# plt.ylabel('Neuron Index')
# plt.show()




