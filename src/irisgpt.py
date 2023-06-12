import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from minisom import MiniSom

# Carregar o conjunto de dados Iris
iris = load_iris()
data = iris.data
target = iris.target

# Normalizar os dados
data_normalized = (data - data.mean(axis=0)) / data.std(axis=0)

# Criar e treinar o mapa auto-organizável
input_dim = data_normalized.shape[1]  # Dimensão das entradas
output_dim = (10, 10)  # Dimensão da grade do mapa
num_epochs = 100000  # Número de épocas de treinamento
learning_rate = 0.1  # Taxa de aprendizado

som = MiniSom(output_dim[0], output_dim[1], input_dim, sigma=1.0, learning_rate=learning_rate)
som.train_random(data_normalized, num_epochs)

# Obter os pesos treinados
weights = som._weights

# Calcular o erro do mapa auto-organizável manualmente
quantization_errors = []
for x in data_normalized:
    w = som.winner(x)
    quantization_errors.append(np.linalg.norm(x - som._weights[w[0], w[1]]))
quantization_error = np.mean(quantization_errors)
print("Quantization Error:", quantization_error)

# Mapear as instâncias para os neurônios vencedores
mapped = som.win_map(data_normalized)

# Criar uma matriz de coordenadas dos neurônios
neuron_positions = np.array([[i, j] for i in range(output_dim[0]) for j in range(output_dim[1])])

# Plotar o mapa auto-organizável como um gráfico de dispersão
plt.figure(figsize=(8, 6))

for i, (x, y) in enumerate(neuron_positions):
    neuron_data = mapped[(x, y)]
    if neuron_data:
        neuron_data = np.array(neuron_data)
        plt.scatter(neuron_data[:, 0], neuron_data[:, 1], color=plt.cm.viridis(i / (output_dim[0] * output_dim[1])))

plt.colorbar(ticks=range(output_dim[0] * output_dim[1]))
plt.title('Self-Organizing Map (Iris Dataset)')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()

# Plotar o mapa auto-organizável como uma matriz
plt.figure(figsize=(8, 8))
plt.imshow(weights.reshape(output_dim[0] * output_dim[1], input_dim), aspect='auto', cmap='viridis')
plt.colorbar()
plt.title('Self-Organizing Map (Iris Dataset)')
plt.xlabel('Feature Index')
plt.ylabel('Neuron Index')
plt.show()

# Plotar o mapa de distância
plt.figure(figsize=(8, 8))
plt.bone()
plt.pcolor(som.distance_map().T)  # plotting the distance map as background
plt.colorbar()
plt.title('Self-Organizing Map Distance Map (Iris Dataset)')
plt.xlabel('Feature Index')
plt.ylabel('Neuron Index')
plt.show()

# Plotar o erro do mapa auto-organizável
plt.plot(quantization_errors)
plt.title('Self-Organizing Map Quantization Error (Iris Dataset)')
plt.xlabel('Sample Index')
plt.ylabel('Quantization Error')
plt.show()
