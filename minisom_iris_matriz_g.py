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
num_epochs = 100  # Número de épocas de treinamento
learning_rate = 0.1  # Taxa de aprendizado

som = MiniSom(output_dim[0], output_dim[1], input_dim, sigma=1.0, learning_rate=0.5)
som.train_random(data_normalized, num_epochs)

# Obter os pesos treinados
weights = som.get_weights()

# Mapear os rótulos de destino para cores
color_map = np.array(['b', 'g', 'r'])  # Mapeamento de cores para cada categoria (0, 1, 2)

# Plotar o mapa auto-organizável com cores de acordo com a categoria
plt.figure(figsize=(8, 8))
for i, (x, t) in enumerate(zip(data_normalized, target)):
    winner = som.winner(x)  # Obter o neurônio vencedor para a instância
    plt.plot(winner[0] + 0.5, winner[1] + 0.5, color=color_map[t], marker='o', markersize=8)

plt.imshow(np.zeros((output_dim[0], output_dim[1], 3)), cmap='viridis', aspect='auto')
plt.title('Self-Organizing Map (Iris Dataset)')
plt.xlabel('Neuron Index (X-axis)')
plt.ylabel('Neuron Index (Y-axis)')
plt.xticks(np.arange(output_dim[0] + 1))
plt.yticks(np.arange(output_dim[1] + 1))
plt.grid(True, color='gray', linestyle='-', linewidth=0.5)
plt.show()
