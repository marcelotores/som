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
output_dim = (5, 5)  # Dimensão da grade do mapa
num_epochs = 100  # Número de épocas de treinamento
learning_rate = 0.1  # Taxa de aprendizado

som = MiniSom(output_dim[0], output_dim[1], input_dim, sigma=1.0, learning_rate=0.5)
som.train_random(data_normalized, num_epochs)

# Obter os pesos treinados
weights = som.get_weights()

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

# Obter os rótulos das categorias
target = np.genfromtxt(target, delimiter=',', usecols=(4), dtype=str)

# Criar um dicionário para mapear as categorias para números
category_dict = {'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2}

# Mapear os rótulos das categorias para números usando o dicionário
target_numeric = np.array([category_dict[category] for category in target])

# Plotar o gráfico da matriz com cores diferentes para cada categoria
plt.figure(figsize=(8, 8))
plt.imshow(weights.reshape(output_dim[0] * output_dim[1], input_dim), aspect='auto', cmap='viridis', vmin=0, vmax=2)
plt.colorbar(ticks=[0, 1, 2], label='Category')
plt.title('Self-Organizing Map (Iris Dataset)')
plt.xlabel('Feature Index')
plt.ylabel('Neuron Index')
plt.show()

# Plotar o mapa auto-organizável como uma matriz
#plt.figure(figsize=(8, 8))
#plt.imshow(weights.reshape(output_dim[0] * output_dim[1], input_dim), aspect='auto', cmap='viridis')
#plt.colorbar()
#plt.title('Self-Organizing Map (Iris Dataset)')
#plt.xlabel('Feature Index')
#plt.ylabel('Neuron Index')
#plt.show()