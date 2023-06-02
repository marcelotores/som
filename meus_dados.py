import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from minisom import MiniSom
import ut

# Carregar o conjunto de dados Iris
iris = load_iris()
data = iris.data
target = iris.target

################

data3 = ut.im_data(3)
#print(np.count_nonzero(data3[:, 24] == 1))

u1 = data3[0:82, :]
u2 = data3[82:164, :]
u3 = data3[175:257, :]
u1_u2_u3 = np.concatenate((u1, u2, u3), axis=0)

# Normalizando os dados
u1_u2_u3 /= np.max(u1_u2_u3, axis=0)


################

# Normalizar os dados
#data_normalized = (data - data.mean(axis=0)) / data.std(axis=0)
data_normalized = u1_u2_u3

# Criar e treinar o mapa auto-organizável
input_dim = data_normalized.shape[1]  # Dimensão das entradas
output_dim = (10, 10)  # Dimensão da grade do mapa
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

# Plotar o mapa auto-organizável como uma matriz
#plt.figure(figsize=(8, 8))
#plt.imshow(weights.reshape(output_dim[0] * output_dim[1], input_dim), aspect='auto', cmap='viridis')
#plt.colorbar()
#plt.title('Self-Organizing Map (Iris Dataset)')
#plt.xlabel('Feature Index')
#plt.ylabel('Neuron Index')
#plt.show()

# Plotting the response for each pattern in the iris dataset
plt.bone()
plt.pcolor(som.distance_map().T)  # plotting the distance map as background
plt.colorbar()
plt.show()