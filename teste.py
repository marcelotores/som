

import matplotlib.pyplot as plt
import numpy as np

# Plotar o mapa auto-organizável
plt.figure(figsize=(6, 6))

# Coordenadas dos neurônios no mapa
neuron_coords = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])  # Coordenadas dos 4 neurônios

# Plotar os neurônios do mapa
for i, coord in enumerate(neuron_coords):
    plt.scatter(weights[i][0], weights[i][1], color='b', label=f'Neuron {i + 1}')

# Plotar os dados de treinamento
for sample in T:
    plt.scatter(sample[0], sample[1], color='r')

# Plotar o teste sample
plt.scatter(s[0], s[1], color='g', label='Test Sample')

# Definir limites do gráfico
x_min, x_max = np.min(weights[:, 0]), np.max(weights[:, 0])
y_min, y_max = np.min(weights[:, 1]), np.max(weights[:, 1])
x_margin = (x_max - x_min) * 0.1
y_margin = (y_max - y_min) * 0.1
plt.xlim(x_min - x_margin, x_max + x_margin)
plt.ylim(y_min - y_margin, y_max + y_margin)

plt.title('Self-Organizing Map')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.show()