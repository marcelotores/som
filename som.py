import math
import matplotlib.pyplot as plt
import numpy as np
class SOM:

    # Function here computes the winning vector
    # by Euclidean distance
    def winner(self, weights, sample):

        D0 = 0
        D1 = 0

        for i in range(len(sample)):

            D0 = D0 + math.pow((sample[i] - weights[0][i]), 2)
            D0 = D0 + math.pow((sample[i] - weights[0][i]), 2)
            D1 = D1 + math.pow((sample[i] - weights[1][i]), 2)

        # Selecting the cluster with smallest distance as winning cluster

        if D0 < D1:
            return 0
        else:
            return 1

    # Function here updates the winning vector
    def update(self, weights, sample, J, alpha):
        # Here iterating over the weights of winning cluster and modifying them
        for i in range(len(weights[0])):
            weights[J][i] = weights[J][i] + alpha * (sample[i] - weights[J][i])

        return weights

# Driver code


def main():

    # Training Examples ( m, n )
    #T = [[1, 1, 0, 0], [0, 0, 0, 1], [1, 0, 0, 0], [0, 0, 1, 1], [1, 0, 1, 1], [0, 1, 1, 1], [0, 0, 0, 0]]
    #T = np.array([[1, 1, 0, 0], [0, 0, 0, 1], [1, 0, 0, 0], [0, 0, 1, 1], [1, 0, 1, 1], [0, 1, 1, 1], [0, 0, 0, 0]])
    data = np.random.rand(100, 4)
    T = data
    # exit()
    m, n = len(T), len(T[0])

    # weight initialization ( n, C )
    weights = [[0.2, 0.6, 0.5, 0.9], [0.8, 0.4, 0.7, 0.3],  [0.8, 0.4, 0.7, 0.3]]
    print(weights[0])
    # training
    ob = SOM()

    epochs = 1000
    alpha = 0.5

    for i in range(epochs):
        for j in range(m):

            # training sample
            sample = T[j]

            # Compute winner vector
            J = ob.winner(weights, sample)

            # Update winning vector
            weights = ob.update(weights, sample, J, alpha)

    # classify test sample
    s = [0, 0, 0, 1]
    J = ob.winner(weights, s)

    print("Test Sample s belongs to Cluster : ", J)
    print("Trained weights : ", weights)



    # Plotar o mapa auto-organizável
    plt.figure(figsize=(6, 6))

    # Coordenadas dos neurônios no mapa
    neuron_coords = np.array([[0, 0], [1, 0], [0, 1]])

    # Plotar os neurônios do mapa
    for i, coord in enumerate(neuron_coords):
        plt.scatter(weights[i][0], weights[i][1], color='b', label=f'Neuron {i + 1}')

    # Plotar os dados de treinamento
    for sample in T:
        plt.scatter(sample[0], sample[1], color='r')

    # Plotar o teste sample
    plt.scatter(s[0], s[1], color='g', label='Test Sample')



    plt.title('Self-Organizing Map')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()



