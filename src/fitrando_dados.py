import numpy as np
from sklearn.preprocessing import MinMaxScaler

import ut


#
#0, 16
#1, 8, 10, (20, 23)
#2
#3, 6, 15, 17, 18, 19, 22
#4, 9, 11, (14, 16),
#5, 12, 13 x
#7
#21
#24
def minmax_scaler(data):
  scaler = MinMaxScaler()
  scaled = scaler.fit_transform(data)
  return scaled

################# Dados - Ensaio de Baumann #################
data3 = ut.im_data(3)

c1, c2, c3 = data3[0:82, :], data3[82:164, :], data3[175:257, :]
data_file = np.concatenate((c1, c2, c3), axis=0)

data_x = data_file[:, :24]
data_y = data_file[:, 24]

train_x_norm = minmax_scaler(data_x) # normalisation

cor = train_x_norm[:, [0, 1, 2, 3, 4, 7, 8, 21]]

print(train_x_norm[:, 5])
print(cor[:, 5])