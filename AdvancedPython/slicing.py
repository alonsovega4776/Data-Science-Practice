import numpy as np

numeros = np.arange(1, 50, 1)
print('numeros normales: ',numeros)


for i in range(1, len(numeros)):
    brinco = i
    print('numeros despues de un brinco de {}: '.format(brinco), numeros[0::brinco])