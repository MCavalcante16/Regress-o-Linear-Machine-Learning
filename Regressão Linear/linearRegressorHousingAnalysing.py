import numpy as np
import matplotlib.pyplot as plt


#===============================Carregar Arquivo===================================
#Usando csv.reader
#with open('Advertising.csv') as f:
#    dados = list(csv.reader(f, delimiter=','))

#Usando NumPy
dados = np.genfromtxt("housing.csv", skip_header=0)

#Embaralhando as amostras
np.random.shuffle(dados)

#Separa a variável LSTAT
X = dados[:,12]

#Separa a variável MEDV
y = dados[:,13]




#===================================================================================


#=========================Método de regressão simples===========================================================
class SampleLinearRegressor():
  #construtor
  def __init__(self):
    pass

  #Calculo de B1 e B0
  def fit(self, X, y):
    xMean = np.mean(X)
    self.yMean = np.mean(y)

    sumUpB1 = 0
    sumDownB1 = 0
    i = 0
    while(i < X.shape[0] ):
        sumUpB1 += (X[i] - xMean) * (y[i] - self.yMean)
        sumDownB1 += (X[i] - xMean) ** 2
        self.B1 = sumUpB1/sumDownB1
        self.B0 = self.yMean - (self.B1 * xMean)
        i = i+1

  #Aplicar a equação no X que se recebe como entrada
  def predictEq(self, x):
      return (self.B0 + (self.B1 * x))

  #Aplica o predict em cada item do vetor da entrada
  def predict(self, x):
    return self.B0 + self.B1*x

   #===================================================



   #==============Medição de Erros======================
  def RSS(self, X, y):
       error = 0
       i = 0
       while i < len(X):
           error = error + ((y[i] - self.predictEq(X[i])) ** 2)
           i = i + 1
       return error

  def RSE(self, X, y):
      return (self.RSS(X, y)/len(X)) ** 0.5

  def TSS(self, X, y):
      error = 0
      i = 0
      while(i < len(X)):
          error += ((y[i] - self.yMean) ** 2)
          i += 1
      return error

  def R2(self, X, y):
      return (1 - (self.RSS(X, y)/self.TSS(X, y)))

  def MSE(self, y):
      return (np.sum((y - self.predict(y)) ** 2)/y.shape[0])

   #====================================================
      
#======================================================================================



# Simplifica chamada do objeto
slr = SampleLinearRegressor()



#================================Front===============================================================================
print("Regressão Linear")
option = input("Opções: 1 - LSTAT x MEDV |  2 - LSTAT² x MEDV  |  3 - LSTAT³ x MEDV\
    ")

# Treinar meu modelo usando fit de acordo com a opção (80% de Treino - 20% de Teste)
if option == '1' or option == '2' or option == '3':

    #De acordo com a opcao selecionada, seleciona a elevação de LSTAT
    if option == '2':
        np.power(X,2)

    elif option == '3':
        np.power(X,3)

    #Separa o que é pra treino e o que é pra teste
    pTreino = X[:int(X.shape[0]*0.8)]
    pTeste  = X[int(X.shape[0]*0.8):]
    alvoTreino = y[:int(y.shape[0]*0.8)]

    #Treinando a predicao com as amostras de treino
    slr.fit(pTreino, y)

    #Plotagem
    plt.title("LSTAT" + option +  " x MEDV")
    plt.xlabel("LSTAT" + option)
    plt.ylabel("MEDV")
    plt.plot(X, y, 'go')
    plt.plot(slr.predict(np.array(range(1,40))))

    # Medir erros
    print("Resíduos - Conjunto de Teste")
    print("RSS: " + str(slr.RSS(pTeste, y)))
    print("RSE: " + str(slr.RSE(pTeste, y)))
    print("R2: " + str(slr.R2(pTeste, y)))
    print("TSS: " + str(slr.TSS(pTeste, y)))
    print("MSE: " + str(slr.MSE(y)))

    print("Resíduos - Conjunto de Treino")
    print("RSS: " + str(slr.RSS(pTreino, y)))
    print("RSE: " + str(slr.RSE(pTreino, y)))
    print("R2: " + str(slr.R2(pTreino, y)))
    print("TSS: " + str(slr.TSS(pTreino, y)))
    print("MSE: " + str(slr.MSE(y)))

    plt.show()

    plt.title("Conjunto de Treino x Valores Alvo")
    plt.xlabel("Conjunto de Treino")
    plt.ylabel("Valores Alvo")
    predicted = slr.predict(pTreino)
    print(predicted)
    plt.plot(predicted, alvoTreino, 'go')
    plt.plot(range(1,40),range(1,40))
    plt.show()
    print("B1: " + str(slr.B1))
    print("B0: " + str(slr.B0))
    



else:
    print("Entrada inválida")




#=====================================================================================