import numpy as np
import matplotlib.pyplot as plt


#===============================Carregar Arquivo===================================
#Usando csv.reader
#with open('Advertising.csv') as f:
#    dados = list(csv.reader(f, delimiter=','))

#Usando NumPy
dados = np.genfromtxt("Advertising.csv", delimiter=',', skip_header=1)
X = dados[:,1:4]
y = dados[:,4]
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
    vetorResult = []
    i = 0
    while i < x.shape[0]:
        vetorResult.append(self.predictEq(x[i]))
        i = i + 1
    return vetorResult

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
   #====================================================
      
#======================================================================================



#=============================Metodo de Regressão Multipla=============================
class MultipleLinearRegressor():
  #construtor
  def __init__(self):
    pass

  #Calculo de B1 e B0
  def fit(self, X, y):
      bias = np.ones((X.shape[0], 1))
      X = np.hstack((bias,X))
      inv = np.linalg.inv(np.matmul(np.transpose(X),X))
      self.b = np.matmul(np.matmul(inv,np.transpose(X)), y)

  #Aplicar a equação no X que se recebe como entrada
  def predict(self, X):
      vetorResult = [self.b[0]]
      i = 1
      while i < self.b.shape[0]:
          vetorResult += (self.b[i]*X[:,i-1])
          i = i+1
      return vetorResult
    

  def RSS(self, X):
        totalSum = 0
        for i in range(0, X.shape[0]):
           roundMinus = y[i]
           for j in range(0, self.b.shape[0]-1):
               roundMinus = roundMinus - (self.b[j]*X[i,j])
           totalSum = totalSum + (roundMinus ** 2)
        return totalSum   

  def RSE(self, X):
      return (self.RSS(X)/len(X)) ** 0.5

  def TSS(self, X):
      error = 0
      i = 0
      while(i < len(X)):
          error += ((y[i] - np.mean(y)) ** 2)
          i += 1
      return error

  def R2(self, X):
      return (1 - (self.RSS(X)/self.TSS(X)))
       



   #===================================================


#==========================================================================================



# Simplifica chamada do objeto
slr = SampleLinearRegressor()
mlr = MultipleLinearRegressor()


#================================Front===============================================================================
print("Regressão Linear")
option = input("Opções: 0 - TV x Sales |  1 - Radio x Sales  |  2 - Newspaper x Sales  |  3 - Investimentos x Sales (Regressao Multipla)\
    ")

# Treinar meu modelo usando fit de acordo com a opção
if option == '0' or option == '1' or option == '2':
    slr.fit(X[:,int(option)], y)

    #Plotagem=======================================================================
    
    #De acordo com a opcao selecionada, produz o plot
    if option == '0':
        plt.title("TVs X Sales - B0: " + str(slr.B0) + " - B1: " + str(slr.B1))
        plt.xlabel("TV")
        plt.ylabel("Sales")
        plt.plot(dados[:,1], y, 'go')
        plt.plot(slr.predict(np.array(range(1,300))))

        # Medir erros
        print("RSS: " + str(slr.RSS(dados[:,1], y)))
        print("RSE: " + str(slr.RSE(dados[:,1], y)))
        print("R2: " + str(slr.R2(dados[:,1], y)))
        print("TSS: " + str(slr.TSS(dados[:,1], y)))

    elif option == '1':
        plt.title("Radio X Sales - B0: " + str(slr.B0) + " - B1: " + str(slr.B1))
        plt.xlabel("Radio")
        plt.ylabel("Sales")
        plt.plot(dados[:,2], y, 'go')
        plt.plot(slr.predict(np.array(range(1,50))))

        # Medir erros
        print("RSS: " + str(slr.RSS(dados[:,2], y)))
        print("RSE: " + str(slr.RSE(dados[:,2], y)))
        print("R2: " + str(slr.R2(dados[:,2], y)))
        print("TSS: " + str(slr.TSS(dados[:,2], y)))

    elif option == '2':
        plt.title("Newspaper X Sales - B0: " + str(slr.B0) + " - B1: " + str(slr.B1))
        plt.xlabel("Newspaper")
        plt.ylabel("Sales")        
        plt.plot(dados[:,1], y, 'go')
        plt.plot(dados[:,2], y, 'go')
        plt.plot(dados[:,3], y, 'go')
        plt.plot(dados[:,0], slr.predict(np.array(range(1,120))))

        # Medir erros
        print("RSS: " + str(mlr.RSS(X, y)))
        print("RSE: " + str(mlr.RSE(X, y)))
        print("R2: " + str(mlr.R2(X, y)))
        print("TSS: " + str(mlr.TSS(X, y)))

    # B1 e B0
    print("B1: " + str(slr.B1))
    print("B0: " + str(slr.B0))
    

    # Plotar na Tela
    plt.show()

elif option == '3':
    mlr.fit(X, y)
    plt.title("Regressão Linear Múltipla")
    plt.xlabel("Investimentos")
    plt.ylabel("Sales")
    plt.plot(mlr.predict(X), 'go')
    plt.show()

    print(mlr.predict(X))

    print("B3: " + str(mlr.b[3]))
    print("B2: " + str(mlr.b[2]))
    print("B1: " + str(mlr.b[1]))
    print("B0: " + str(mlr.b[0]))

    print("RSS: " + str(mlr.RSS(X)))
    print("RSE: " + str(mlr.RSE(X)))
    print("TSS: " + str(mlr.TSS(X)))
    print("R2: " + str(mlr.R2(X)))


else:
    print("nao foi")




#=====================================================================================