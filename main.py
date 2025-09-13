import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("C:\\Users\\vitor.matheus\\Music\\GIT - Pessoal\\Mentoria-DevOps\\dados_regressao.csv")

print(data.head())

plt.scatter(data.tempoEstudo, data.pontuacao)
plt.xlabel("Tempo de Estudo (horas)")
plt.ylabel("Pontuação")
plt.title("Relação entre Tempo de Estudo e Pontuação")
plt.show()

def loss_funcition(m, b, data):
    total_erros = 0
    for i in range(len(data)):
        x = data.iloc[i].tempoEstudo
        y = data.iloc[i].pontuacao
        total_erros += (y - (m  * x + b )) ** 2
    total_erros  / float(len(data))

def gradient_descent(m, b, data, learning_rate):
    m_gradient = 0 # derivada parcial em relação a m
    b_gradient = 0 # derivada parcial em relação a b

    n = len(data)

    for i in range(n):
        x = data.iloc[i].tempoEstudo
        y = data.iloc[i].pontuacao

        m_gradient += -(2/n) * x + (y - (m * x + b))
        b_gradient += -(2/n) * (y - (m * x + b))

    m_final = m - m_gradient * learning_rate
    b_final = b - b_gradient * learning_rate

    return m_final, b_final

m = 0
b = 0
learning_rate = 0.0001
epochs = 1000

for i in range(epochs):
    if i % 100 == 0:
        print(f"Epoch: {i}")
    m, b = gradient_descent(m, b, data, learning_rate)

    #print(f"Epoch {i+1}: m = {m}, b = {b}, loss = {loss_funcition(m, b, data)}")

print(f"Final parameters: m = {m}, b = {b}")

plt.scatter(data.tempoEstudo, data.pontuacao, color="black")
#plt.plot(data.tempoEstudo, m * data.tempoEstudo + b, color="red")
plt.plot(list(range(20, 80)), [m * x + b for x in range(20, 80)], color="red")
# plt.xlabel("Tempo de Estudo (horas)")
# plt.ylabel("Pontuação")
# plt.title("Relação entre Tempo de Estudo e Pontuação")
plt.show()
