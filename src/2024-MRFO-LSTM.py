# Instalar as bibliotecas necessárias
%pip install pandas numpy matplotlib scikit-learn keras tensorflow-macos tensorflow-metal prometheus-api-client

# Importar as bibliotecas
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras

# Definir a função MRFO

class MRFO:
    def __init__(self, obj_func, dim, SearchAgents_no, max_iter, lb, ub):
        self.obj_func = obj_func  # Função objetivo a ser minimizada
        self.dim = dim  # Dimensionalidade do problema (número de hiperparâmetros)
        self.SearchAgents_no = SearchAgents_no  # Número de agentes (manta rays)
        self.max_iter = max_iter  # Número máximo de iterações
        self.lb = lb  # Limites inferiores dos parâmetros
        self.ub = ub  # Limites superiores dos parâmetros

        # Inicialização das posições dos agentes
        self.positions = np.random.uniform(0, 1, (self.SearchAgents_no, self.dim)) * (self.ub - self.lb) + self.lb
        self.fitness = np.array([self.obj_func(ind) for ind in self.positions])

        # Encontrar a melhor posição inicial
        self.best_idx = np.argmin(self.fitness)
        self.gbest = self.positions[self.best_idx].copy()
        self.gbest_fitness = self.fitness[self.best_idx]

    def chain_foraging(self, i):
        """Implementação da fase de Chain Foraging"""
        r = np.random.rand(self.dim)
        self.positions[i] = self.positions[i] + r * (self.gbest - self.positions[i])

    def cyclone_foraging(self, i, t, max_iter):
        """Implementação da fase de Cyclone Foraging"""
        r = np.random.rand(self.dim)
        A = 2 * (1 - t / max_iter)  # Fator de controle que diminui ao longo das iterações
        direction = np.random.choice([-1, 1], size=self.dim)  # Escolher aleatoriamente o sentido (horário/anti-horário)
        self.positions[i] = self.positions[i] + A * direction * r * (self.gbest - self.positions[i])

    def somersault_foraging(self, i):
        """Implementação da fase de Somersault Foraging"""
        S = 2 * np.random.rand(self.dim) - 1  # Vetor aleatório entre -1 e 1
        somersault_factor = 2  # Fator que controla a magnitude do somersault
        self.positions[i] = self.positions[i] + somersault_factor * (S * self.gbest - self.positions[i])

    def optimize(self):
        for t in range(self.max_iter):
            for i in range(self.SearchAgents_no):
                # Escolher aleatoriamente uma fase de forrageamento a ser aplicada
                if np.random.rand() < 0.5:
                    # Chain Foraging
                    self.chain_foraging(i)
                else:
                    # Cyclone Foraging
                    self.cyclone_foraging(i, t, self.max_iter)

                # Aplicar o limite de busca
                self.positions[i] = np.clip(self.positions[i], self.lb, self.ub)

                # Avaliar a nova posição
                fitness_candidate = self.obj_func(self.positions[i])

                # Atualizar a posição e a melhor solução encontrada
                if fitness_candidate < self.fitness[i]:
                    self.fitness[i] = fitness_candidate
                    if fitness_candidate < self.gbest_fitness:
                        self.gbest_fitness = fitness_candidate
                        self.gbest = self.positions[i].copy()

            # Somersault Foraging aplicado a todos os agentes uma vez por iteração
            for i in range(self.SearchAgents_no):
                self.somersault_foraging(i)

            print(f"Iteração {t+1}/{self.max_iter}, Melhor Fitness: {self.gbest_fitness}")

        return self.gbest, self.gbest_fitness


# Definir a função objetivo
def objective_function(params):
    """
    Função objetivo para o MRFO.
    params: array-like, contendo os hiperparâmetros [lstm_units, dropout_rate, batch_size, learning_rate]
    Retorna: RMSE do modelo treinado com os hiperparâmetros especificados.
    """
    lstm_units = int(params[0])
    dropout_rate = params[1]
    batch_size = int(params[2])
    learning_rate = params[3]

    # Definir o modelo com os hiperparâmetros especificados
    model = Sequential()
    model.add(LSTM(lstm_units, activation='tanh', input_shape=(n_steps, n_features), return_sequences=True))
    model.add(Dropout(dropout_rate))
    model.add(LSTM(lstm_units, activation='tanh'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(1))
    
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mse')

    # Treinar o modelo
    history = model.fit(X_train, y_train, epochs=10, batch_size=batch_size, verbose=0)

    # Fazer previsões no conjunto de validação (test)
    test_pred = model.predict(X_test)
    test_pred = scaler.inverse_transform(test_pred)
    actual = test.values[n_steps:]

    # Calcular RMSE
    rmse = np.sqrt(mean_squared_error(actual, test_pred))
    
    return rmse

# Carregar os dados
metric_df = pd.read_pickle("../data/ts.pkl")

# Resample para intervalos de 30 minutos
ts = metric_df["value"].astype(float).resample("30min").mean()

# Dividir em treino e teste
train = ts[:"2021-02-07"]
test = ts["2021-02-08":]

# Escalonar os dados
scaler = StandardScaler()
train_scaled = scaler.fit_transform(train.values.reshape(-1, 1))
test_scaled = scaler.transform(test.values.reshape(-1, 1))

# Definir númfero de passos e features
n_steps = 40
n_features = 1

# Criar sequências para treino
X_train, y_train = [], []
for i in range(n_steps, len(train_scaled)):
    X_train.append(train_scaled[i-n_steps:i, 0])
    y_train.append(train_scaled[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)

# Reshape para 3D
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], n_features))

# Criar sequências para teste
X_test, y_test = [], []
for j in range(n_steps, len(test_scaled)):
    X_test.append(test_scaled[j-n_steps:j, 0])
    y_test.append(test_scaled[j, 0])
X_test, y_test = np.array(X_test), np.array(y_test)

# Reshape para 3D
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], n_features))

# Configurações do MRFO
lb = np.array([50, 0.2, 32, 0.0001])  # Limites inferiores
ub = np.array([150, 0.5, 128, 0.001])  # Limites superiores
dim = 4  # Número de hiperparâmetros
SearchAgents_no = 10  # Número de agentes
max_iter = 20  # Número de iterações

# Instanciar e executar o MRFO
mrfo = MRFO(obj_func=objective_function, dim=dim, SearchAgents_no=SearchAgents_no, max_iter=max_iter, lb=lb, ub=ub)
#best_params, best_fitness = mrfo.optimize()

print(f"Melhores Hiperparâmetros: {best_params}")
print(f"Melhor Fitness (RMSE): {best_fitness}")

print("Melhores Hiperparâmetros Encontrados:")
print(f"LSTM Units: {int(best_params[0])}")
print(f"Dropout Rate: {best_params[1]:.2f}")
print(f"Batch Size: {int(best_params[2])}")
#print(f"Learning Rate: {best_params[3]:.5f}")
print(f"Learning Rate: {best_params[3]::.5f}")
print(f"Melhor RMSE: {best_fitness}")