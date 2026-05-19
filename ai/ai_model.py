from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
import tensorflow as tf
from tensorflow.keras import layers, models,optimizers
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import joblib

#####=======================================================================================================
##### O X são os dados numéricos da música, como MFCCs, Tom, Ritmo, etc.
##### O y é a resposta certa para o conjunto de dados numéricos daquele X emespecífico.
##### O y é a popularidade da música.
##### Então, o objetivo é conseguir prever a popularidade de uma música com base nos dados numéricos.
#####=======================================================================================================

path = os.getcwd() + r"\dataset"

#================================ Carregar os dados para treino e teste =====================================#

# Vou usar apenas esse grupo de dados, que vão dar 343 colunas e não 518.
colunas_selecionadas = [
    'chroma_stft', 
    'tonnetz', 
    'mfcc', 
    'spectral_centroid', 
    'spectral_bandwidth', 
    'spectral_contrast', 
    'spectral_rolloff', 
    'rmse' # No FMA original, o RMS é chamado de 'rmse'
]

# Carregando os metadados (quem é a música)
# header=[0, 1] porque o tracks.csv tem duas linhas de cabeçalho
tracks = pd.read_csv(path+r'\tracks.csv', index_col=0, header=[0, 1])

# Carregando as características (os números para a IA)
# header=[0, 1, 2] porque o features.csv tem três linhas de cabeçalho
features = pd.read_csv(path + r'\features.csv', index_col=0, header=[0, 1, 2])

## USANDO O DATASET MEDIUM
# # 1. Pegando apenas o subset small e medium, com 25.000 músicas
# subset_mask = tracks[('set', 'subset')].isin(['small', 'medium'])
# subset_ids = tracks.index[subset_mask]

# # 2. Pegar os rótulos (y) e as características (X) apenas desse subset
# y = tracks.loc[subset_ids, ('track', 'listens')].values
# X = features.loc[subset_ids, colunas_selecionadas].values

## USANDO O DATASET FULL
# 1. Pegar os rótulos (y) e as características (X)
y = tracks[('track', 'listens')].values
X = features[colunas_selecionadas].values

# Como 'listens' pode ter números gigantescos, aplicar o logaritmo 
# ajuda MUITO a rede neural a convergir mais rápido.
y = np.log1p(y) 

print(f"Dados prontos! Temos {X.shape[0]} músicas e {X.shape[1]} características para cada uma.")

#================================ Dividindo dados de treino e teste =======================================#

# 80% para treino e 20% para teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalizando os dados (deixando tudo na mesma escala)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Salvar o scaler para usar no teste de outras musicas
joblib.dump(scaler, 'scaler.joblib')
print("Scaler salvo")

#================================ Criando a Rede Neural =======================================#

# 1. Expandindo a capacidade da rede para aguentar as 106k músicas
model = models.Sequential([
    # Camada de entrada mais robusta
    layers.Dense(512, activation='relu', input_shape=(X_train.shape[1],)),
    layers.BatchNormalization(), # Estabiliza o aprendizado entre as camadas
    layers.Dropout(0.3),
    
    # Camadas intermediárias
    layers.Dense(256, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    
    layers.Dense(128, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.2),
    
    layers.Dense(64, activation='relu'),
    
    # Camada de saída para regressão linear pura
    layers.Dense(1, activation='linear')
])

# 2. Controlando a taxa de aprendizado (Crucial para regressão logarítmica)
# O padrão do Adam (0.001) é muito rápido. Reduzir para 0.0005 ajuda a achar o ajuste fino.
otimizador_ajustado = optimizers.Adam(learning_rate=0.0005)

model.compile(optimizer=otimizador_ajustado, loss='mse', metrics=['mae'])

# model.summary() # Mostra o desenho da rede no terminal


#================================ Treinando a Rede Neural =======================================#

# Aumentamos o batch_size para 128 ou 256. Com 106k músicas, lotes maiores dão 
# um direcionamento matemático muito mais estável para o gradiente da rede.
# IMPORTANTE: Adicione validation_data se você tiver o X_val/y_val separado!
model.fit(
    X_train, y_train, 
    epochs=60, 
    batch_size=128, # Subiu de 32 para 128
    validation_split=0.15 # Usa 15% dos dados para validar e o EarlyStopping funcionar
)

# Salvando o modelo treinado
model.save('music_quality_prediction_model.keras')


#================================ Testando o modelo treinado =======================================#

# y com os valores previstos, ou o y^
y_pred = model.predict(X_test)

#MAE (Mean Absolute Error): Diz, em média, o quanto a IA errou para mais ou para menos.
# R^2 Score (Coeficiente de Determinação): Diz a porcentagem de variação dos dados que seu modelo consegue explicar (vai de -infinito até 1.0, onde 1.0 é a perfeição).

# Calculando o erro médio absoluto
mae = mean_absolute_error(y_test, y_pred)
# Calculando o R² (quanto mais próximo de 1, melhor)
r2 = r2_score(y_test, y_pred)

print(f"Erro Médio Absoluto (MAE): {mae:.4f}")
print(f"Coeficiente R²: {r2:.4f}")


#================================ Plotando gráficos do teste =======================================#

plt.figure(figsize=(8, 6))
# Plota os valores reais vs os valores preditos
plt.scatter(y_test, y_pred, alpha=0.5, color='purple')

# Linha diagonal que representa a perfeição (onde Predito = Real)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)

plt.xlabel('Valores Reais (y_test)')
plt.ylabel('Previsões da IA (y_pred)')
plt.title('Comparação: Real vs. Predito')
plt.grid(True)
plt.show()