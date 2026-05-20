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

# Vou usar apenas esse grupo de dados, que vão dar 343 colunas, e não 518.
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

# Carregando os metadados 
# header=[0, 1] porque o tracks.csv tem duas linhas de cabeçalho
tracks = pd.read_csv(path+r'\tracks.csv', index_col=0, header=[0, 1])

# Carregando as características 
# header=[0, 1, 2] porque o features.csv tem três linhas de cabeçalho
features = pd.read_csv(path + r'\features.csv', index_col=0, header=[0, 1, 2])



# ## USANDO O DATASET SMALL
# # Pegando apenas o subset small com 8000 músicas
# subset_mask = tracks[('set', 'subset')].isin(['small'])
# subset_ids = tracks.index[subset_mask]

# # Pegar os rótulos (y) e as características (X) apenas desse subset
# y = tracks.loc[subset_ids, ('track', 'listens')].values
# X = features.loc[subset_ids, colunas_selecionadas].values



# ## USANDO O DATASET MEDIUM
# # Pegando apenas o subset small e medium, com 25.000 músicas
# subset_mask = tracks[('set', 'subset')].isin(['small', 'medium'])
# subset_ids = tracks.index[subset_mask]

# # Pegar os rótulos (y) e as características (X) apenas desse subset
# y = tracks.loc[subset_ids, ('track', 'listens')].values
# X = features.loc[subset_ids, colunas_selecionadas].values



# USANDO O DATASET FULL
# Pegar os rótulos (y) e as características (X)
y = tracks[('track', 'listens')].values
X = features[colunas_selecionadas].values



# Como 'listens' pode ter números gigantescos, aplicar o logaritmo 
# ajuda a rede neural a convergir mais rápido.
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


model = models.Sequential([
    # Camada de entrada 
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
    
    # Camada de saida
    layers.Dense(1, activation='linear')
])

# 2. Controlando a taxa de aprendizado 
otimizador_ajustado = optimizers.Adam(learning_rate=0.0005)

model.compile(optimizer=otimizador_ajustado, loss='mse', metrics=['mae'])


#================================ Treinando a Rede Neural =======================================#

history = model.fit(
    X_train, y_train, 
    epochs=60, 
    batch_size=128, 
    validation_split=0.15 
)

# Salvando o modelo treinado
model.save('music_quality_prediction_model.keras')


#================================ Testando o modelo treinado =======================================#

# y com os valores previstos, ou o y^
y_pred = model.predict(X_test)

# MAE (Mean Absolute Error): Diz, em média, o quanto a IA errou para mais ou para menos.
# R^2 Score (Coeficiente de Determinação): Diz a porcentagem de variação dos dados que o modelo consegue explicar 

# Calculando o erro médio absoluto
mae = mean_absolute_error(y_test, y_pred)
# Calculando o R² 
r2 = r2_score(y_test, y_pred)

print(f"Erro Médio Absoluto (MAE): {mae:.4f}")
print(f"Coeficiente R²: {r2:.4f}")

#================================ Plotando gráficos do teste =======================================#

plt.figure(figsize=(14, 5))

# Grafico 1: Evolução da Perda (Loss / MSE) ao longo das epocas
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Erro no Treino (MSE)', color='blue')
plt.plot(history.history['val_loss'], label='Erro na Validação (MSE)', color='orange')
plt.title('Evolução do Erro Durante o Treino')
plt.xlabel('Épocas')
plt.ylabel('Erro Quadrático Médio (MSE)')
plt.legend()
plt.grid(True)

# Grafico 2: Dispersao (Valores Reais vs. Valores Previstos)
plt.subplot(1, 2, 2)
plt.scatter(y_test, y_pred, alpha=0.3, color='purple', label='Músicas de Teste')
# Linha perfeita onde o previsto seria exatamente igual ao real
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', lw=2, linestyle='--', label='Previsão Perfeita')
plt.title('Valores Reais vs. Previsões da IA')
plt.xlabel('Popularidade Real (Escala Log)')
plt.ylabel('Popularidade Prevista (Escala Log)')
plt.legend()
plt.grid(True)

# Exibe os gráficos na tela
plt.tight_layout()
plt.show()