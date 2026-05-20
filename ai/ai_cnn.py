import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import librosa
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from tensorflow.keras import models, layers, optimizers

# ==============================================================================
# 1. FUNÇÃO PARA TRANSFORMAR UM MP3 EM IMAGEM (ESPECTROGRAMA)
# ==============================================================================
def mp3_para_espectrograma(caminho_mp3, n_mels=128, max_time_steps=128):
    try:
        # Carrega os primeiros 30 segundos do áudio 
        y, sr = librosa.load(caminho_mp3, duration=30, sr=22050)
        
        # Calcula o Espectrograma de Mel
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
        
        # Converte para a escala de Decibéis 
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Força todas as imagens a terem o mesmo tamanho horizontal (tempo)
        if mel_spec_db.shape[1] < max_time_steps:
            # Se for curta, preenche com zeros 
            pad_width = max_time_steps - mel_spec_db.shape[1]
            mel_spec_db = np.pad(mel_spec_db, ((0, 0), (0, pad_width)), mode='constant')
        else:
            # Se for longa, corta no tamanho máximo
            mel_spec_db = mel_spec_db[:, :max_time_steps]
            
        return mel_spec_db
    except Exception as e:
        # Se o MP3 estiver corrompido, retorna None
        print(f"Erro ao processar {caminho_mp3}: {e}")
        return None

# ==============================================================================
# 2. CARREGAMENTO DOS DADOS E MAPEAMENTO DOS DIRETÓRIOS
# ==============================================================================

PATH_METADATA = os.getcwd() + r"\dataset"
PATH_AUDIO = os.getcwd() + r"\dataset\fma_small"

print("Carregando planilhas...")
tracks = pd.read_csv(os.path.join(PATH_METADATA, 'tracks.csv'), index_col=0, header=[0, 1])

# Filtrando apenas o small
subset_mask = tracks[('set', 'subset')] == 'small'
tracks_small = tracks.loc[subset_mask]

X_lista = []
y_lista = []

print("Iniciando a conversão dos MP3s em imagens (Espectrogramas)...")

for track_id, row in tracks_small.iterrows():
    # Descobre o nome da pasta com base no ID (id 1234 -> pasta '001' arquivo '001234.mp3')
    tid_str = '{:06d}'.format(track_id)
    pasta = tid_str[:3]
    caminho_musica = os.path.join(PATH_AUDIO, pasta, f"{tid_str}.mp3")
    
    if os.path.exists(caminho_musica):
        espectrograma = mp3_para_espectrograma(caminho_musica)
        
        if espectrograma is not None:
            X_lista.append(espectrograma)
            y_lista.append(np.log1p(row[('track', 'listens')]))

X = np.array(X_lista)
y = np.array(y_lista)

# O Keras precisa de uma dimensão extra para indicar o canal de cor (1 = Escala de Cinza)
X = X.reshape(X.shape[0], X.shape[1], X.shape[2], 1)

# Normalizando os pixels da "imagem" entre 0 e 1 
X = (X - X.min()) / (X.max() - X.min())

print(f"Processamento concluído! Formato da nossa matriz de imagens: {X.shape}")

# Divisão de Treino e Teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ==============================================================================
# 3. CRIANDO A ARQUITETURA DA CNN (REDE NEURAL CONVOLUCIONAL)
# ==============================================================================
model = models.Sequential([

    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(X.shape[1], X.shape[2], 1)),
    layers.MaxPooling2D((2, 2)),
    
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    

    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    
    layers.Flatten(),
    
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.3),
    
    layers.Dense(1, activation='linear')
])

# Compilaçao focada em regressao
model.compile(optimizer=optimizers.Adam(learning_rate=0.001), loss='mse', metrics=['mae'])

# ==============================================================================
# 4. TREINAMENTO
# ==============================================================================
print("Treinando a CNN...")
history = model.fit(X_train, y_train, epochs=30, batch_size=32, validation_split=0.1)

# Salva o novo modelo de audio 
model.save('modelo_cnn_audio_bruto.keras')
print("Modelo salvo com sucesso!")


#================================ Testando o modelo treinado =======================================#

# y com os valores previstos, ou o y^
y_pred = model.predict(X_test)

# transformando em um vetor simples
y_pred = y_pred.flatten()

mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Erro Médio Absoluto (MAE): {mae:.4f}")
print(f"Coeficiente R²: {r2:.4f}")


# ==============================================================================
# 6. PLOTAGEM DOS GRÁFICOS
# ==============================================================================

plt.figure(figsize=(14, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Erro no Treino (MSE)', color='blue')
plt.plot(history.history['val_loss'], label='Erro na Validação (MSE)', color='orange')
plt.title('Evolução do Erro Durante o Treino')
plt.xlabel('Épocas')
plt.ylabel('Erro Quadrático Médio (MSE)')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.scatter(y_test, y_pred, alpha=0.3, color='purple', label='Músicas de Teste')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', lw=2, linestyle='--', label='Previsão Perfeita')
plt.title('Valores Reais vs. Previsões da IA')
plt.xlabel('Popularidade Real (Escala Log)')
plt.ylabel('Popularidade Prevista (Escala Log)')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()