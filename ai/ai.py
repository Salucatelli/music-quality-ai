from tensorflow.keras.models import load_model
from get_mp3_data import compute_features
import numpy as np
import joblib


# Carrega o modelo da IA
meu_modelo_treinado = load_model('music_quality_prediction_model.keras')
scaler = joblib.load("scaler.joblib")

# 1. Pegar as características
X_raw = compute_features('./songs/teste.mp3')

print(f"SHAPE ORIGINAL: {X_raw.shape}") 

# 2. Reshape
X_reshaped = X_raw.reshape(1, -1)
print(f"SHAPE APÓS RESHAPE: {X_reshaped.shape}") 

# 3. Escalonamento
try:
    X_final = scaler.transform(X_reshaped)

    # Essa limitação dos dados server para evitar que ruídos aleatórios da música quebrem o modelo, levando ele a prever um número infinitamente grande. Essa faixa de -3 a 3 cobre praticamente todos os casos normais de uma música.
    X_final = np.clip(X_final, -3, 3) 

    print("MÉDIA DOS DADOS ESCALADOS:", np.mean(X_final)) 
    # Se a média for um número absurdo (tipo 5000), o scaler está errado
except Exception as e:
    print(f"ERRO NO SCALER: {e}")

# 4. Predição
previsao = meu_modelo_treinado.predict(X_final)
print(f"PREVISÃO BRUTA (LOG): {previsao[0][0]}") 

# Se a previsão log for > 50, o expm1 vai explodir
if previsao[0][0] > 50:
    print("ERRO: O modelo previu um valor logarítmico impossível. Verifique se as colunas estão na ordem certa.")
else:
    popularidade_real = np.expm1(previsao)
    resultado_final = int(max(0, popularidade_real[0][0]))
    print(f"Popularidade estimada: {resultado_final} plays")

