from tensorflow.keras.models import load_model
from get_mp3_data import compute_features
import numpy as np
import joblib


# Carrega o modelo da IA
meu_modelo_treinado = load_model('music_quality_prediction_model.keras')
scaler = joblib.load("scaler.joblib")

# 1. Pegar as características
X_raw = compute_features('./songs/teste.mp3')

# Verificação de segurança 1: Tamanho do vetor
print(f"SHAPE ORIGINAL: {X_raw.shape}") 
# Se você treinou com 343 colunas, aqui PRECISA aparecer (343,)

# 2. Reshape
X_reshaped = X_raw.reshape(1, -1)
print(f"SHAPE APÓS RESHAPE: {X_reshaped.shape}") # Deve ser (1, 343)

# 3. Escalonamento
try:
    X_final = scaler.transform(X_reshaped)

    # Essa limitação dos dados server para evitar que ruídos aleatórios da música quebrem o modelo, levando ele a prever um número infinitamente grande. Essa faixa de -3 a 3 cobre praticamente todos os casos normais de uma música.
    X_final = np.clip(X_final, -3, 3) # Limita os dados entre -3 e 3

    print("MÉDIA DOS DADOS ESCALADOS:", np.mean(X_final)) 
    # Se a média for um número absurdo (tipo 5000), o scaler está errado
except Exception as e:
    print(f"ERRO NO SCALER: {e}")

# 4. Predição
previsao = meu_modelo_treinado.predict(X_final)
print(f"PREVISÃO BRUTA (LOG): {previsao[0][0]}") 

# Verificação de segurança 2: Se a previsão log for > 50, o expm1 vai explodir
if previsao[0][0] > 50:
    print("ERRO: O modelo previu um valor logarítmico impossível. Verifique se as colunas estão na ordem certa.")
else:
    popularidade_real = np.expm1(previsao)
    resultado_final = int(max(0, popularidade_real[0][0]))
    print(f"Popularidade estimada: {resultado_final} plays")


"""
# Carrega o modelo da IA
meu_modelo_treinado = load_model('music_quality_prediction_model.keras')
scaler = joblib.load("scaler.joblib")

# Pegar as características de uma música atravéz de um arquivo mp3
X_novo = compute_features('./songs/teste.mp3')

# 1. Ajustar o formato para o modelo (o modelo espera uma matriz, não um vetor simples)
X_novo = X_novo.reshape(1, -1)

X_novo_escalado = scaler.transform(X_novo)

# Previsão de popularidade dessa música
previsao = meu_modelo_treinado.predict(X_novo_escalado)

# 4. Reverter o Logaritmo (Inversa de log1p é expm1)
# Isso transforma o valor de "escala de 0 a 10" de volta para "0 a 1.000.000"
popularidade_real = np.expm1(previsao)

# 5. Pegar o número limpo e arredondar
resultado_final = int(max(0, popularidade_real[0][0]))

print(f"--- Resultado da Análise ---")
print(f"Popularidade estimada: {resultado_final} ouvintes/plays")"""