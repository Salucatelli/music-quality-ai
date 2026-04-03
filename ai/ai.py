from tensorflow.keras.models import load_model

# Recarrega o cérebro da IA prontinho para usar
meu_modelo_treinado = load_model('modelo_qualidade_musica.keras')

# Dados de uma música para prever sua populridade
X_novo = []

# Agora você pode dar o X de uma música NOVA e ela vai prever a nota!
previsao = meu_modelo_treinado.predict(X_novo)