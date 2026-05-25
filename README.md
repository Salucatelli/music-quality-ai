# Music Quality & Popularity AI 🎵🧠

Este projeto foi desenvolvido como projeto em uma disciplina de Inteligência Artificial 2. O objetivo principal é investigar, treinar e comparar diferentes arquiteturas de redes neurais artificiais capazes de analisar arquivos de áudio e prever o sucesso/popularidade de obras musicais utilizando o ecossistema do dataset **FMA (Free Music Archive)**.

---

## 🛠️ Arquitetura do Projeto e Fluxo de Arquivos

O repositório está estruturado de forma a separar os scripts de processamento de dados, treinamento das redes e inferência do usuário final:

* **`get_mp3_data.py` / `compute_features`**: Módulo responsável por simular localmente a extração das 343 características acústicas utilizadas do FMA (MFCC, Chroma, Spectral Contrast, etc.) usando a biblioteca `librosa`.
* **Scripts de Treinamento**: Notebooks ou scripts Python que carregam os metadados brutos do FMA (`tracks.csv` e `features.csv`), aplicam transformações logarítmicas ($np.log1p$) na variável alvo (`listens`) para estabilizar o aprendizado de regressão, e treinam as redes.
* **`ai.py`**: O script principal de inferência. Ele carrega as redes neurais salvas, lê uma música de teste (`./songs/teste.mp3`), formata os dados e exibe a estimativa real de plays revertendo a escala logarítmica ($np.expm1$).

---

## 🔬 Abordagens e Modelos Desenvolvidos

O projeto foi dividido em duas abordagens distintas de Aprendizado Profundo:

### 1. Perceptron Multicamadas (MLP) 
Focada em processar dados tabulares em larga escala (utilizando o dataset **FMA Full** com mais de 106.000 músicas), esta rede analisa os 343 metadados estatísticos previamente extraídos do áudio e armazenados no dataset FMA.

* **Topologia da Rede**: Estrutura piramidal profunda com afunilamento gradual (`Dense(512) -> Dense(256) -> Dense(128) -> Dense(64) -> Dense(1)`).
* **Técnicas de Regularização**: Utilização de camadas de **Batch Normalization** após cada ativação ReLU para garantir estabilidade numérica e impedir explosão de gradientes em lotes massivos, combinada com **Dropout (30% e 20%)** para atenuar o *overfitting*.
* **Otimização**: Compilada com a função de perda Erro Quadrático Médio (`MSE`), utilizando o otimizador Adam com uma taxa de aprendizado reduzida (`learning_rate=0.0005`) para garantir passos de ajuste finos e precisos.
* **Batch Size**: Configurado em `128` para estabilizar o direcionamento matemático do gradiente dadas as dimensões da base *Full*.

### 2. Rede Neural Convolucional (CNN) 
Esta abordagem elimina planilhas de metadados humanos e faz a IA "ouvir" o áudio bruto do subset **FMA Small** (8.000 músicas).

* **Processamento de Entrada**: Os arquivos `.mp3` brutos são convertidos usando a transformada de Fourier em **Espectrogramas de Mel** (matrizes de intensidade tempo-frequência), normalizados entre `0` e `1` e tratados como imagens digitais.
* **Topologia Convolucional**: Três blocos alternados de convoluções bidimensionais e subamostragem (`Conv2D` de 32 e 64 filtros com tamanho $3 \times 3$ + `MaxPooling2D` de $2 \times 2$). As convoluções extraem características fluídas e tímbricas diretamente do espectrograma, enquanto o pooling garante invariância espacial.
* **Camada de Decisão**: A matriz resultante é achatada (`Flatten`), repassada por uma camada densa de 64 neurônios com Dropout (30%) e finalizada em uma saída linear de regressão.

---

## 📊 Resultados Obtidos 

Ao avaliar os modelos através das métricas de **Erro Médio Absoluto (MAE)** e **Coeficiente de Determinação ($R^2$)**, o projeto gerou uma valiosa discussão teórica para a área de recuperação de informação musical (MIR):

1.  **O Limite do Áudio**: Ambos os modelos obtiveram desempenhos estáveis de MAE em escala logarítmica (oscilando próximo de `1.03`).
2.  **Fenômeno da Média Amostral**: Os gráficos de dispersão revelaram que, diante de características estritamente acústicas, as redes neurais tendem a predizer valores próximos à média geral de acessos da base (centralizados na faixa de 7.8 a 8.2 no log).
3.  **Conclusão**: O projeto provou empiricamente que **a popularidade de uma música não está gravada exclusivamente em suas frequências sonoras**. Faixas acusticamente idênticas possuem desempenhos de mercado caóticos e distintos no mundo real devido a fatores externos invisíveis ao áudio (relevância do artista, algoritmos de recomendação das plataformas, investimentos em marketing, contexto temporal de lançamento, etc.).

---

## 🚀 Como Executar o Projeto

### Pré-requisitos
Certifique-se de ter o Python 3.10+ e as dependências instaladas. Recomenda-se o uso de um ambiente virtual (`venv`):

```bash
python -m venv venv
./venv/Scripts/activate  # No Windows
pip install tensorflow numpy pandas librosa scikit-learn joblib matplotlib
```

### Instalação do Dataset (Para Treinamento)
* Baixe o arquivo de tabelas fma_metadata.zip e a pasta de áudios brutos `fma_small.zip` direto do repositório oficial do Free Music Archive.

* Descompacte-os e configure os caminhos de diretório (`PATH_METADATA` e `PATH_AUDIO`) nos scripts de treino.

### Rodando a Inferência em Tempo Real
* Para submeter uma música de sua preferência à análise da Inteligência Artificial:

* Insira um arquivo de áudio no formato MP3 dentro da pasta `./songs/` e execute o arquivo principal de inferência no terminal:

```bash
python ai.py
```
