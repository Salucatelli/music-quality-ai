# music-quality-ai

## Visão Geral

Este repositório apresenta um projeto de Inteligência Artificial focado na avaliação da qualidade musical. Utilizando uma Rede Neural Convolucional (CNN) e o dataset Free Music Archive (FMA), o objetivo é desenvolver um modelo capaz de prever a popularidade ou a qualidade percebida de uma música com base em suas características numéricas.

## Estrutura do Projeto

O projeto está organizado da seguinte forma:

*   `ai/`:
    *   `ai_model.py`: Contém a definição, treinamento e avaliação da Rede Neural Convolucional (CNN).
    *   `ai.py`: Demonstra como carregar o modelo treinado e utilizá-lo para fazer previsões de qualidade musical.
    *   `requirements.txt`: Lista todas as dependências Python necessárias para executar o projeto.

## Dataset

O projeto utiliza o **Free Music Archive (FMA)**, um dataset abrangente de músicas com metadados ricos, incluindo características numéricas (como MFCCs, Tom, Ritmo, etc.) e informações sobre popularidade/qualidade. Essas características são usadas como entrada para a CNN, enquanto a popularidade serve como o rótulo de qualidade a ser previsto.

Comecei com o dataset small do FMA, que possui 8.000 músicas e 8 gêneros, sendo 100% equilibrado, com 1.000 músicas para cada gênero. 

Agora pretendo testar o dataset Large, que contem 106.574 músicas e 161 gêneros, mas ele não é 100% equilibrado, podendo causar um "vício" na IA dependendo do estilo da música prevista. Porém, esse dataset desequilibrado representa melhor a realidade, com gêneros possuindo muito mais músicas que outros.

## Tecnologias Utilizadas

As principais tecnologias e bibliotecas empregadas neste projeto incluem:

*   **Python**
*   **TensorFlow/Keras**: Para a construção e treinamento da Rede Neural Convolucional.
*   **scikit-learn**: Para pré-processamento de dados e divisão em conjuntos de treino/teste.
*   **Pandas**: Para manipulação e análise de dados.
*   **NumPy**: Para operações numéricas eficientes.
*   **Matplotlib**: Para visualização de dados (implícito no `ai_model.py` para gráficos de treinamento).

## Como Usar

Para configurar e executar este projeto localmente, siga os passos abaixo:

### 1. Download do Dataset FMA Metadata

O modelo utiliza metadados do dataset Free Music Archive (FMA). Como esses arquivos não estão incluídos no repositório (devido ao tamanho e à licença), você precisará baixá-los separadamente.

1.  **Baixe o arquivo `fma_metadata.zip`** do link oficial: [https://os.unil.cloud.switch.ch/fma/fma_metadata.zip](https://os.unil.cloud.switch.ch/fma/fma_metadata.zip).
2.  **Descompacte o arquivo** `fma_metadata.zip` na raiz do seu projeto. Isso criará uma pasta `fma_metadata` contendo os arquivos `tracks.csv`, `features.csv`, entre outros, que são essenciais para o funcionamento do modelo.

### 2. Clonar o Repositório

```bash
git clone https://github.com/Salucatelli/music-quality-ai.git
cd music-quality-ai
```

### 3. Instalar Dependências

Certifique-se de ter o `pip` instalado. Em seguida, instale as dependências listadas no `requirements.txt`:

```bash
pip install -r ai/requirements.txt
```

### 4. Treinar o Modelo (Opcional)

O arquivo `ai/ai_model.py` contém o código para treinar a CNN. Se você deseja treinar o modelo do zero ou com dados atualizados, execute:

```bash
python ai/ai_model.py
```

Este script irá carregar os dados do FMA, pré-processá-los, treinar a CNN e salvar o modelo treinado como `modelo_qualidade_musica.keras`.

### 5. Fazer Previsões

Para utilizar o modelo treinado e prever a qualidade de novas músicas, execute o script `ai/ai.py`:

```bash
python ai/ai.py
```

Este script carrega o modelo salvo e demonstra como fazer previsões. Você precisará fornecer os dados de entrada (`X_novo`) para as músicas que deseja avaliar.

## Métricas de Avaliação

O desempenho do modelo é avaliado utilizando as seguintes métricas:

*   **MAE (Mean Absolute Error)**: Indica, em média, o quanto a IA errou para mais ou para menos na previsão da qualidade musical.
*   **R^2 Score (Coeficiente de Determinação)**: Representa a porcentagem da variação dos dados que o modelo consegue explicar. Um valor mais próximo de 1 indica um ajuste melhor do modelo aos dados.
