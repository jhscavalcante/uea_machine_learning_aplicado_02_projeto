# 🧪 Previsibilidade de doença cardíaca com Machine Learning

 **Definição do Problema**

- **Tema:** Saúde
- **Pergunta de Pesquisa:** "É possível prever a probabilidade de um paciente ter uma doença cardíaca com base em um conjunto de atributos clínicos?"
- **Tipo de Problema:** Classificação Binária (Target: 1 = Possui Doença, 0 = Não Possui Doença)
- **Dataset:** https://www.kaggle.com/datasets/redwankarimsony/heart-disease-data?resource=download

---

## ⚙️ Tecnologias Utilizadas

- **Python**
- **Pandas**
- **Scikit-learn**
- **Streamlit**

---

## 🗂️ Estrutura do Projeto

- **`main.ipynb`**  
  Notebook Jupyter contendo a análise exploratória dos dados, o treinamento e a exportação do modelo.

- **`app.py`**  
  Aplicação web desenvolvida com Streamlit que carrega o modelo treinado (`modelo_regressao_logistica.joblib`) e permite que o usuário insira as características de um paciente para obter a previsão de doença.

- **`heart_disease_uci.csv`**  
  Arquivo do dataset.

  - **`scaler.joblib`**  
  Arquivo de scaler.

  - **`model_columns.joblib`**  
  Arquivo contendo o nome das colunas.
