# 🧪 Previsibilidade de doença cardíaca com Machine Learning

 **Definição do Problema**

- **Tema:** Saúde
- **Pergunta de Pesquisa:** "É possível prever a probabilidade de um paciente ter uma doença cardíaca com base em um conjunto de atributos clínicos?"
- **Tipo de Problema:** Classificação Binária (Target: 1 = Possui Doença, 0 = Não Possui Doença)
- **Dataset:** https://www.kaggle.com/datasets/redwankarimsony/heart-disease-data?resource=download

Descrição das colunas:
- **id:** ID único para cada paciente.
- **age:** idade do paciente em anos.
- **sex:** gênero (Masculino -> "Male"; Feminino -> "Female").
- **dataset:** local de coleta de dados.
- **cp:** tipo de dor no peito (angina típica -> "typical angina"; angina atípica -> "atypical angina"; não anginosa -> "non-anginal"; assintomática -> "asymptomatic").
- **trestbps:** pressão arterial em repouso (em mmHg na admissão ao hospital).
- **chol:** medição de colesterol sérico em mg/dl.
- **fbs:** glicemia em jejum (se a glicemia em jejum for superior a 120 mg/dl).
- **restecg:** eletrocardiograma (ECG) em repouso,
  valores -> (normal; anormalidade do segmento ST -> "stt abnormality"; hipertrofia do ventrículo esquerdo -> "lv hypertrophy").
- **thalach:** frequência cardíaca máxima atingida.
- **exang:** angina induzida por exercício (Verdadeiro -> "True"; Falso -> "False")
- **oldpeak:** depressão do segmento ST induzida pelo exercício em relação ao repouso.
- **slope:** a inclinação do segmento ST no pico do exercício.
- **ca:** número de vasos principais (0-3) visualizados por fluoroscopia.
- **thal:** (normal; defeito fixo -> "fixed defect"; defeito reversível -> "reversible defect").
- **num:** o atributo previsto.
---

## ⚙️ Tecnologias Utilizadas

- **Python**
- **Pandas**
- **Scikit-learn**
- **Streamlit**
- **Cursor** (IDE)



---

## 🗂️ Estrutura do Projeto

- **`main.ipynb`**  
  Notebook Jupyter contendo a análise exploratória dos dados, o treinamento e a exportação do modelo. Neste arquivo está contido os passos para a execução do projeto.

- **`app.py`**  
  Aplicação web desenvolvida com Streamlit que carrega o modelo treinado (`modelo_regressao_logistica.joblib`) e permite que o usuário insira as características de um paciente para obter a previsão de doença.

- **`heart_disease_uci.csv`**  
  Arquivo do dataset.

  - **`scaler.joblib`**  
  Arquivo de scaler.

  - **`model_columns.joblib`**  
  Arquivo contendo o nome das colunas.
