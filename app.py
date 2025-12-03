import streamlit as st
import pandas as pd
import numpy as np
import joblib

# --- CARREGAMENTO DOS MODELOS E DADOS ---
# Usar st.cache_resource para garantir que os arquivos sejam carregados apenas uma vez

@st.cache_resource
def load_artifacts():
    """Carrega o modelo, o scaler e a lista de colunas."""
    model = joblib.load('modelo_regressao_logistica.joblib')
    scaler = joblib.load('scaler.joblib')
    model_columns = joblib.load('model_columns.joblib')
    return model, scaler, model_columns

model, scaler, model_columns = load_artifacts()


# --- INTERFACE DO USUÁRIO (INPUTS) ---

st.title("Previsão de Doenças Cardíacas")
st.markdown("Insira os dados do paciente para obter uma previsão sobre o risco de doença cardíaca, baseada em um modelo de Regressão Logística com 85% de acurácia.")

col1, col2, col3 = st.columns(3)

with col1:
    age = st.slider("Idade", 20, 80, 50)
    sex = st.selectbox("Sexo", ("Masculino", "Feminino"))
    cp = st.selectbox("Tipo de Dor no Peito (cp)", (0, 1, 2, 3), help="0: típico, 1: atípico, 2: não anginoso, 3: assintomático")
    trestbps = st.slider("Pressão Arterial em Repouso (trestbps)", 90, 200, 120)

with col2:
    chol = st.slider("Colesterol Sérico (chol)", 100, 600, 200)
    fbs = st.selectbox("Glicemia de Jejum > 120 mg/dl (fbs)", ("Sim", "Não"))
    restecg = st.selectbox("Eletrocardiograma em Repouso (restecg)", (0, 1, 2))
    thalach = st.slider("Frequência Cardíaca Máxima (thalach)", 70, 220, 150)

with col3:
    exang = st.selectbox("Angina Induzida por Exercício (exang)", ("Sim", "Não"))
    oldpeak = st.slider("Depressão do Segmento ST (oldpeak)", 0.0, 6.2, 1.0)
    slope = st.selectbox("Inclinação do Segmento ST (slope)", (0, 1, 2))
    thal = st.selectbox("Resultado do Teste de Talassemia (thal)", (0, 1, 2, 3))


# --- BOTÃO E LÓGICA DE PREDIÇÃO ---

if st.button("Fazer Previsão"):
    
    # 1. Coletar os dados de input em um dicionário
    user_input = {
        'age': age,
        'sex': 1 if sex == 'Masculino' else 0,
        'trestbps': trestbps,
        'chol': chol,
        'fbs': 1 if fbs == 'Sim' else 0,
        'thalach': thalach,
        'exang': 1 if exang == 'Sim' else 0,
        'oldpeak': oldpeak,
        # Adicionar as colunas categóricas que serão transformadas em dummies
        'cp': cp,
        'restecg': restecg,
        'slope': slope,
        'thal': thal
    }
    
    # 2. Converter o dicionário para um DataFrame do Pandas
    user_df = pd.DataFrame([user_input])
    
    # 3. Aplicar o One-Hot Encoding (get_dummies)
    # Isso transforma colunas como 'cp' em 'cp_1', 'cp_2', etc.
    user_df_processed = pd.get_dummies(user_df, columns=['cp', 'restecg', 'slope', 'thal'])
    
    # 4. Alinhar as colunas com as colunas do modelo
    # Garante que o DataFrame do usuário tenha EXATAMENTE as mesmas colunas que o modelo foi treinado
    # Quaisquer colunas faltantes serão adicionadas com valor 0.
    user_df_aligned = user_df_processed.reindex(columns=model_columns, fill_value=0)
    
    # 5. Aplicar a normalização (scaling)
    # Usamos o mesmo scaler que foi treinado com os dados do notebook
    user_df_scaled = scaler.transform(user_df_aligned)
    
    # 6. Fazer a predição e obter a probabilidade
    prediction = model.predict(user_df_scaled)
    probability = model.predict_proba(user_df_scaled)

    # 7. Exibir o resultado
    st.subheader("Resultado da Previsão:")
    if prediction[0] == 1:
        st.error("ALTO RISCO de Doença Cardíaca")
        st.write(f"Probabilidade de ter a doença: {probability[0][1]*100:.2f}%")
    else:
        st.success("BAIXO RISCO de Doença Cardíaca")
        st.write(f"Probabilidade de não ter a doença: {probability[0][0]*100:.2f}%")