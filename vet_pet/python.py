import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import binom, poisson
import numpy as np 

#configuração da pagina
st.set_page_config(page_title="Painel de atendimento médico", layout="wide") 

# Função para carregar os dados
@st.cache_data
def carregar_dados():
    df = pd.read_csv("atendimentos.csv", sep=';', encoding='latin-1')
    df.columns = df.columns.str.strip()
    return df

# Carregar os dados
df = carregar_dados()

# Titulo
st.title("Painel de atendimento médico")

# Cards de métricas
media_idade = df["idade"].mean()
total_atestados = df[df["atestado"] == 1]. shape[0]
total_respiratorio = df[df["sindrespiratoria"] == 1]. shape[0]

st.markdown("### Resumo dos atendimentos")
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Média de idade", f"{media_idade:.1f} anos")
with col2:
    st.metric("Atestados emitidos", total_atestados)
with col3:
    st.metric("casos respiratorios", total_respiratorio)

st.divider()

#linha de gráficos
with st.container():
    col_graf1, col_graf2 = st.columns(2)

    with col_graf1:
        st.markdown("atendimentos por médicos")
        fig1, ax1 = plt.subplots(figsize=(3,5, 2.5))
        sns.countplot(data=df, x="medico", ax=ax1, palette="coolwarm")
        ax1.set_xlabel("")
        ax1.set_xlabel("Qtd")
        plt.xticks(rotation=45)
        st.pyplot(fig1)

    with col_graf2:
        st.markdown("atendimentos por Turno")
        fig2, ax2 = plt.subplots(figsize=(3,5, 2.5))
        sns.countplot(data=df, x="Turno", order=df["Turno"].value_counts().index, ax=ax2, palette="viridis")
        ax2.set_xlabel("")
        ax2.set_ylabel("Qtd")
        st.pyplot(fig2)

# Segunda linha de gráficos
with st.container():
    col_graf3, col_graf4 = st.columns(2)

    with col_graf3:
        st.markdown("Casos respiratórios por idade")
        respiratorio_df = df[df["Sindrespiratoria"] == 1]
        fig3, ax3 = plt.subplots(figsize=(3.5, 2.5))
        sns.histplot(respiratorio_df["idade"], bins=10, kde=True, color="purple", ax=ax3)
        ax3.set_xlabel("idade")
        ax3.set_ylabel("Casos")
        st.pyplot(fig3)
    
    with col_graf4:
        st.markdown("distribuição por gênero")
        fig4, ax4 = plt.subplots(figsize=(3.5, 2.5))
        sns.countplot(data=df, x="Gênero", ax=ax4, palette="pastel")
        ax4.set_xlabel("")
        ax4.set_ylabel("Qtd")
        st.pyplot(fig4)

st.divider()

# Exportar CVS
st.markdown("### exportar dados")
csv = df.to_csv(index=False, sep=';', encoding='utf-8-sig').encode('utf-8-sig')
st.download_button(
    label="Baixar CSV",
    data=csv,
    file_name='atendimentos_export.csv',
    mime='text/csv',
)

st.divider()

# Análises estatisticas interativas
st.markdown("### Análises Estatisticas (Distribuições)")

# Binomial - probabilidade de atestados
st.markdown("### Probabilidade de atestados (Distribuição Binomial)")
p_atestado = df["Atestado"].mean()

col_a, col_b = st.columns (2)
with col_a:
    n = st.slider("Número de pacientes simulados", min_value=5, max_value=50, value=10, step=1)
with col_b:
    k = st.slider("Número de atestados desejados (ou mais)", min_value=1, max_value=50, value=5, step=1)

if k > n:
    st.error("O número de atestados desejados não pode ser maior que oo número de pacientes.")
else:
    prob_5oumais = 1 - binom.cdf(k - 1, n, p_atestado)
    st.write(f"Com base em uma taxa observadora de {p_atestado:.1%} de emissão de atestados. ")
    st.write(f"a probabilidade de pelo menos {k} atestados em {n} pacientes é **{prob_5oumais:.2%}**.")

    # Gráfico da distribuição binomial
    probs_binum = [binom.pmf(i,n, p_atestado) for i in range(n+1)]
    fig_b, ax_b = plt.subplots(figsize=(5, 3))
    bars = ax_b.bar(range(n+1), probs_binom, color=["gray" if i < k else "orange" for i in range(n+1)])
    ax_b.set_xlabel("Numero de atestados")
    ax_b.set_ylabel("Probabilidade")
    ax_b.set_title("Distribuição Binomial")
    st.pyplot(fig_b)

st.divider()

#POISSON - probabilidade de casos respiratorios
st.markdown("### casos Respiratórios por Turno (distribuição de Poisson)")
casos_por_turno = df.groupby("Turno")["Sindrespiratoria"].sum().mean()

k_poisson = st.slider("Número de casos respiratorios desejados (ou mais)", min_value=0, max_value=10, value=3, step=1)
prob_3oumais = 1 - poisson.cdf(k_poisson - 1, casos_por_turno)

st.write(f"A media de casos respiratorios por turno é **{casos_por_turno:.2f}**.")
st.write(f"A probabilidade de pelo menos {k_poisson} casos em um turno é **{prob_3oumais:.2%}**.")

# Grafico da distribuição de poisson
max_k = 10
probs_poisson = [poisson.pmf(i, casos_por_turno) for i in range(max_k+1)]
fig_p, ax_p = plt.subplots(figsize=(5,3))
bars_p = ax_p.bar(range(max_k+1), probs_poisson, color=["gray" if i < k_poisson else "orange" for i in range(max_k+1)])
ax_p.set_xlabel("Numero de casos")
ax_p.set_ylabel("Probabilidade")
ax_p.set_title("distribuição de poisson")
st.pyplot(fig_p)