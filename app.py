import streamlit as st
import yfinance as yf
import pandas as pd

st.set_page_config(page_title="Trading Rooms IA", layout="wide")

st.title("🚀 Trading Rooms IA – Salas de Negociação")

st.write("Escolha a sala e informe os ativos (código separado por vírgula, ex.: PETR4.SA, VALE3.SA).")

# Entrada de lista de ativos
tickers_input = st.text_input(
    "Lista de ativos",
    value="PETR4.SA, VALE3.SA, ITUB4.SA",
    help="Use os códigos do Yahoo Finance, separados por vírgula."
)

# Converte string em lista limpa
tickers = [t.strip() for t in tickers_input.split(",") if t.strip()]

tab_day, tab_swing, tab_position = st.tabs(["Day Trade", "Swing Trade", "Position Trade"])

def carregar_dados(periodo: str):
    if not tickers:
        return pd.DataFrame()
    df = yf.download(tickers, period=periodo, interval="1d", group_by="ticker", auto_adjust=True)
    return df

def montar_ranking(df: pd.DataFrame, periodo_label: str):
    if df.empty:
        st.warning("Nenhum dado retornado. Verifique os códigos dos ativos.")
        return

    resultados = []
    for ticker in tickers:
        try:
            hist = df[ticker]["Close"].dropna()
            if len(hist) < 2:
                continue
            retorno = (hist.iloc[-1] / hist.iloc[0] - 1) * 100
            resultados.append({"Ativo": ticker, f"Retorno {periodo_label} (%)": retorno})
        except Exception:
            continue

    if not resultados:
        st.warning("Não foi possível calcular o ranking com os dados atuais.")
        return

    rank_df = pd.DataFrame(resultados).sort_values(by=f"Retorno {periodo_label} (%)", ascending=False)
    st.dataframe(rank_df, use_container_width=True)

with tab_day:
    st.subheader("Day Trade – variação do dia")
    if st.button("Atualizar ranking - Day"):
        dados_day = carregar_dados("1d")
        montar_ranking(dados_day, "dia")

with tab_swing:
    st.subheader("Swing Trade – últimos 5 dias úteis")
    if st.button("Atualizar ranking - Swing"):
        dados_swing = carregar_dados("5d")
        montar_ranking(dados_swing, "5 dias")

with tab_position:
    st.subheader("Position Trade – último mês")
    if st.button("Atualizar ranking - Position"):
        dados_pos = carregar_dados("1mo")
        montar_ranking(dados_pos, "1 mês")
