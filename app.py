import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np

# ---------------------------
# CONFIG GERAL
# ---------------------------

st.set_page_config(page_title="Trading Rooms IA", layout="wide")

st.title("🚀 Trading Rooms IA – Atlas Lite para ActivTrader")
st.write(
    "Use esta versão enxuta do Atlas para ver sinais de Day, Swing e Position "
    "em qualquer ativo compatível com Yahoo Finance (Forex, índices, commodities, ações, cripto)."
)

default_tickers = (
    "EURUSD=X, GBPUSD=X, USDJPY=X, XAUUSD=X, SPX, IBOV, PETR4.SA, VALE3.SA, BTC-USD"
)

tickers_input = st.text_input(
    "Lista de ativos (separados por vírgula)",
    value=default_tickers,
    help="Use os códigos do Yahoo Finance, separados por vírgula.",
)

tickers = [t.strip() for t in tickers_input.split(",") if t.strip()]

# ---------------------------
# FUNÇÕES DE INDICADORES
# ---------------------------

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """RSI simples, baseado em ganhos/perdas médios.[web:257][web:284]"""
    delta = series.diff()
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)

    roll_up = pd.Series(gain, index=series.index).rolling(period).mean()
    roll_down = pd.Series(loss, index=series.index).rolling(period).mean()

    rs = roll_up / (roll_down.replace(0, np.nan))
    rsi_val = 100.0 - (100.0 / (1.0 + rs))
    return rsi_val


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average True Range, medida clássica de volatilidade.[web:284][web:289]"""
    high = df["High"]
    low = df["Low"]
    close = df["Close"]
    prev_close = close.shift(1)

    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()

    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_val = tr.rolling(period).mean()
    return atr_val


@st.cache_data(show_spinner=False)
def baixar_dados(tickers_list, period, interval="1d"):
    """
    Baixa OHLC para múltiplos tickers via yfinance.[web:258][web:282]
    Retorna dict {ticker: DataFrame}
    """
    if not tickers_list:
        return {}

    try:
        data = yf.download(
            tickers_list,
            period=period,
            interval=interval,
            group_by="ticker",
            auto_adjust=False,
            threads=True,
            progress=False,
        )
    except Exception:
        return {}

    result = {}
    for ticker in tickers_list:
        try:
            if isinstance(data.columns, pd.MultiIndex):
                df = data[ticker].copy()
            else:
                df = data.copy()
            df = df.dropna()
            if not df.empty:
                result[ticker] = df
        except Exception:
            continue

    return result


def calcular_contexto(df: pd.DataFrame, ema_fast_win=9, ema_slow_win=21):
    """
    Calcula contexto básico: EMAs, RSI, ATR, retorno e volatilidade percentual.
    """
    closes = df["Close"]
    ema_fast = closes.ewm(span=ema_fast_win, adjust=False).mean()
    ema_slow = closes.ewm(span=ema_slow_win, adjust=False).mean()
    rsi_val = rsi(closes, 14)
    atr_val = atr(df, 14)

    # Retorno do período
    if len(closes) >= 2:
        ret_total = (closes.iloc[-1] / closes.iloc[0] - 1) * 100.0
    else:
        ret_total = np.nan

    # Volatilidade: desvio padrão dos retornos diários (anualizado ~252 dias)
    rets = closes.pct_change().dropna()
    if len(rets) >= 2:
        vol_pct = rets.std() * np.sqrt(252) * 100.0
    else:
        vol_pct = np.nan

    return ema_fast, ema_slow, rsi_val, atr_val, ret_total, vol_pct


def gerar_sinal(final_close, ema_fast_last, ema_slow_last, rsi_last, atr_last,
                ret_total, vol_pct, tipo_sala: str):
    """
    Regras objetivas para direção, score e confiança.
    """
    if any(np.isnan(x) for x in [final_close, ema_fast_last, ema_slow_last, rsi_last, ret_total]):
        return "NEUTRO", 0, "baixa", "Dados insuficientes."

    score = 50
    narrativa = []

    # Tendência por EMAs
    if ema_fast_last > ema_slow_last:
        score += 15
        narrativa.append("Tendência de alta (EMA curta acima da longa).")
        tendencia = "alta"
    elif ema_fast_last < ema_slow_last:
        score -= 15
        narrativa.append("Tendência de baixa (EMA curta abaixo da longa).")
        tendencia = "baixa"
    else:
        narrativa.append("EMAs sem direção clara.")
        tendencia = "neutra"

    # Retorno do período
    if ret_total > 0:
        score += min(ret_total / 2, 15)
        narrativa.append(f"Retorno positivo de {ret_total:.2f}% no período.")
    else:
        score += max(ret_total / 2, -15)
        narrativa.append(f"Retorno negativo de {ret_total:.2f}% no período.")

    # RSI – evitar extremos muito fortes
    if rsi_last > 70:
        score -= 10
        narrativa.append("RSI em sobrecompra (acima de 70).")
    elif rsi_last < 30:
        score += 10
        narrativa.append("RSI em sobrevenda (abaixo de 30).")
    else:
        narrativa.append("RSI em zona neutra.")

    # Volatilidade / ATR – penaliza volatilidade extremamente alta
    if not np.isnan(vol_pct) and vol_pct > 100:
        score -= 10
        narrativa.append("Volatilidade muito alta, risco elevado.")
    elif not np.isnan(vol_pct):
        narrativa.append(f"Volatilidade anualizada aprox.: {vol_pct:.1f}%.")

    # Ajustes por sala
    if tipo_sala == "day":
        narrativa.append("Sala Day: leitura voltada para movimentos de 1–2 dias.")
    elif tipo_sala == "swing":
        narrativa.append("Sala Swing: leitura voltada para movimentos de alguns dias.")
    else:
        narrativa.append("Sala Position: leitura voltada para tendência mais longa.")

    # Clamp do score
    score = int(max(0, min(100, score)))

    # Direção
    if score >= 60 and tendencia == "alta":
        direction = "COMPRA"
    elif score <= 40 and tendencia == "baixa":
        direction = "VENDA"
    else:
        direction = "NEUTRO"

    # Confiança
    if score >= 80 or score <= 20:
        confidence = "alta"
    elif 60 <= score < 80 or 20 < score <= 40:
        confidence = "média"
    else:
        confidence = "baixa"

    return direction, score, confidence, " ".join(narrativa)


def rodar_sala(tipo_sala: str):
    """
    Roda o motor para todos os ativos, de acordo com a sala.
    """
    if tipo_sala == "day":
        period = "5d"      # mais dias para dar contexto
    elif tipo_sala == "swing":
        period = "1mo"
    else:
        period = "3mo"

    dados = baixar_dados(tickers, period=period, interval="1d")

    resultados = []
    narrativas = {}
    series_precos = {}

    for ticker, df in dados.items():
        ema_fast, ema_slow, rsi_val, atr_val, ret_total, vol_pct = calcular_contexto(df)

        final_close = df["Close"].iloc[-1]
        ema_fast_last = ema_fast.iloc[-1]
        ema_slow_last = ema_slow.iloc[-1]
        rsi_last = rsi_val.iloc[-1]
        atr_last = atr_val.iloc[-1] if not atr_val.isna().all() else np.nan

        direction, score, conf, texto = gerar_sinal(
            final_close,
            ema_fast_last,
            ema_slow_last,
            rsi_last,
            atr_last,
            ret_total,
            vol_pct,
            tipo_sala,
        )

        narrativas[ticker] = texto
        series_precos[ticker] = df["Close"].copy()

        resultados.append(
            {
                "Ativo": ticker,
                "Direção": direction,
                "Score": score,
                "Retorno (%)": round(ret_total, 2) if not np.isnan(ret_total) else None,
                "RSI": round(rsi_last, 1) if not np.isnan(rsi_last) else None,
                "Confiança": conf,
            }
        )

    if not resultados:
        st.warning("Não foi possível calcular sinais com os dados atuais. Verifique a lista de ativos.")
        return None, None, None

    df_rank = pd.DataFrame(resultados)
    ordem_direcao = {"COMPRA": 0, "VENDA": 1, "NEUTRO": 2}
    df_rank["ordem_direcao"] = df_rank["Direção"].map(ordem_direcao)
    df_rank = df_rank.sort_values(by=["ordem_direcao", "Score"], ascending=[True, False]).drop(
        columns=["ordem_direcao"]
    )

    return df_rank, narrativas, series_precos


def mostrar_xray(ativo: str, preco_series: pd.Series, narrativa: str):
    st.markdown(f"### Raio-X rápido – {ativo}")
    st.line_chart(preco_series)
    st.markdown("**Comentário Atlas Lite:**")
    st.write(narrativa)


# ---------------------------
# LAYOUT DAS SALAS
# ---------------------------

tab_day, tab_swing, tab_position = st.tabs(["Day Trade", "Swing Trade", "Position Trade"])

with tab_day:
    st.subheader("Day Trade – leitura de curtíssimo prazo")
    if st.button("Atualizar ranking - Day"):
        df_day, narr_day, prices_day = rodar_sala("day")
        if df_day is not None:
            st.dataframe(df_day, use_container_width=True)
            ativo_sel = st.selectbox(
                "Escolha um ativo para ver o Raio-X (Day)",
                df_day["Ativo"].tolist(),
                key="day_select",
            )
            if ativo_sel in narr_day and ativo_sel in prices_day:
                mostrar_xray(ativo_sel, prices_day[ativo_sel], narr_day[ativo_sel])

with tab_swing:
    st.subheader("Swing Trade – movimentos de alguns dias")
    if st.button("Atualizar ranking - Swing"):
        df_sw, narr_sw, prices_sw = rodar_sala("swing")
        if df_sw is not None:
            st.dataframe(df_sw, use_container_width=True)
            ativo_sel = st.selectbox(
                "Escolha um ativo para ver o Raio-X (Swing)",
                df_sw["Ativo"].tolist(),
                key="swing_select",
            )
            if ativo_sel in narr_sw and ativo_sel in prices_sw:
                mostrar_xray(ativo_sel, prices_sw[ativo_sel], narr_sw[ativo_sel])

with tab_position:
    st.subheader("Position Trade – tendências mais longas")
    if st.button("Atualizar ranking - Position"):
        df_pos, narr_pos, prices_pos = rodar_sala("position")
        if df_pos is not None:
            st.dataframe(df_pos, use_container_width=True)
            ativo_sel = st.selectbox(
                "Escolha um ativo para ver o Raio-X (Position)",
                df_pos["Ativo"].tolist(),
                key="position_select",
            )
            if ativo_sel in narr_pos and ativo_sel in prices_pos:
                mostrar_xray(ativo_sel, prices_pos[ativo_sel], narr_pos[ativo_sel])
