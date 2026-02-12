import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from streamlit_autorefresh import st_autorefresh

# ============================
# AUTO-REFRESH (30 s)
# ============================

st_autorefresh(interval=30000, key="atlas_autorefresh")

# ============================
# LISTAS RESUMIDAS (PRINCIPAIS)
# ============================

EXCEL_STOCKS_LIST = [
    # Big Tech / EUA
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA",
    # Bancos / Financeiro
    "JPM", "BAC", "GS", "MS",
    # Consumo / Marcas fortes
    "KO", "PEP", "MCD", "DIS", "NKE", "SBUX",
    # Saúde / Farma
    "JNJ", "PFE", "LLY", "MRNA",
    # Energia / Materiais
    "XOM", "CVX", "COP", "FCX",
    # Brasil (ADR ou B3 via Yahoo)
    "VALE3.SA", "PETR4.SA", "ITUB4.SA", "WEGE3.SA", "B3SA3.SA"
]

FOREX_LIST = [
    "EURUSD=X", "GBPUSD=X", "USDJPY=X", "AUDUSD=X", "NZDUSD=X",
    "USDCAD=X", "USDCHF=X", "USDNOK=X", "USDSEK=X", "USDBRL=X"
]

CRYPTO_LIST = [
    "BTC-USD", "ETH-USD", "BNB-USD", "XRP-USD", "SOL-USD",
    "ADA-USD", "DOGE-USD", "MATIC-USD", "LTC-USD", "AVAX-USD"
]

INDICES_LIST = [
    "^GSPC",  # S&P 500
    "^DJI",   # Dow Jones
    "^IXIC",  # Nasdaq
    "^FTSE",  # FTSE 100
    "^GDAXI", # DAX
    "^HSI",   # Hang Seng
    "^N225",  # Nikkei
    "^BVSP"   # Ibovespa
]

PRESETS = {
    "Ações (principais)": EXCEL_STOCKS_LIST,
    "Forex (majors)": FOREX_LIST,
    "Cripto (top)": CRYPTO_LIST,
    "Índices globais": INDICES_LIST,
}

# ============================
# CONFIG / TEMA
# ============================

st.set_page_config(page_title="Trading Rooms IA", layout="wide")

st.markdown(
    """
    <style>
    .main { background-color: #f5f5f7; }
    .signal-card {
        background-color: #ffffff;
        border-radius: 10px;
        padding: 0.75rem 1rem;
        margin-bottom: 0.5rem;
        border: 1px solid #e0e0e0;
    }
    .signal-title { font-weight: 600; font-size: 0.95rem; }
    .signal-meta { font-size: 0.8rem; color: #555555; }
    .signal-badge-compra { color: #0b7a35; font-weight: 600; }
    .signal-badge-venda  { color: #b00020; font-weight: 600; }
    .signal-badge-neutro { color: #555555; font-weight: 600; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("📊 Trading Rooms IA – Atlas Lite Dashboard")

st.write(
    "Principais Ações, Forex, Índices e Cripto via Yahoo Finance. "
    "Escolha a categoria na barra lateral, ajuste a lista e veja os sinais nas salas Day/Swing/Position."
)

# ============================
# SIDEBAR – SELEÇÃO
# ============================

with st.sidebar:
    st.header("⚙️ Seleção de Ativos")
    categoria = st.selectbox("Categoria:", options=list(PRESETS.keys()))
    lista_padrao = PRESETS[categoria]

    tickers_selecionados = st.multiselect(
        f"Ativos ({len(lista_padrao)})",
        options=lista_padrao,
        default=lista_padrao,
    )

    st.caption("Listas reduzidas para performance. Você pode adicionar/remover símbolos aqui.")

if not tickers_selecionados:
    st.warning("Selecione pelo menos um ativo na barra lateral.")
    st.stop()

# ============================
# FUNÇÕES BACKEND
# ============================

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    roll_up = pd.Series(gain, index=series.index).rolling(period).mean()
    roll_down = pd.Series(loss, index=series.index).rolling(period).mean()
    rs = roll_up / (roll_down.replace(0, np.nan))
    rsi_val = 100.0 - (100.0 / (1.0 + rs))
    return rsi_val


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
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
    closes = df["Close"]
    ema_fast = closes.ewm(span=ema_fast_win, adjust=False).mean()
    ema_slow = closes.ewm(span=ema_slow_win, adjust=False).mean()
    rsi_val = rsi(closes, 14)
    atr_val = atr(df, 14)

    if len(closes) >= 2:
        ret_total = (closes.iloc[-1] / closes.iloc[0] - 1) * 100.0
    else:
        ret_total = np.nan

    rets = closes.pct_change().dropna()
    if len(rets) >= 2:
        vol_pct = rets.std() * np.sqrt(252) * 100.0
    else:
        vol_pct = np.nan

    return ema_fast, ema_slow, rsi_val, atr_val, ret_total, vol_pct


def gerar_sinal(final_close, ema_fast_last, ema_slow_last, rsi_last, atr_last,
                ret_total, vol_pct, tipo_sala: str):
    if any(np.isnan(x) for x in [final_close, ema_fast_last, ema_slow_last, rsi_last, ret_total]):
        return "NEUTRO", 0, "baixa", "Dados insuficientes.", final_close, final_close, final_close

    score = 50
    narrativa = []

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

    if ret_total > 0:
        score += min(ret_total / 2, 15)
        narrativa.append(f"Retorno positivo de {ret_total:.2f}% no período.")
    else:
        score += max(ret_total / 2, -15)
        narrativa.append(f"Retorno negativo de {ret_total:.2f}% no período.")

    if rsi_last > 70:
        score -= 10
        narrativa.append("RSI em sobrecompra (acima de 70).")
    elif rsi_last < 30:
        score += 10
        narrativa.append("RSI em sobrevenda (abaixo de 30).")
    else:
        narrativa.append("RSI em zona neutra.")

    if not np.isnan(vol_pct) and vol_pct > 100:
        score -= 10
        narrativa.append("Volatilidade muito alta, risco elevado.")
    elif not np.isnan(vol_pct):
        narrativa.append(f"Volatilidade anualizada aprox.: {vol_pct:.1f}%.")

    if tipo_sala == "day":
        narrativa.append("Sala Day: leitura focada em 1–5 dias.")
        risco_mult = 1.0
        alvo_mult = 2.0
    elif tipo_sala == "swing":
        narrativa.append("Sala Swing: leitura focada em algumas semanas.")
        risco_mult = 1.5
        alvo_mult = 2.0
    else:
        narrativa.append("Sala Position: leitura focada em tendência mais longa.")
        risco_mult = 2.0
        alvo_mult = 2.5

    score = int(max(0, min(100, score)))

    if score >= 60 and tendencia == "alta":
        direction = "COMPRA"
    elif score <= 40 and tendencia == "baixa":
        direction = "VENDA"
    else:
        direction = "NEUTRO"

    if score >= 80 or score <= 20:
        confidence = "alta"
    elif 60 <= score < 80 or 20 < score <= 40:
        confidence = "média"
    else:
        confidence = "baixa"

    if np.isnan(atr_last) or atr_last <= 0:
        risco_abs = final_close * 0.03
    else:
        risco_abs = atr_last * risco_mult

    if direction == "COMPRA":
        entrada = final_close
        stop = final_close - risco_abs
        alvo = final_close + risco_abs * alvo_mult
    elif direction == "VENDA":
        entrada = final_close
        stop = final_close + risco_abs
        alvo = final_close - risco_abs * alvo_mult
    else:
        entrada = final_close
        stop = final_close
        alvo = final_close

    narrativa.append(
        f"Estrutura de trade: entrada ~ {entrada:.2f}, stop {stop:.2f}, alvo {alvo:.2f}, "
        f"risco de ~{risco_abs / final_close * 100:.1f}% por unidade."
    )

    return direction, score, confidence, " ".join(narrativa), entrada, stop, alvo


def rodar_sala(tipo_sala: str, lista_tickers):
    if tipo_sala == "day":
        period = "5d"
    elif tipo_sala == "swing":
        period = "1mo"
    else:
        period = "3mo"

    dados = baixar_dados(lista_tickers, period=period, interval="1d")

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

        direction, score, conf, texto, entrada, stop, alvo = gerar_sinal(
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
                "Entrada": round(entrada, 2),
                "Stop": round(stop, 2),
                "Alvo": round(alvo, 2),
            }
        )

    if not resultados:
        return None, None, None

    df_rank = pd.DataFrame(resultados)
    ordem_direcao = {"COMPRA": 0, "VENDA": 1, "NEUTRO": 2}
    df_rank["ordem_direcao"] = df_rank["Direção"].map(ordem_direcao)
    df_rank = df_rank.sort_values(by=["ordem_direcao", "Score"], ascending=[True, False]).drop(
        columns=["ordem_direcao"]
    )
    return df_rank, narrativas, series_precos


def mostrar_resumo(df_rank: pd.DataFrame, titulo: str):
    col1, col2, col3, col4 = st.columns(4)
    n_compra = (df_rank["Direção"] == "COMPRA").sum()
    n_venda = (df_rank["Direção"] == "VENDA").sum()
    n_neutro = (df_rank["Direção"] == "NEUTRO").sum()

    with col1:
        st.metric("Sinais de COMPRA", n_compra)
    with col2:
        st.metric("Sinais de VENDA", n_venda)
    with col3:
        st.metric("Neutros / Observação", n_neutro)

    melhor = df_rank.sort_values("Score", ascending=False).iloc[0]
    with col4:
        st.metric(f"Melhor sinal ({titulo})", f"{melhor['Ativo']} ({melhor['Score']})")


def icone_direcao(direction: str) -> str:
    if direction == "COMPRA":
        return "🟢⬆"
    if direction == "VENDA":
        return "🔴⬇"
    return "⚪️➖"


def classe_badge(direction: str) -> str:
    if direction == "COMPRA":
        return "signal-badge-compra"
    if direction == "VENDA":
        return "signal-badge-venda"
    return "signal-badge-neutro"


def mostrar_cards(df_rank: pd.DataFrame, sala_key: str):
    st.subheader("Lista de sinais")

    sel_key = f"ativo_selecionado_{sala_key}"
    if sel_key not in st.session_state and not df_rank.empty:
        st.session_state[sel_key] = df_rank["Ativo"].iloc[0]

    for _, row in df_rank.iterrows():
        direction = row["Direção"]
        css_class = classe_badge(direction)
        icon = icone_direcao(direction)
        ativo = row["Ativo"]
        score = row["Score"]
        retorno = row["Retorno (%)"]
        rsi_val = row["RSI"]
        conf = row["Confiança"]
        entrada = row["Entrada"]
        stop = row["Stop"]
        alvo = row["Alvo"]

        clicado = st.button(
            f"{icon} {direction} – {ativo}",
            key=f"{sala_key}_{ativo}",
            use_container_width=True,
        )
        st.markdown(
            f"""
            <div class="signal-card">
              <div class="signal-title">
                <span class="{css_class}">{direction}</span> – {ativo}
              </div>
              <div class="signal-meta">
                Score: <b>{score}</b> | Retorno: {retorno}% | RSI: {rsi_val} | Confiança: {conf}<br/>
                Entrada: {entrada:.2f} · Stop: {stop:.2f} · Alvo: {alvo:.2f}
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if clicado:
            st.session_state[sel_key] = ativo

    return st.session_state.get(sel_key)


def mostrar_xray(ativo: str, preco_series: pd.Series, narrativa: str):
    st.markdown(f"### 📌 Raio-X – {ativo}")
    st.line_chart(preco_series)
    st.markdown("**Comentário Atlas Lite (plano completo):**")
    st.write(narrativa)


# ============================
# LAYOUT – TABS
# ============================

tab_day, tab_swing, tab_position = st.tabs(["Day Trade", "Swing Trade", "Position Trade"])

with tab_day:
    st.subheader("Day Trade – curtíssimo prazo")
    df_day, narr_day, prices_day = rodar_sala("day", tickers_selecionados)
    if df_day is None:
        st.warning("Não foi possível calcular sinais para a lista atual.")
    else:
        mostrar_resumo(df_day, "Day")
        col_list, col_detail = st.columns([1.2, 1.5])
        with col_list:
            ativo_sel = mostrar_cards(df_day, "day")
        with col_detail:
            if ativo_sel in narr_day and ativo_sel in prices_day:
                mostrar_xray(ativo_sel, prices_day[ativo_sel], narr_day[ativo_sel])

with tab_swing:
    st.subheader("Swing Trade – alguns dias/semanas")
    df_sw, narr_sw, prices_sw = rodar_sala("swing", tickers_selecionados)
    if df_sw is None:
        st.warning("Não foi possível calcular sinais para a lista atual.")
    else:
        mostrar_resumo(df_sw, "Swing")
        col_list, col_detail = st.columns([1.2, 1.5])
        with col_list:
            ativo_sel = mostrar_cards(df_sw, "swing")
        with col_detail:
            if ativo_sel in narr_sw and ativo_sel in prices_sw:
                mostrar_xray(ativo_sel, prices_sw[ativo_sel], narr_sw[ativo_sel])

with tab_position:
    st.subheader("Position Trade – tendências longas")
    df_pos, narr_pos, prices_pos = rodar_sala("position", tickers_selecionados)
    if df_pos is None:
        st.warning("Não foi possível calcular sinais para a lista atual.")
    else:
        mostrar_resumo(df_pos, "Position")
        col_list, col_detail = st.columns([1.2, 1.5])
        with col_list:
            ativo_sel = mostrar_cards(df_pos, "position")
        with col_detail:
            if ativo_sel in narr_pos and ativo_sel in prices_pos:
                mostrar_xray(ativo_sel, prices_pos[ativo_sel], narr_pos[ativo_sel])
