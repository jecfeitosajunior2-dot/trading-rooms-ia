def mostrar_cards(df_rank: pd.DataFrame, sala_key: str):
    st.subheader("Lista de sinais")

    # inicializa estado
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
                Score: <b>{score}</b> | Retorno: {retorno}% | RSI: {rsi_val} | Confiança: {conf}
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if clicado:
            st.session_state[sel_key] = ativo

    return st.session_state.get(sel_key)
