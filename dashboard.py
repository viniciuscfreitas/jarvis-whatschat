import streamlit as st
import os
import json
import glob
import pandas as pd
import time
import hashlib
from main import run_analysis, INPUT_DIR, OUTPUT_DIR, get_chat_metadata

# Configurações da página
st.set_page_config(page_title="WhatsApp Chat Analytics", page_icon="📊", layout="wide")

def load_chat_files():
    # Carrega arquivos de metadados para ter os títulos bonitos
    meta_files = glob.glob(os.path.join(OUTPUT_DIR, "*_metadata.json"))
    chat_info = []
    for f in meta_files:
        with open(f, 'r', encoding='utf-8') as m:
            chat_info.append(json.load(m))
    # Ordenar por data (opcional) ou nome
    return sorted(chat_info, key=lambda x: x.get('last_date', ''), reverse=True)

def load_data(identity):
    json_path = os.path.join(OUTPUT_DIR, f"chat_{identity}_structured.json")
    summary_path = os.path.join(OUTPUT_DIR, f"chat_{identity}_resumo.md")

    data = []
    if os.path.exists(json_path):
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception:
            data = []

    summary = ""
    if os.path.exists(summary_path):
        try:
            with open(summary_path, 'r', encoding='utf-8') as f:
                summary = f.read()
        except Exception:
            summary = ""

    return data, summary

# --- SIDEBAR: GESTÃO DE ARQUIVOS ---
st.sidebar.title("🚀 Processamento")

uploaded_files = st.sidebar.file_uploader(
    "1. Upload de arquivos (.txt, .ogg, .m4a)",
    accept_multiple_files=True,
    type=["txt", "ogg", "opus", "m4a", "wav"],
    help="Arraste o _chat.txt e os áudios correspondentes."
)

if uploaded_files:
    if st.sidebar.button("🔥 Iniciar Análise Completa", use_container_width=True):
        progress_bar = st.sidebar.progress(0)
        status_text = st.sidebar.empty()

        def update_progress(progress, message="Processando..."):
            progress_bar.progress(progress)
            status_text.text(f"{message} {int(progress * 100)}%")

        with st.spinner("Salvando e Analisando..."):
            # Passo 1: Salvar Arquivos Automaticamente
            if not os.path.exists(INPUT_DIR): os.makedirs(INPUT_DIR)
            for uploaded_file in uploaded_files:
                content = uploaded_file.getvalue()
                fname = uploaded_file.name

                if fname == "_chat.txt":
                    meta = get_chat_metadata(content)
                    if meta:
                        identity = meta['identity']
                        fname = f"chat_{identity}.txt"
                        meta_path = os.path.join(OUTPUT_DIR, f"chat_{identity}_metadata.json")
                        if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)
                        with open(meta_path, 'w', encoding='utf-8') as f:
                            json.dump(meta, f, indent=4, ensure_ascii=False)

                with open(os.path.join(INPUT_DIR, fname), "wb") as f:
                    f.write(content)

            # Passo 2: Rodar Análise
            success = run_analysis(progress_callback=update_progress)

            if success:
                st.sidebar.success("Pronto!")
                time.sleep(1)
                st.rerun()
            else:
                st.sidebar.error("Erro no processamento.")

st.sidebar.markdown("---")

# --- MAIN UI ---
st.title("📊 WhatsApp Chat Analytics")
st.markdown("---")

st.sidebar.title("🧐 Visualizar")
chat_list = load_chat_files()

if not chat_list:
    st.info("Nenhum chat processado.")
else:
    titles_map = {chat['title']: chat['identity'] for chat in chat_list}
    selected_title = st.sidebar.selectbox("Selecionar Conversa", list(titles_map.keys()))

    if selected_title:
        identity = titles_map[selected_title]
        data, summary = load_data(identity)

        if not data:
            st.info("Aguardando processamento desta conversa...")
            st.stop()

        df = pd.DataFrame(data)

        tab1, tab2, tab3 = st.tabs(["📝 Resumo IA", "💬 Mensagens", "📦 Dados Brutos"])

        with tab1:
            if summary:
                st.markdown(summary)
            else:
                st.info("Resumo não disponível para este chat.")

        with tab2:
            st.subheader(f"Explorador: {selected_title}")

            # Filtros
            col1, col2 = st.columns(2)
            with col1:
                authors = ["Todos"] + sorted(df["author"].unique().tolist())
                selected_author = st.selectbox("Filtrar por Autor", authors)
            with col2:
                search_term = st.text_input("Buscar por termo", "")

            # Aplicação dos filtros
            filtered_df = df.copy()
            if "author" in filtered_df.columns:
                if selected_author != "Todos":
                    filtered_df = filtered_df[filtered_df["author"] == selected_author]

            if search_term and "message" in filtered_df.columns:
                filtered_df = filtered_df[filtered_df["message"].str.contains(search_term, case=False, na=False)]

            st.write(f"Exibindo {len(filtered_df)} de {len(df)} mensagens")

            # Exibição estilizada
            for _, row in filtered_df.iterrows():
                with st.chat_message("user" if row["author"] == df["author"].unique()[0] else "assistant"):
                    st.write(f"**{row['author']}** - *{row['timestamp']}*")
                    st.write(row["message"])
                    if row["type"] == "audio":
                        st.caption("🎙️ Mensagem de Áudio")
                    st.markdown("---")

        with tab3:
            st.subheader("JSON Estruturado")
            st.json(data)
