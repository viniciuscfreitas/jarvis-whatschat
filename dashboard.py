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

@st.cache_data
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

def get_processed_df(data):
    """Transforma dados brutos em DataFrame com colunas de tempo úteis."""
    if not data:
        return pd.DataFrame()

    df = pd.DataFrame(data)
    if 'timestamp_iso' in df.columns:
        df['dt'] = pd.to_datetime(df['timestamp_iso'])
    else:
        # Fallback para versões antigas do JSON
        def parse_dt(ts):
            clean_ts = ts.replace('\u202f', ' ').replace('\xa0', ' ').strip()
            for fmt in ["%d/%m/%Y, %I:%M:%S %p", "%d/%m/%Y, %H:%M:%S"]:
                try:
                    return pd.to_datetime(clean_ts, format=fmt)
                except:
                    continue
            return pd.to_datetime(clean_ts, errors='coerce')
        df['dt'] = df['timestamp'].apply(parse_dt)

    df['hour'] = df['dt'].dt.hour
    df['day_name'] = df['dt'].dt.day_name()
    return df

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

    if st.sidebar.button("🗑️ Deletar Conversa Selecionada"):
        if selected_title:
            identity = titles_map[selected_title]
            files_to_delete = glob.glob(os.path.join(OUTPUT_DIR, f"chat_{identity}*"))
            input_file = os.path.join(INPUT_DIR, f"chat_{identity}.txt")
            if os.path.exists(input_file):
                files_to_delete.append(input_file)

            for f in files_to_delete:
                try:
                    os.remove(f)
                except:
                    pass
            st.sidebar.success("Conversa deletada!")
            time.sleep(1)
            st.rerun()

    if selected_title:
        identity = titles_map[selected_title]
        data, summary = load_data(identity)

        if not data:
            st.info("Aguardando processamento desta conversa...")
            st.stop()

        df = get_processed_df(data)

        tab1, tab2, tab3, tab4 = st.tabs(["📝 Resumo IA", "📊 Insights", "💬 Mensagens", "📦 Dados Brutos"])

        with tab1:
            if summary:
                st.markdown(summary)
            else:
                st.info("Resumo não disponível para este chat.")

        with tab2:
            st.subheader("Análise de Engajamento")

            # Métricas principais
            m1, m2, m3 = st.columns(3)
            total_msgs = len(df)
            audio_count = len(df[df["type"] == "audio"])
            text_count = total_msgs - audio_count

            m1.metric("Total de Mensagens", total_msgs)
            m2.metric("Áudios", audio_count)
            m3.metric("Texto", text_count)

            col_a, col_b = st.columns(2)

            with col_a:
                st.write("**Mensagens por Hora do Dia**")
                hourly_counts = df.groupby('hour').size().reindex(range(24), fill_value=0)
                st.bar_chart(hourly_counts)

            with col_b:
                st.write("**Mensagens por Dia da Semana**")
                days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                dow_counts = df.groupby('day_name').size().reindex(days_order, fill_value=0)
                st.bar_chart(dow_counts)

            st.write("**Top Participantes**")
            author_counts = df['author'].value_counts()
            st.bar_chart(author_counts)

        with tab3:
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
                        if "filename" in row and row["filename"]:
                            audio_path = os.path.join(INPUT_DIR, row["filename"])
                            if os.path.exists(audio_path):
                                st.audio(audio_path)
                    st.markdown("---")

        with tab4:
            st.subheader("JSON Estruturado")
            col_d1, col_d2 = st.columns(2)
            with col_d1:
                st.download_button(
                    label="📥 Baixar JSON",
                    data=json.dumps(data, indent=4, ensure_ascii=False),
                    file_name=f"chat_{identity}_structured.json",
                    mime="application/json"
                )
            with col_d2:
                if summary:
                    st.download_button(
                        label="📥 Baixar Resumo",
                        data=summary,
                        file_name=f"chat_{identity}_resumo.md",
                        mime="text/markdown"
                    )
            st.json(data)
