import streamlit as st
import os
import json
import glob
import pandas as pd
import time
import zipfile
import io
from main import run_analysis, INPUT_DIR, OUTPUT_DIR, get_chat_metadata

st.set_page_config(page_title="WhatsApp Chat Analytics", page_icon="📊", layout="wide")

def load_chat_files():
    meta_files = glob.glob(os.path.join(OUTPUT_DIR, "*_metadata.json"))
    chat_info = []
    for f in meta_files:
        with open(f, 'r', encoding='utf-8') as m:
            chat_info.append(json.load(m))
    return sorted(chat_info, key=lambda x: x.get('last_date', ''), reverse=True)

@st.cache_data(show_spinner=False)
def load_data(identity):
    json_path = os.path.join(OUTPUT_DIR, f"chat_{identity}_structured.json")
    summary_path = os.path.join(OUTPUT_DIR, f"chat_{identity}_resumo.md")

    data, summary = [], ""
    if os.path.exists(json_path):
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except: pass

    if os.path.exists(summary_path):
        try:
            with open(summary_path, 'r', encoding='utf-8') as f:
                summary = f.read()
        except: pass

    return data, summary

def get_processed_df(data):
    if not data: return pd.DataFrame()
    df = pd.DataFrame(data)
    if 'timestamp_iso' in df.columns:
        df['dt'] = pd.to_datetime(df['timestamp_iso'])
    else:
        def parse_dt(ts):
            clean_ts = ts.replace('\u202f', ' ').replace('\xa0', ' ').strip()
            for fmt in ["%d/%m/%Y, %I:%M:%S %p", "%d/%m/%Y, %H:%M:%S"]:
                try: return pd.to_datetime(clean_ts, format=fmt)
                except: continue
            return pd.to_datetime(clean_ts, errors='coerce')
        df['dt'] = df['timestamp'].apply(parse_dt)
    df['hour'] = df['dt'].dt.hour
    df['day_name'] = df['dt'].dt.day_name()
    return df

st.sidebar.title("🚀 Processamento")
uploaded_files = st.sidebar.file_uploader(
    "1. Upload de arquivos (ZIP ou .txt + áudios)",
    accept_multiple_files=True,
    type=["txt", "ogg", "opus", "m4a", "wav", "zip"]
)

if uploaded_files and st.sidebar.button("🔥 Iniciar Análise Completa", use_container_width=True):
    progress_bar = st.sidebar.progress(0)
    status_text = st.sidebar.empty()
    def update_progress(progress, message="Processando..."):
        progress_bar.progress(progress)
        status_text.text(f"{message} {int(progress * 100)}%")

    with st.spinner("Analisando..."):
        if not os.path.exists(INPUT_DIR): os.makedirs(INPUT_DIR)
        files_to_process = []
        for uploaded_file in uploaded_files:
            if uploaded_file.name.endswith(".zip"):
                with zipfile.ZipFile(io.BytesIO(uploaded_file.getvalue())) as z:
                    for zinfo in z.infolist():
                        if zinfo.is_dir() or "__MACOSX" in zinfo.filename or zinfo.filename.startswith("."):
                            continue
                        with z.open(zinfo) as zf:
                            files_to_process.append({"name": os.path.basename(zinfo.filename), "content": zf.read()})
            else:
                files_to_process.append({"name": uploaded_file.name, "content": uploaded_file.getvalue()})

        for item in files_to_process:
            fname, content = item["name"], item["content"]
            if fname.endswith(".txt"):
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

        if run_analysis(progress_callback=update_progress):
            st.sidebar.success("Pronto!")
            time.sleep(1)
            st.rerun()
        else:
            st.sidebar.error("Erro no processamento.")

st.sidebar.markdown("---")
st.title("📊 WhatsApp Chat Analytics")
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
            for f in glob.glob(os.path.join(OUTPUT_DIR, f"chat_{identity}*")):
                try: os.remove(f)
                except: pass
            input_file = os.path.join(INPUT_DIR, f"chat_{identity}.txt")
            if os.path.exists(input_file):
                try: os.remove(input_file)
                except: pass
            st.cache_data.clear()
            st.rerun()

    if selected_title:
        identity = titles_map[selected_title]
        data, summary = load_data(identity)

        if not data:
            st.info("Aguardando processamento... Os arquivos estão sendo gerados.")
            if st.button("🔄 Verificar novamente"):
                st.cache_data.clear()
                st.rerun()
            st.stop()

        df = get_processed_df(data)
        tab1, tab2, tab3, tab4 = st.tabs(["📝 Resumo IA", "📊 Insights", "💬 Mensagens", "📦 Dados Brutos"])

        with tab1:
            if summary: st.markdown(summary)
            else: st.info("Resumo não disponível.")

        with tab2:
            st.subheader("Análise de Engajamento")
            m1, m2, m3 = st.columns(3)
            audio_count = len(df[df["type"] == "audio"])
            m1.metric("Total de Mensagens", len(df))
            m2.metric("Áudios", audio_count)
            m3.metric("Texto", len(df) - audio_count)

            col_a, col_b = st.columns(2)
            with col_a:
                st.write("**Mensagens por Hora**")
                st.bar_chart(df.groupby('hour').size().reindex(range(24), fill_value=0))
            with col_b:
                st.write("**Mensagens por Dia**")
                days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                st.bar_chart(df.groupby('day_name').size().reindex(days_order, fill_value=0))
            st.write("**Top Participantes**")
            st.bar_chart(df['author'].value_counts())

        with tab3:
            col1, col2 = st.columns(2)
            with col1:
                authors = ["Todos"] + sorted(df["author"].unique().tolist())
                sel_author = st.selectbox("Autor", authors)
            with col2:
                search = st.text_input("Buscar", "")

            f_df = df.copy()
            if sel_author != "Todos": f_df = f_df[f_df["author"] == sel_author]
            if search: f_df = f_df[f_df["message"].str.contains(search, case=False, na=False)]

            st.write(f"Exibindo {len(f_df)} de {len(df)} mensagens")
            for idx, row in f_df.iterrows():
                with st.chat_message("user" if row["author"] == df["author"].unique()[0] else "assistant"):
                    st.write(f"**{row['author']}** - *{row['timestamp']}*")
                    st.write(row["message"])
                    if row["filename"]:
                        file_p = os.path.join(INPUT_DIR, row["filename"])
                        if os.path.exists(file_p):
                            ext = os.path.splitext(row["filename"])[1].lower()
                            if ext in [".opus", ".ogg", ".m4a", ".wav"]:
                                st.audio(file_p)
                            elif ext in [".jpg", ".jpeg", ".png"]:
                                st.image(file_p, caption=row["filename"], width="stretch")
                            elif ext in [".pdf", ".docx"]:
                                # Key única garantida usando hash do filename + timestamp + índice
                                unique_id = f"{row['filename']}_{row['timestamp']}_{idx}".replace(" ", "_")
                                btn_key = f"dl_{unique_id}"
                                st.download_button(
                                    f"📄 Baixar {row['filename']}",
                                    open(file_p, "rb").read(),
                                    row["filename"],
                                    key=btn_key
                                )
                    st.markdown("---")

        with tab4:
            st.subheader("JSON Estruturado")
            c1, c2 = st.columns(2)
            with c1: st.download_button("📥 Baixar JSON", json.dumps(data, indent=4, ensure_ascii=False), f"chat_{identity}.json", "application/json")
            with c2:
                if summary: st.download_button("📥 Baixar Resumo", summary, f"resumo_{identity}.md", "text/markdown")
            st.json(data)
