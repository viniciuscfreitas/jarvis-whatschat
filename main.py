import os
import re
import logging
import json
import glob
import hashlib
from datetime import datetime
import os
import re
import logging
import json
import glob
import hashlib
from datetime import datetime
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv

from engines import AudioTranscriber, DatabaseManager, Summarizer, DocumentProcessor
from utils import COMBINED_PATTERN, ATTACHMENT_PATTERN, FILE_TIMESTAMP_PATTERNS, sanitize_line, parse_log_date, clean_extra_whitespace

# Configuração
load_dotenv()

INPUT_DIR = os.getenv("INPUT_DIR", "input")
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "output")
MODEL_SIZE = os.getenv("MODEL_SIZE", "large-v3-turbo")
DEVICE = os.getenv("DEVICE", "cuda")
COMPUTE_TYPE = os.getenv("COMPUTE_TYPE", "float16")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3-flash-preview")

# Removido patterns daqui, estão no utils.py

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("WhatsAppETL")

AUDIO_HIDDEN_MARKERS = ["áudio ocultado", "audio ocultado", "Anexo de áudio"]

# Removido sanitize_line e parse_log_date daqui, estão no utils.py

def get_chat_metadata(file_path):
    try:
        with open(file_path, 'rb') as f:
            # Pega as primeiras linhas
            head = [f.readline().decode('utf-8', errors='ignore') for _ in range(500)]

            # Pega as últimas linhas (vê o final do arquivo)
            f.seek(0, os.SEEK_END)
            size = f.tell()
            f.seek(max(0, size - 10000)) # Pega os últimos ~10KB
            tail = f.read().decode('utf-8', errors='ignore').split('\n')
    except Exception as e:
        logger.error(f"Erro ao ler metadados: {e}")
        return None

    lines = [l.strip() for l in head if l.strip()]
    if not lines: return None

    identity = hashlib.md5(lines[0].encode()).hexdigest()[:10]

    participants = set()
    for line in lines[:100]:
        clean = sanitize_line(line)
        match = COMBINED_PATTERN.match(clean)
        if match:
            author = match.group("ios_author") or match.group("and_author")
            if author: participants.add(author)

    # Busca a última data no final do arquivo
    last_msg_date = "Data desconhecida"
    for line in reversed(tail):
        clean = sanitize_line(line)
        match = COMBINED_PATTERN.match(clean)
        if match:
            last_msg_date = match.group("ios_date") or match.group("and_date")
            break

    # Título mais robusto: Nome do arquivo + Participantes + Data
    filename_core = os.path.splitext(os.path.basename(file_path))[0].replace("chat_", "")
    participants_str = ", ".join(list(participants)[:2])
    title = f"[{filename_core}] {participants_str}... ({last_msg_date})"

    return {
        "identity": identity,
        "title": title,
        "last_date": last_msg_date,
        "hash": f"{os.path.getsize(file_path)}_{os.path.getmtime(file_path)}"
    }

def index_files_by_timestamp(input_dir):
    index = {}
    if not os.path.exists(input_dir): return index

    # os.scandir é mais rápido que os.listdir para muitos arquivos
    with os.scandir(input_dir) as entries:
        for entry in entries:
            if entry.is_file() and not entry.name.startswith("."):
                f = entry.name
                matched = False
                for p in FILE_TIMESTAMP_PATTERNS:
                    m = p.search(f)
                    if m:
                        d_str = m.group("date").replace("-", "")
                        meta = {"filename": f}
                        if "h" in m.groupdict():
                            meta["time"] = f"{m.group('h')}{m.group('m')}"
                        if d_str not in index: index[d_str] = []
                        index[d_str].append(meta)
                        matched = True
                        break
                if not matched:
                    if "others" not in index: index["others"] = []
                    index["others"].append({"filename": f})
    return index

def find_audio_by_timestamp_optimized(date_str, file_index):
    dt = parse_log_date(date_str)
    if not dt: return None
    date_key = dt.strftime("%Y%m%d")
    candidates = file_index.get(date_key, [])
    if not candidates: return None

    h = dt.strftime("%H")
    m = dt.strftime("%M")
    target_hm = f"{h}{m}"

    for c in candidates:
        if c.get("time") == target_hm:
            return c["filename"]

    for c in candidates:
        if c["filename"].startswith("PTT-"):
            return c["filename"]

    return None

def pre_process_attachments(chat_file_path, doc_processor, transcriber, db, file_index):
    try:
        attachments = set()
        with open(chat_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                clean_line = sanitize_line(line)

                # Busca anexos explícitos
                att_match = ATTACHMENT_PATTERN.search(clean_line)
                if att_match:
                    attachments.add(att_match.group("filename"))

                # Busca audios ocultos por timestamp
                if any(marker in clean_line for marker in AUDIO_HIDDEN_MARKERS):
                    match = COMBINED_PATTERN.match(clean_line)
                    if match:
                        date_str = match.group("ios_date") or match.group("and_date")
                        filename = find_audio_by_timestamp_optimized(date_str, file_index)
                        if filename:
                            attachments.add(filename)

        to_process_docs = []
        to_process_audio = []

        for filename in attachments:
            path = os.path.join(INPUT_DIR, filename)
            if os.path.exists(path):
                ext = os.path.splitext(filename)[1].lower()
                if ext in [".pdf", ".docx", ".jpg", ".jpeg", ".png"]:
                    file_hash = doc_processor.get_file_hash(path)
                    if not db.get_document_cache(file_hash):
                        to_process_docs.append(path)
                elif ext in [".opus", ".ogg", ".m4a", ".wav"]:
                    if not db.get_transcription(filename):
                        to_process_audio.append(path)

        if to_process_docs or to_process_audio:
            total = len(to_process_docs) + len(to_process_audio)
            logger.info(f"⚡ Pre-processando {total} novos anexos ({len(to_process_docs)} docs, {len(to_process_audio)} áudios)...")

            # Use separate workers for docs (Gemini API - network bound) and audio (Whisper - CPU/GPU bound)
            if to_process_docs:
                # Reduzido para 2 workers para evitar estourar cota de 10 req/min muito rápido
                with ThreadPoolExecutor(max_workers=2) as executor:
                    def process_doc_no_commit(p):
                        doc_processor.get_content(p, commit=False)
                    list(executor.map(process_doc_no_commit, to_process_docs))
                db.conn.commit()

            if to_process_audio:
                # For audio, we use fewer workers to avoid OOM if using GPU, or high CPU load
                # If DEVICE is cuda, we should probably stick to sequential or very few workers
                max_audio_workers = 1 if DEVICE == "cuda" else 2
                with ThreadPoolExecutor(max_workers=max_audio_workers) as executor:
                    def transcribe_and_cache(p):
                        try:
                            fname = os.path.basename(p)
                            text = transcriber.transcribe(p)
                            db.save_transcription(fname, text, commit=False)
                        except Exception as e:
                            logger.error(f"Erro ao transcrever {p}: {e}")

                    list(executor.map(transcribe_and_cache, to_process_audio))
                db.conn.commit()

    except Exception as e:
        logger.error(f"Erro no pre-processamento: {e}")

def process_file(file_path, transcriber, doc_processor, db, file_index, identity_prefix, progress_callback=None):
    logger.info(f"📄 Processando: {os.path.basename(file_path)}")

    output_txt_path = os.path.join(OUTPUT_DIR, f"{identity_prefix}_final.txt")
    output_json_path = os.path.join(OUTPUT_DIR, f"{identity_prefix}_structured.json")

    with open(file_path, 'r', encoding='utf-8') as f, \
         open(output_txt_path, 'w', encoding='utf-8') as out_f, \
         open(output_json_path, 'w', encoding='utf-8') as out_j:

        out_j.write("[\n") # Inicia array JSON

        plain_text_for_summary = []
        current_plain_chars = 0

        file_size = os.path.getsize(file_path)
        bytes_processed = 0
        detected_fmt = None
        first_json_entry = True

        pbar = tqdm(total=file_size, unit='B', unit_scale=True, desc="Processando", leave=False)

        for i, line in enumerate(f):
            line_bytes = len(line.encode('utf-8'))
            bytes_processed += line_bytes
            pbar.update(line_bytes)

            if progress_callback and i % 500 == 0:
                progress_callback(bytes_processed / file_size)

            clean_line = sanitize_line(line)
            match = COMBINED_PATTERN.match(clean_line)
            if not match: continue

            if match.group("ios_date"):
                date_str, author, content = match.group("ios_date"), match.group("ios_author"), match.group("content")
            else:
                date_str, author, content = match.group("and_date"), match.group("and_author"), match.group("content")

            dt_obj, fmt = parse_log_date(date_str, detected_fmt)
            if fmt: detected_fmt = fmt
            iso_date = dt_obj.isoformat() if dt_obj else None

            attachment_match = ATTACHMENT_PATTERN.search(content)
            is_hidden_audio = any(marker in content for marker in AUDIO_HIDDEN_MARKERS)

            msg_type, filename, final_message = "text", None, content

            if attachment_match or is_hidden_audio:
                if attachment_match:
                    potential_filename = attachment_match.group("filename")
                    if os.path.exists(os.path.join(INPUT_DIR, potential_filename)):
                        filename = potential_filename
                if not filename:
                    filename = find_audio_by_timestamp_optimized(date_str, file_index)

                if filename:
                    ext = os.path.splitext(filename)[1].lower()
                    if ext in [".opus", ".ogg", ".m4a", ".wav"] or is_hidden_audio:
                        msg_type = "audio"
                        cached = db.get_transcription(filename)
                        if cached:
                            text, is_err = cached
                            final_message = f"[ÁUDIO CACHED]: {text}" if not is_err else f"[ÁUDIO ERRO]: {filename}"
                        else:
                            path = os.path.join(INPUT_DIR, filename)
                            if os.path.exists(path):
                                try:
                                    text = transcriber.transcribe(path)
                                    db.save_transcription(filename, text)
                                    final_message = f"[ÁUDIO]: {text}"
                                except Exception as e:
                                    db.save_transcription(filename, str(e), is_error=1)
                                    final_message = f"[ERRO ÁUDIO]: {filename}"
                            else:
                                final_message = f"[FALTA ÁUDIO]: {filename}"
                    elif ext in [".pdf", ".docx", ".jpg", ".jpeg", ".png"]:
                        msg_type = "attachment"
                        extracted = doc_processor.get_content(os.path.join(INPUT_DIR, filename))
                        # Limpa espaços extras do conteúdo extraído para o log final
                        extracted = clean_extra_whitespace(extracted)
                        final_message = f"[ANEXO {filename}]: {extracted}"

            entry = {
                "id": i,
                "timestamp": date_str, "timestamp_iso": iso_date,
                "author": author, "message": final_message,
                "type": msg_type, "filename": filename
            }

            # Streaming JSON: grava no disco na hora
            if not first_json_entry: out_j.write(",\n")
            json.dump(entry, out_j, ensure_ascii=False)
            first_json_entry = False

            final_line = f"ID[{i}] [{date_str}] {author}: {final_message}"
            out_f.write(final_line + "\n")

            if current_plain_chars < 500000:
                plain_text_for_summary.append(final_line)
                current_plain_chars += len(final_line)

        out_j.write("\n]") # Fecha array JSON

    pbar.close()
    if progress_callback: progress_callback(1.0)
    return output_json_path, output_txt_path

# Cache global para evitar recarregamento de modelos (especialmente Whisper)
_CACHED_ENGINES = None

def get_engines():
    global _CACHED_ENGINES
    if _CACHED_ENGINES is None:
        db = DatabaseManager(os.path.join(OUTPUT_DIR, "transcription_cache.db"))
        transcriber = AudioTranscriber(MODEL_SIZE, DEVICE, COMPUTE_TYPE)
        doc_processor = DocumentProcessor(GEMINI_API_KEY, db, GEMINI_MODEL)
        summarizer = Summarizer(GEMINI_API_KEY, GEMINI_MODEL)
        _CACHED_ENGINES = (db, transcriber, doc_processor, summarizer)
    return _CACHED_ENGINES

def run_analysis(progress_callback=None):
    if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)

    db, transcriber, doc_processor, summarizer = get_engines()

    chat_files = glob.glob(os.path.join(INPUT_DIR, "*.txt"))
    if not chat_files:
        logger.error("Nenhum .txt no input.")
        db.close()
        return False

    # Calculate index once
    file_index = index_files_by_timestamp(INPUT_DIR)

    for file_path in chat_files:
        try:
            meta = get_chat_metadata(file_path)
            identity_prefix = f"chat_{meta['identity']}" if meta else os.path.splitext(os.path.basename(file_path))[0]
            if meta:
                with open(os.path.join(OUTPUT_DIR, f"{identity_prefix}_metadata.json"), 'w', encoding='utf-8') as mf:
                    json.dump(meta, mf, indent=4, ensure_ascii=False)

            if progress_callback: progress_callback(0.05, f"Pre-processando {identity_prefix}...")
            pre_process_attachments(file_path, doc_processor, transcriber, db, file_index)

            structured, plain = process_file(file_path, transcriber, doc_processor, db, file_index, identity_prefix, progress_callback)

            # JSON já foi salvo via streaming dentro de process_file

            summary_path = os.path.join(OUTPUT_DIR, f"{identity_prefix}_resumo.md")
            if not os.path.exists(summary_path) or os.path.getmtime(file_path) > os.path.getmtime(summary_path):
                if progress_callback: progress_callback(0.95, "Gerando resumo detalhado...")
                # Passamos o caminho do arquivo agora, não o texto na RAM
                summary = summarizer.summarize(plain)
                if summary:
                    with open(summary_path, 'w', encoding='utf-8') as f: f.write(f"# Resumo: {identity_prefix}\n\n{summary}")
        except Exception as e:
            logger.error(f"Erro ao processar {file_path}: {e}")

    db.conn.commit()
    # Não fechamos mais o banco aqui para permitir reuso no dashboard
    return True

if __name__ == "__main__":
    run_analysis()
