import os
import re
import sqlite3
import logging
import json
import glob
import hashlib
from datetime import datetime
from dotenv import load_dotenv
from faster_whisper import WhisperModel
from tqdm import tqdm
from google import genai
from google.genai import types
import docx
from pypdf import PdfReader
import time
from concurrent.futures import ThreadPoolExecutor

# Configuração
load_dotenv()

INPUT_DIR = os.getenv("INPUT_DIR", "input")
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "output")
MODEL_SIZE = os.getenv("MODEL_SIZE", "large-v3-turbo")
DEVICE = os.getenv("DEVICE", "cuda")
COMPUTE_TYPE = os.getenv("COMPUTE_TYPE", "float16")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

IOS_MSG_PATTERN = re.compile(r"^\[(?P<date>\d{2}/\d{2}/\d{4}, \d{1,2}:\d{2}:\d{2}.*?)\] (?P<author>.*?): (?P<content>.*)")
ANDROID_MSG_PATTERN = re.compile(r"^(?P<date>\d{1,2}/\d{1,2}/\d{2,4}, \d{1,2}:\d{2}) - (?P<author>.*?): (?P<content>.*)")
ATTACHMENT_PATTERN = re.compile(r"(?:<anexado: |)(?P<filename>.*?\.(?:opus|ogg|m4a|wav|pdf|docx|jpg|jpeg|png))(?:>| \(file attached\))")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("WhatsAppETL")

AUDIO_HIDDEN_MARKERS = ["áudio ocultado", "audio ocultado", "Anexo de áudio"]
DATE_FORMATS = [
    "%d/%m/%Y, %I:%M:%S %p",
    "%d/%m/%Y, %H:%M:%S",
    "%m/%d/%y, %H:%M",
    "%d/%m/%y, %H:%M"
]

def get_chat_metadata(file_content):
    lines = [l.strip() for l in file_content.decode('utf-8').split('\n') if l.strip()]
    if not lines: return None

    first_msg = lines[0]
    identity = hashlib.md5(first_msg.encode()).hexdigest()[:10]

    participants = set()
    for line in lines[:100]:
        clean = sanitize_line(line)
        match = IOS_MSG_PATTERN.match(clean) or ANDROID_MSG_PATTERN.match(clean)
        if match:
            participants.add(match.group("author"))

    last_msg_date = "Data desconhecida"
    for line in reversed(lines):
        clean = sanitize_line(line)
        match = IOS_MSG_PATTERN.match(clean) or ANDROID_MSG_PATTERN.match(clean)
        if match:
            last_msg_date = match.group("date")
            break

    title = f"{', '.join(list(participants)[:2])}... ({last_msg_date})"
    return {
        "identity": identity,
        "title": title,
        "last_date": last_msg_date,
        "hash": hashlib.md5(file_content).hexdigest()
    }

def index_files_by_timestamp(input_dir):
    """
    Cria um índice de arquivos baseado em padrões de data conhecidos no nome do arquivo.
    Retorna: { "YYYYMMDD_HHMM": [filename1, filename2], ... }
    """
    index = {}
    files = os.listdir(input_dir)

    # Padrões para extrair data de PTTs e arquivos
    patterns = [
        # PTT-20250324-WA0002.opus -> 20250324
        re.compile(r"PTT-(?P<date>\d{8})-WA"),
        # WhatsApp Ptt 2025-03-24 at 14.30.55.ogg -> 2025-03-24, 14, 30
        re.compile(r"WhatsApp Ptt (?P<date>\d{4}-\d{2}-\d{2}) at (?P<h>\d{2})\.(?P<m>\d{2})"),
        # IMG-20250324-WA0000.jpg -> 20250324
        re.compile(r"IMG-(?P<date>\d{8})-WA"),
        # DOC-20250324-WA0000.pdf -> 20250324
        re.compile(r"DOC-(?P<date>\d{8})-WA"),
        # VID-20250324-WA0000.mp4 -> 20250324
        re.compile(r"VID-(?P<date>\d{8})-WA")
    ]

    for f in files:
        if f.startswith("."): continue # Pula arquivos ocultos

        # Tenta casar com algum padrão
        matched = False
        for p in patterns:
            m = p.search(f)
            if m:
                # Chave simples baseada na DATA (YYYYMMDD) para busca rápida
                # Se tiver hora, refinamos, mas a chave principal é a data
                d_str = m.group("date").replace("-", "") # Normaliza para YYYYMMDD

                # Se tiver hora/minuto (padrão iOS), adiciona info extra na lista
                meta = {"filename": f, "full_match": m.group(0)}
                if "h" in m.groupdict():
                    meta["time"] = f"{m.group('h')}{m.group('m')}"

                if d_str not in index: index[d_str] = []
                index[d_str].append(meta)
                matched = True
                break

        # Se não casou padrão conhecido, guarda num bucket genérico "others"
        if not matched:
            if "others" not in index: index["others"] = []
            index["others"].append({"filename": f})

    return index

def find_audio_by_timestamp_optimized(date_str, file_index):
    try:
        dt = None
        clean_date = sanitize_line(date_str)
        for fmt in DATE_FORMATS:
            try:
                dt = datetime.strptime(clean_date, fmt)
                break
            except ValueError:
                continue

        if not dt: return None

        date_key = dt.strftime("%Y%m%d") # Chave de busca: YYYYMMDD

        # 1. Busca direta na data (O(1))
        candidates = file_index.get(date_key, [])
        if not candidates: return None

        # 2. Refinamento dentro dos candidatos do dia (O(K) onde K é pequeno)
        h_with_zero = dt.strftime("%I")
        h_no_zero = str(int(h_with_zero))
        m = dt.strftime("%M")
        s = dt.strftime("%S")
        p = dt.strftime("%p").lower()

        # Busca exata (iOS style com hora)
        for c in candidates:
            f = c["filename"]
            f_low = f.lower()

            # Padrão iOS Ptt
            # "WhatsApp Ptt 2025-03-24 at 14.30.55"
            if "WhatsApp Ptt" in f:
                # Tenta match solto de hora/minuto
                target_hm = f"{h_no_zero}.{m}"
                target_hm_alt = f"{h_with_zero}.{m}"
                if (target_hm in f_low or target_hm_alt in f_low) and p in f_low:
                     if f_low.endswith(('.ogg', '.opus', '.m4a', '.wav')):
                        return f

            # Padrão Android PTT-YYYYMMDD-WA...
            # Como a chave já é a data, qualquer PTT desse dia é candidato
            # Mas geralmente queremos associar se não tiver um específico de hora
            if f.startswith("PTT-") and f_low.endswith(('.ogg', '.opus', '.m4a', '.wav')):
                 # Aqui poderíamos refinar se tivéssemos o índice sequencial WA0001
                 # Por simplicidade, retorna o primeiro compatível se a busca exata falhou
                 pass

        # Se não achou exato, tenta o primeiro PTT do dia (fallback comum)
        for c in candidates:
            f = c["filename"]
            if f.startswith("PTT-") and f.lower().endswith(('.ogg', '.opus', '.m4a', '.wav')):
                return f

    except Exception as e:
        logger.error(f"Erro em find_audio_by_timestamp_optimized: {e}")
    return None

class AudioTranscriber:
    def __init__(self):
        self.model = None

    def _load_model(self):
        if self.model is None:
            logger.info(f"🐢 Carregando Whisper ({MODEL_SIZE}) em {DEVICE}...")
            self.model = WhisperModel(MODEL_SIZE, device=DEVICE, compute_type=COMPUTE_TYPE)

    def transcribe(self, audio_path):
        self._load_model()
        segments, _ = self.model.transcribe(audio_path, beam_size=5, language="pt")
        return " ".join([s.text for s in segments]).strip()

class DatabaseManager:
    def __init__(self, db_path):
        self.conn = sqlite3.connect(db_path)
        self.cursor = self.conn.cursor()
        self.cursor.execute("CREATE TABLE IF NOT EXISTS transcriptions (filename TEXT PRIMARY KEY, text TEXT, is_error INTEGER DEFAULT 0, processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)")
        self.cursor.execute("CREATE TABLE IF NOT EXISTS document_cache (hash TEXT PRIMARY KEY, filename TEXT, text_content TEXT, processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)")
        self.conn.commit()

    def get_transcription(self, filename):
        self.cursor.execute("SELECT text, is_error FROM transcriptions WHERE filename = ?", (filename,))
        return self.cursor.fetchone()

    def save_transcription(self, filename, text, is_error=0):
        self.cursor.execute("INSERT OR REPLACE INTO transcriptions (filename, text, is_error) VALUES (?, ?, ?)", (filename, text, is_error))
        self.conn.commit()

    def get_document_cache(self, file_hash):
        self.cursor.execute("SELECT text_content FROM document_cache WHERE hash = ?", (file_hash,))
        result = self.cursor.fetchone()
        return result[0] if result else None

    def save_document_cache(self, file_hash, filename, text_content):
        self.cursor.execute("INSERT OR REPLACE INTO document_cache (hash, filename, text_content) VALUES (?, ?, ?)", (file_hash, filename, text_content))
        self.conn.commit()

    def close(self):
        self.conn.close()

class Summarizer:
    def __init__(self, api_key):
        self.enabled = False
        if api_key and api_key != "seu_token_aqui":
            self.client = genai.Client(api_key=api_key)
            self.model_id = "gemini-3-flash-preview"
            self.enabled = True
        else:
            logger.warning("⚠️ GEMINI_API_KEY não configurada. O resumo será pulado.")

    def summarize(self, text):
        if not self.enabled: return None
        logger.info(f"🧠 Gerando resumo via {self.model_id}...")
        prompt = f"Resuma os pontos principais desta conversa de WhatsApp de forma executiva e em português:\n\n{text[:500000]}"
        try:
            response = self.client.models.generate_content(
                model=self.model_id,
                contents=prompt
            )
            return response.text
        except Exception as e:
            logger.error(f"Erro ao gerar resumo: {e}")
            return f"Erro ao gerar resumo: {e}"

class DocumentProcessor:
    def __init__(self, api_key, db):
        self.db = db
        self.enabled = False
        self.model_id = "gemini-3-flash-preview"
        if api_key and api_key != "seu_token_aqui":
            self.client = genai.Client(api_key=api_key)
            self.enabled = True
        else:
            logger.warning("⚠️ GEMINI_API_KEY não configurada para DocumentProcessor.")

    def get_file_hash(self, file_path):
        # Hash em chunks para não estourar memória com arquivos grandes
        hasher = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hasher.update(chunk)
        return hasher.hexdigest()

    def extract_text_from_pdf(self, file_path):
        try:
            reader = PdfReader(file_path)
            if reader.is_encrypted:
                logger.warning(f"🔒 PDF encriptado detectado: {file_path}")
                return "[PDF PROTEGIDO POR SENHA - TEXTO NÃO EXTRAÍDO]"

            text = ""
            for page in reader.pages:
                text += page.extract_text() or ""
            return text.strip()
        except Exception as e:
            logger.error(f"Erro ao ler PDF {file_path}: {e}")
            return f"[ERRO LEITURA PDF: {e}]"

    def extract_text_from_docx(self, file_path):
        try:
            doc = docx.Document(file_path)
            return "\n".join([para.text for para in doc.paragraphs]).strip()
        except Exception as e:
            logger.error(f"Erro ao ler DOCX {file_path}: {e}")
            return ""

    def process_with_gemini(self, file_path, prompt, retries=3):
        if not self.enabled:
            return "[GEMINI DISABLED]"

        try:
            with open(file_path, "rb") as f:
                file_data = f.read()

            mime_type = "image/jpeg"
            low_path = file_path.lower()
            if low_path.endswith(".png"): mime_type = "image/png"
            elif low_path.endswith(".pdf"): mime_type = "application/pdf"

            for attempt in range(retries):
                try:
                    response = self.client.models.generate_content(
                        model=self.model_id,
                        contents=[
                            prompt,
                            types.Part.from_bytes(data=file_data, mime_type=mime_type)
                        ]
                    )
                    return response.text
                except Exception as e:
                    if attempt == retries - 1: raise e
                    logger.warning(f"⚠️ Erro Gemini (Tentativa {attempt+1}/{retries}): {e}. Retentando em 2s...")
                    time.sleep(2)

        except Exception as e:
            logger.error(f"Erro no Gemini para {file_path}: {e}")
            return f"[ERRO GEMINI]: {e}"

    def get_content(self, file_path):
        if not os.path.exists(file_path):
            return "[ARQUIVO NÃO ENCONTRADO]"

        file_hash = self.get_file_hash(file_path)
        cached = self.db.get_document_cache(file_hash)
        if cached:
            return cached

        ext = os.path.splitext(file_path)[1].lower()
        content = ""

        if ext == ".pdf":
            content = self.extract_text_from_pdf(file_path)
            if len(content) < 100:
                logger.info(f"🔍 PDF {os.path.basename(file_path)} parece escaneado ou imagem. Usando Gemini OCR...")
                content = self.process_with_gemini(file_path, "Extraia todo o texto deste PDF. Se houver imagens de documentos, descreva-os detalhadamente.")
        elif ext == ".docx":
            content = self.extract_text_from_docx(file_path)
        elif ext in [".jpg", ".jpeg", ".png"]:
            logger.info(f"👁️ Analisando imagem {os.path.basename(file_path)}...")
            content = self.process_with_gemini(file_path, "Descreva esta imagem detalhadamente para um caso jurídico. Se for um documento ou print, extraia todo o texto visível.")

        if content:
            self.db.save_document_cache(file_hash, os.path.basename(file_path), content)
            return content

        return "[CONTEÚDO NÃO EXTRAÍDO]"

def sanitize_line(line):
    return line.replace('\u200e', '').replace('\u200f', '').replace('\u202f', ' ').replace('\xa0', ' ').strip()

def parse_log_date(date_str):
    clean_date = sanitize_line(date_str)
    for fmt in DATE_FORMATS:
        try:
            return datetime.strptime(clean_date, fmt)
        except ValueError:
            continue
    return None

def pre_process_attachments(chat_file_path, doc_processor):
    """
    Varre o chat em busca de anexos e processa-os em paralelo para encher o cache.
    Isso evita o processamento sequencial lento durante o loop principal.
    """
    try:
        with open(chat_file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Encontra nomes de arquivos únicos citados no chat
        attachments = list(set(ATTACHMENT_PATTERN.findall(content)))
        to_process = []

        for filename in attachments:
            path = os.path.join(INPUT_DIR, filename)
            if os.path.exists(path):
                ext = os.path.splitext(filename)[1].lower()
                # Apenas documentos e imagens (áudio já é rápido via faster-whisper local)
                if ext in [".pdf", ".docx", ".jpg", ".jpeg", ".png"]:
                    # Só adiciona se NÃO estiver no cache
                    file_hash = doc_processor.get_file_hash(path)
                    if not doc_processor.db.get_document_cache(file_hash):
                        to_process.append(path)

        if not to_process:
            return

        logger.info(f"⚡ Pre-processando {len(to_process)} novos anexos em paralelo (max 5 workers)...")

        # Processa em paralelo (Gemini lida bem com múltiplas requisições)
        with ThreadPoolExecutor(max_workers=5) as executor:
            # list() força a execução do gerador map
            list(executor.map(doc_processor.get_content, to_process))

    except Exception as e:
        logger.error(f"Erro no pre-processamento de anexos: {e}")

def process_file(file_path, transcriber, doc_processor, db, progress_callback=None):
    logger.info(f"📄 Processando: {os.path.basename(file_path)}")

    # 1. Indexação prévia dos arquivos (O(N) uma vez só)
    file_index = index_files_by_timestamp(INPUT_DIR)
    input_files_list = os.listdir(INPUT_DIR) # Fallback para busca antiga se necessário

    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    is_valid = any((IOS_MSG_PATTERN.match(sanitize_line(l)) or ANDROID_MSG_PATTERN.match(sanitize_line(l))) for l in lines[:10] if l.strip())
    if not is_valid and len(lines) > 0:
        logger.error(f"❌ Arquivo inválido ou formato incompatível: {file_path}")
        return [], ""

    structured_data = []
    plain_text_lines = []
    total_lines = len(lines)

    for i, line in enumerate(tqdm(lines, desc="Linhas", leave=False)):
        if progress_callback: progress_callback(i / total_lines)

        clean_line = sanitize_line(line)
        match = IOS_MSG_PATTERN.match(clean_line) or ANDROID_MSG_PATTERN.match(clean_line)
        if not match: continue

        date_str = match.group("date")
        author = match.group("author")
        content = match.group("content")

        dt_obj = parse_log_date(date_str)
        iso_date = dt_obj.isoformat() if dt_obj else None

        attachment_match = ATTACHMENT_PATTERN.search(content)
        is_hidden_audio = any(marker in content for marker in AUDIO_HIDDEN_MARKERS)

        final_message = content
        msg_type = "text"
        filename = None

        if attachment_match or is_hidden_audio:
            # Tenta pegar pelo nome do anexo (prioridade)
            if attachment_match:
                potential_filename = attachment_match.group("filename")
                # Verifica se existe direto
                if os.path.exists(os.path.join(INPUT_DIR, potential_filename)):
                    filename = potential_filename

            # Se não achou pelo nome (ou é áudio oculto), busca pelo timestamp (Otimizado)
            if not filename:
                filename = find_audio_by_timestamp_optimized(date_str, file_index)

            if filename:
                ext = os.path.splitext(filename)[1].lower()
                if ext in [".opus", ".ogg", ".m4a", ".wav"] or is_hidden_audio:
                    msg_type = "audio"
                    cached = db.get_transcription(filename)
                    if cached:
                        text, is_error = cached
                        final_message = f"[ÁUDIO CACHED]: {text}" if not is_error else f"[ÁUDIO INVÁLIDO]: {filename}"
                    else:
                        audio_path = os.path.join(INPUT_DIR, filename)
                        if os.path.exists(audio_path):
                            try:
                                text = transcriber.transcribe(audio_path)
                                db.save_transcription(filename, text)
                                final_message = f"[ÁUDIO TRANSCRITO]: {text}"
                            except Exception as e:
                                db.save_transcription(filename, str(e), is_error=1)
                                final_message = f"[ERRO TRANSCRIÇÃO]: {filename}"
                        else:
                            final_message = f"[ÁUDIO NÃO ENCONTRADO]: {filename}"
                elif ext in [".pdf", ".docx", ".jpg", ".jpeg", ".png"]:
                    msg_type = "attachment"
                    doc_path = os.path.join(INPUT_DIR, filename)
                    extracted = doc_processor.get_content(doc_path)
                    final_message = f"[CONTEÚDO DO ANEXO {filename}]: {extracted}"
            else:
                if is_hidden_audio:
                    final_message = f"[ÁUDIO NÃO IDENTIFICADO]: {content}"

        structured_data.append({
            "timestamp": date_str,
            "timestamp_iso": iso_date,
            "author": author,
            "message": final_message,
            "type": msg_type,
            "filename": filename
        })
        plain_text_lines.append(f"[{date_str}] {author}: {final_message}")

    if progress_callback: progress_callback(1.0)
    return structured_data, "\n".join(plain_text_lines)

def run_analysis(progress_callback=None):
    if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)
    if not os.path.exists(INPUT_DIR): os.makedirs(INPUT_DIR)

    db = DatabaseManager(os.path.join(OUTPUT_DIR, "transcription_cache.db"))
    transcriber = AudioTranscriber()
    doc_processor = DocumentProcessor(GEMINI_API_KEY, db)
    summarizer = Summarizer(GEMINI_API_KEY)

    chat_files = glob.glob(os.path.join(INPUT_DIR, "*.txt"))
    if not chat_files:
        logger.error("Nenhum arquivo .txt encontrado na pasta input.")
        return False

    for file_path in chat_files:
        base_name = os.path.splitext(os.path.basename(file_path))[0]

        # Tenta extrair ID para padronizar o nome do output
        identity_prefix = base_name
        try:
            with open(file_path, 'rb') as f:
                content = f.read()
                meta = get_chat_metadata(content)
                if meta:
                    identity_prefix = f"chat_{meta['identity']}"
                    meta_path = os.path.join(OUTPUT_DIR, f"{identity_prefix}_metadata.json")
                    with open(meta_path, 'w', encoding='utf-8') as mf:
                        json.dump(meta, mf, indent=4, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Erro ao gerar metadados para {base_name}: {e}")

        # Otimização: Pre-processa anexos em paralelo antes do loop sequencial
        if progress_callback: progress_callback(0.05, f"Pre-processando anexos de {base_name}...")
        pre_process_attachments(file_path, doc_processor)

        structured, plain_text = process_file(file_path, transcriber, doc_processor, db, progress_callback)

        with open(os.path.join(OUTPUT_DIR, f"{identity_prefix}_final.txt"), 'w', encoding='utf-8') as f:
            f.write(plain_text)

        with open(os.path.join(OUTPUT_DIR, f"{identity_prefix}_structured.json"), 'w', encoding='utf-8') as f:
            json.dump(structured, f, indent=4, ensure_ascii=False)

        summary_path = os.path.join(OUTPUT_DIR, f"{identity_prefix}_resumo.md")
        chat_mtime = os.path.getmtime(file_path)
        summary_mtime = os.path.getmtime(summary_path) if os.path.exists(summary_path) else 0

        if chat_mtime > summary_mtime:
            if progress_callback: progress_callback(0.95, f"Gerando resumo para {base_name}...")
            summary = summarizer.summarize(plain_text)
            if summary:
                with open(summary_path, 'w', encoding='utf-8') as f:
                    f.write(f"# Resumo da Conversa: {base_name}\n\n{summary}")
        else:
            logger.info(f"⏩ Resumo já está atualizado para {base_name}.")

    db.close()
    logger.info("✅ Pipeline concluído com sucesso.")
    return True

if __name__ == "__main__":
    run_analysis()
