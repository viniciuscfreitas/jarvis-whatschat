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

# --- CONFIGURAÇÃO E CONSTANTES ---
load_dotenv()

INPUT_DIR = os.getenv("INPUT_DIR", "input")
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "output")
MODEL_SIZE = os.getenv("MODEL_SIZE", "large-v3-turbo")
DEVICE = os.getenv("DEVICE", "cuda")
COMPUTE_TYPE = os.getenv("COMPUTE_TYPE", "float16")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Regex Híbrido (24h e AM/PM) para iOS
IOS_MSG_PATTERN = re.compile(r"^\[(?P<date>\d{2}/\d{2}/\d{4}, \d{1,2}:\d{2}:\d{2}.*?)\] (?P<author>.*?): (?P<content>.*)")

# Regex para detectar anexo de áudio no conteúdo da mensagem (formato <anexado: ...>)
AUDIO_ATTACHMENT_PATTERN = re.compile(r"<anexado: (?P<filename>.*?\.(opus|ogg|m4a|wav))>")

# Configuração de Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("WhatsAppETL")

# Marcadores de áudio sem nome de arquivo (comum em exportações de iPhone)
AUDIO_HIDDEN_MARKERS = ["áudio ocultado", "audio ocultado", "Anexo de áudio"]

def get_chat_metadata(file_content):
    """Extrai informações de identidade e exibição do chat."""
    lines = [l.strip() for l in file_content.decode('utf-8').split('\n') if l.strip()]
    if not lines: return None

    # Identidade: Baseada na primeira mensagem (quase impossível dois chats diferentes começarem igual)
    first_msg = lines[0]
    identity = hashlib.md5(first_msg.encode()).hexdigest()[:10]

    # Participantes (amostragem das primeiras 100 linhas)
    participants = set()
    for line in lines[:100]:
        match = IOS_MSG_PATTERN.match(sanitize_line(line))
        if match:
            participants.add(match.group("author"))

    # Data da última mensagem
    last_msg_date = "Data desconhecida"
    for line in reversed(lines):
        match = IOS_MSG_PATTERN.match(sanitize_line(line))
        if match:
            last_msg_date = match.group("date")
            break

    title = f"{', '.join(list(participants)[:2])}... ({last_msg_date})"
    return {
        "identity": identity,
        "title": title,
        "last_date": last_msg_date,
        "hash": hashlib.md5(file_content).hexdigest() # Hash total para detectar mudanças de conteúdo
    }

def find_audio_by_timestamp(date_str, files):
    """Tenta encontrar um áudio correspondente ao timestamp usando a lista de arquivos fornecida."""
    try:
        dt = None
        # Limpeza extra para o timestamp do iOS
        clean_date = date_str.replace('\u202f', ' ').replace('\xa0', ' ').strip()

        for fmt in ["%d/%m/%Y, %I:%M:%S %p", "%d/%m/%Y, %H:%M:%S"]:
            try:
                dt = datetime.strptime(clean_date, fmt)
                break
            except ValueError:
                continue

        if not dt:
            return None

        # 1. Busca exata (já existente)
        date_part = dt.strftime("%Y-%m-%d")
        h_with_zero = dt.strftime("%I")
        h_no_zero = str(int(h_with_zero))
        m = dt.strftime("%M")
        s = dt.strftime("%S")
        p = dt.strftime("%p")

        for h in [h_no_zero, h_with_zero]:
            target_pattern = f"WhatsApp Ptt {date_part} at {h}.{m}.{s} {p}".lower()
            for f in files:
                if f.lower().startswith(target_pattern) and f.lower().endswith(('.ogg', '.opus', '.m4a', '.wav')):
                    return f

        # 2. Busca aproximada (mesmo minuto, ignorando segundos)
        target_minute = f"WhatsApp Ptt {date_part} at {h_no_zero}.{m}".lower()
        target_minute_alt = f"WhatsApp Ptt {date_part} at {h_with_zero}.{m}".lower()

        for f in files:
            f_low = f.lower()
            if (f_low.startswith(target_minute) or f_low.startswith(target_minute_alt)) and \
               p.lower() in f_low and \
               f_low.endswith(('.ogg', '.opus', '.m4a', '.wav')):
                return f

        # 3. Formato Android PTT-YYYYMMDD-WA (apenas por data se não tiver outro jeito)
        date_alt = dt.strftime("%Y%m%d")
        target_pattern_alt = f"PTT-{date_alt}-WA".lower()
        for f in files:
            if f.lower().startswith(target_pattern_alt) and f.lower().endswith(('.ogg', '.opus', '.m4a', '.wav')):
                return f

    except Exception as e:
        logger.error(f"Erro em find_audio_by_timestamp: {e}")
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
        self.conn.commit()

    def get_transcription(self, filename):
        self.cursor.execute("SELECT text, is_error FROM transcriptions WHERE filename = ?", (filename,))
        return self.cursor.fetchone()

    def save_transcription(self, filename, text, is_error=0):
        self.cursor.execute("INSERT OR REPLACE INTO transcriptions (filename, text, is_error) VALUES (?, ?, ?)", (filename, text, is_error))
        self.conn.commit()

    def close(self):
        self.conn.close()

class Summarizer:
    """Gera resumos das conversas usando o Google Gemini 3.0 Flash."""
    def __init__(self, api_key):
        self.enabled = False
        if api_key and api_key != "seu_token_aqui" and api_key:
            self.client = genai.Client(api_key=api_key)
            self.model_id = "gemini-3-flash-preview"
            self.enabled = True
        else:
            logger.warning("⚠️ GEMINI_API_KEY não configurada. O resumo será pulado.")

    def summarize(self, text):
        if not self.enabled: return None
        logger.info(f"🧠 Gerando resumo via {self.model_id}...")
        prompt = f"Resuma os pontos principais desta conversa de WhatsApp de forma executiva e em português:\n\n{text[:100000]}"
        try:
            response = self.client.models.generate_content(
                model=self.model_id,
                contents=prompt
            )
            return response.text
        except Exception as e:
            logger.error(f"Erro ao gerar resumo: {e}")
            return f"Erro ao gerar resumo: {e}"

def sanitize_line(line):
    return line.replace('\u200e', '').replace('\u200f', '').replace('\u202f', ' ').replace('\xa0', ' ').strip()

def parse_log_date(date_str):
    """Converte a string de data do log para um objeto datetime."""
    clean_date = date_str.replace('\u202f', ' ').replace('\xa0', ' ').strip()
    for fmt in ["%d/%m/%Y, %I:%M:%S %p", "%d/%m/%Y, %H:%M:%S"]:
        try:
            return datetime.strptime(clean_date, fmt)
        except ValueError:
            continue
    return None

def process_file(file_path, transcriber, db, progress_callback=None):
    logger.info(f"📄 Processando: {os.path.basename(file_path)}")
    input_files = os.listdir(INPUT_DIR)
    with open(file_path, 'r', encoding='utf-8') as f:
        # Validação básica: as primeiras 5 linhas devem ter o padrão de mensagem
        sample_lines = [f.readline() for _ in range(5)]
        f.seek(0)
        lines = f.readlines()

    is_valid = any(IOS_MSG_PATTERN.match(sanitize_line(l)) for l in sample_lines if l.strip())
    if not is_valid and len(lines) > 0:
        logger.error(f"❌ Arquivo inválido ou formato incompatível: {file_path}")
        return [], ""

    structured_data = []
    plain_text_lines = []
    total_lines = len(lines)

    for i, line in enumerate(tqdm(lines, desc="Linhas", leave=False)):
        if progress_callback:
            progress_callback(i / total_lines)

        clean_line = sanitize_line(line)
        match = IOS_MSG_PATTERN.match(clean_line)
        if not match: continue

        date_str = match.group("date")
        author = match.group("author")
        content = match.group("content")

        dt_obj = parse_log_date(date_str)
        iso_date = dt_obj.isoformat() if dt_obj else None

        audio_match = AUDIO_ATTACHMENT_PATTERN.search(content)
        is_hidden_audio = any(marker in content for marker in AUDIO_HIDDEN_MARKERS)

        final_message = content
        msg_type = "text"
        filename = None

        if audio_match or is_hidden_audio:
            msg_type = "audio"
            filename = audio_match.group("filename") if audio_match else find_audio_by_timestamp(date_str, input_files)

            if filename:
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
            else:
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

    if progress_callback:
        progress_callback(1.0)

    return structured_data, "\n".join(plain_text_lines)

def run_analysis(progress_callback=None):
    """Função principal para ser chamada via dashboard ou terminal."""
    if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)
    if not os.path.exists(INPUT_DIR): os.makedirs(INPUT_DIR)

    db = DatabaseManager(os.path.join(OUTPUT_DIR, "transcription_cache.db"))
    transcriber = AudioTranscriber()
    summarizer = Summarizer(GEMINI_API_KEY)

    chat_files = glob.glob(os.path.join(INPUT_DIR, "*.txt"))
    if not chat_files:
        logger.error("Nenhum arquivo .txt encontrado na pasta input.")
        return False

    for file_path in chat_files:
        base_name = os.path.splitext(os.path.basename(file_path))[0]

        # Se o arquivo for do tipo chat_IDENTITY.txt, extrai metadados para garantir que o .json exista
        if base_name.startswith("chat_"):
            try:
                with open(file_path, 'rb') as f:
                    content = f.read()
                    meta = get_chat_metadata(content)
                    if meta:
                        meta_path = os.path.join(OUTPUT_DIR, f"{base_name}_metadata.json")
                        with open(meta_path, 'w', encoding='utf-8') as mf:
                            json.dump(meta, mf, indent=4, ensure_ascii=False)
            except Exception as e:
                logger.error(f"Erro ao atualizar metadados para {base_name}: {e}")

        structured, plain_text = process_file(file_path, transcriber, db, progress_callback)

        with open(os.path.join(OUTPUT_DIR, f"{base_name}_final.txt"), 'w', encoding='utf-8') as f:
            f.write(plain_text)

        with open(os.path.join(OUTPUT_DIR, f"{base_name}_structured.json"), 'w', encoding='utf-8') as f:
            json.dump(structured, f, indent=4, ensure_ascii=False)

        # Resumo Inteligente (Atualiza se o arquivo de entrada for mais novo que o resumo)
        summary_path = os.path.join(OUTPUT_DIR, f"{base_name}_resumo.md")
        chat_mtime = os.path.getmtime(file_path)
        summary_mtime = os.path.getmtime(summary_path) if os.path.exists(summary_path) else 0

        if chat_mtime > summary_mtime:
            if progress_callback: progress_callback(0.95, f"Gerando novo resumo para {base_name}...")
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
