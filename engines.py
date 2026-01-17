import os
import sqlite3
import logging
import hashlib
import time
from faster_whisper import WhisperModel
from google import genai
from google.genai import types
import docx
from pypdf import PdfReader

import concurrent.futures
from utils import clean_extra_whitespace

logger = logging.getLogger("WhatsAppETL.Engines")

class AudioTranscriber:
    def __init__(self, model_size, device, compute_type):
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.model = None

    def _load_model(self):
        if self.model is None:
            logger.info(f"🐢 Carregando Whisper ({self.model_size}) em {self.device}...")
            self.model = WhisperModel(
                self.model_size,
                device=self.device,
                compute_type=self.compute_type,
                download_root=os.path.join(os.getcwd(), "models")
            )

    def transcribe(self, audio_path):
        self._load_model()
        # beam_size=5 é o ideal para máxima precisão (exigência do usuário)
        # vad_filter=True continua ativo para evitar alucinações em silêncios
        segments, _ = self.model.transcribe(
            audio_path,
            beam_size=5,
            language="pt",
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=500)
        )
        return " ".join([s.text for s in segments]).strip()

class DatabaseManager:
    def __init__(self, db_path):
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.cursor = self.conn.cursor()
        self._setup_db()

    def _setup_db(self):
        # WAL mode e PRAGMAs para performance máxima em SQLite
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self.conn.execute("PRAGMA cache_size=-64000") # 64MB de cache em RAM

        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS transcriptions (
                filename TEXT PRIMARY KEY,
                text TEXT,
                is_error INTEGER DEFAULT 0,
                processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS document_cache (
                hash TEXT PRIMARY KEY,
                filename TEXT,
                text_content TEXT,
                processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        self.conn.commit()

    def get_transcription(self, filename):
        self.cursor.execute("SELECT text, is_error FROM transcriptions WHERE filename = ?", (filename,))
        return self.cursor.fetchone()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            self.conn.commit()
        else:
            self.conn.rollback()

    def save_transcription(self, filename, text, is_error=0, commit=True):
        self.cursor.execute("INSERT OR REPLACE INTO transcriptions (filename, text, is_error) VALUES (?, ?, ?)", (filename, text, is_error))
        if commit:
            self.conn.commit()

    def get_document_cache(self, file_hash):
        self.cursor.execute("SELECT text_content FROM document_cache WHERE hash = ?", (file_hash,))
        result = self.cursor.fetchone()
        return result[0] if result else None

    def save_document_cache(self, file_hash, filename, text_content, commit=True):
        self.cursor.execute("INSERT OR REPLACE INTO document_cache (hash, filename, text_content) VALUES (?, ?, ?)", (file_hash, filename, text_content))
        if commit:
            self.conn.commit()

    def get_all_transcriptions(self):
        # Retorna todas as transcrições em um dicionário para busca instantânea
        self.cursor.execute("SELECT filename, text, is_error FROM transcriptions")
        return {row[0]: (row[1], row[2]) for row in self.cursor.fetchall()}

    def get_all_document_cache(self):
        # Retorna todo o cache de documentos
        self.cursor.execute("SELECT hash, text_content FROM document_cache")
        return {row[0]: row[1] for row in self.cursor.fetchall()}

    def close(self):
        self.conn.close()

class Summarizer:
    def __init__(self, api_key, model_id):
        self.enabled = False
        self.model_id = model_id
        if api_key and api_key != "seu_token_aqui":
            self.client = genai.Client(api_key=api_key)
            self.enabled = True
        else:
            logger.warning("⚠️ GEMINI_API_KEY não configurada. O resumo será pulado.")

    def summarize(self, file_path):
        if not self.enabled: return None
        logger.info(f"🧠 Gerando resumo hierárquico via {self.model_id}...")

        try:
            chunks = []
            chunk_size = 200000 # ~200k chars por bloco

            # 1. Leitura em streaming para evitar carregar o arquivo todo na RAM
            with open(file_path, 'r', encoding='utf-8') as f:
                while True:
                    chunk = f.read(chunk_size)
                    if not chunk: break
                    chunks.append(chunk)

            if not chunks: return None
            if len(chunks) == 1:
                return self._generate(f"Resuma esta conversa com o máximo de detalhes, fatos e cronologia. IMPORTANTE: Sempre que citar um fato crítico (valores, datas, doenças), cite o ID da mensagem correspondente no formato [ID:X] (ex: [ID:123]). Mantenha as referências a arquivos de ANEXO mencionados:\n\n{chunks[0]}")

            # 2. Extração Paralela (Map) preservando a ORDEM
            logger.info(f"⚡ Chat grande ({len(chunks)} partes). Extraindo detalhes em paralelo...")
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                # Reduzido para 2 workers e adicionado delay entre requests para respeitar cota de 10 RPM
                results = []
                for chunk_idx, chunk in enumerate(chunks):
                    prompt = f"Extraia todos os fatos, nomes, valores e decisões deste trecho de conversa (PARTE {chunk_idx+1}). Sempre que possível, inclua o [ID:X] da mensagem original para cada fato extraído:\n\n{chunk}"
                    results.append(executor.submit(self._generate, prompt))
                    time.sleep(1.5) # Delay de segurança entre disparos

                partial_results = [r.result() for r in results if r.result()]

            # Remover falhas (None)
            valid_results = [r for r in partial_results if r]

            # 3. Síntese Final (Reduce)
            combined_context = "\n\n--- CRONOLOGIA CONTINUA NA PRÓXIMA PARTE ---\n\n".join(valid_results)
            return self._generate(f"Sintetize estes fatos em um relatório jurídico cronológico final. NÃO OMITA NENHUM DETALHE, nome, data ou valor mencionado. Use referências cruzadas citando os [ID:X] originais e os ANEXOS correspondentes:\n\n{combined_context}")

        except Exception as e:
            logger.error(f"Erro no resumo hierárquico: {e}")
            return f"Erro ao gerar resumo: {e}"

    def _generate(self, prompt, retries=3):
        for attempt in range(retries):
            try:
                response = self.client.models.generate_content(
                    model=self.model_id,
                    contents=prompt
                )
                return response.text
            except Exception as e:
                err_msg = str(e)
                if ("429" in err_msg or "RESOURCE_EXHAUSTED" in err_msg) and attempt < retries - 1:
                    wait_time = 30 * (attempt + 1)
                    time.sleep(wait_time)
                    continue
                return None

class DocumentProcessor:
    def __init__(self, api_key, db, model_id):
        self.db = db
        self.model_id = model_id
        self.enabled = False
        if api_key and api_key != "seu_token_aqui":
            self.client = genai.Client(api_key=api_key)
            self.enabled = True
        else:
            logger.warning("⚠️ GEMINI_API_KEY não configurada para DocumentProcessor.")

    def get_file_hash(self, file_path):
        # Otimização: Hash rápido usando metadados antes de ler o arquivo todo
        stats = os.stat(file_path)
        quick_meta = f"{file_path}_{stats.st_size}_{stats.st_mtime}"
        return hashlib.md5(quick_meta.encode()).hexdigest()

    def extract_text_from_pdf(self, file_path):
        try:
            reader = PdfReader(file_path)
            if reader.is_encrypted:
                return "[PDF PROTEGIDO POR SENHA - TEXTO NÃO EXTRAÍDO]"
            text = ""
            for page in reader.pages:
                page_text = page.extract_text() or ""
                text += page_text + "\n"
            return clean_extra_whitespace(text)
        except Exception as e:
            logger.error(f"Erro ao ler PDF {file_path}: {e}")
            return f"[ERRO LEITURA PDF: {e}]"

    def extract_text_from_docx(self, file_path):
        try:
            doc = docx.Document(file_path)
            # Filtra parágrafos vazios para evitar buracos no log
            text = "\n".join([para.text for para in doc.paragraphs if para.text.strip()]).strip()
            return clean_extra_whitespace(text)
        except Exception as e:
            logger.error(f"Erro ao ler DOCX {file_path}: {e}")
            return ""

    def process_with_gemini(self, file_path, prompt, retries=3):
        if not self.enabled: return "[GEMINI DISABLED]"
        try:
            with open(file_path, "rb") as f:
                file_data = f.read()

            ext = os.path.splitext(file_path)[1].lower()
            mime_map = {".png": "image/png", ".pdf": "application/pdf"}
            mime_type = mime_map.get(ext, "image/jpeg")

            for attempt in range(retries):
                try:
                    response = self.client.models.generate_content(
                        model=self.model_id,
                        contents=[prompt, types.Part.from_bytes(data=file_data, mime_type=mime_type)]
                    )
                    return response.text
                except Exception as e:
                    err_msg = str(e)
                    if "429" in err_msg or "RESOURCE_EXHAUSTED" in err_msg:
                        wait_time = (attempt + 1) * 20  # Backoff progressivo
                        logger.warning(f"⚠️ Gemini Quota (429). Esperando {wait_time}s (tentativa {attempt+1}/{retries})...")
                        time.sleep(wait_time)
                        continue
                    if attempt == retries - 1: raise e
                    time.sleep(5)
        except Exception as e:
            logger.error(f"Erro no Gemini para {file_path}: {e}")
            return f"[ERRO GEMINI]: {e}"

    def get_content(self, file_path, commit=True):
        if not os.path.exists(file_path): return "[ARQUIVO NÃO ENCONTRADO]"
        file_hash = self.get_file_hash(file_path)
        cached = self.db.get_document_cache(file_hash)
        if cached: return cached

        ext = os.path.splitext(file_path)[1].lower()
        content = ""
        if ext == ".pdf":
            content = self.extract_text_from_pdf(file_path)
            if len(content) < 100:
                content = self.process_with_gemini(file_path, "Extraia todo o texto deste PDF. Se houver imagens de documentos, descreva-os detalhadamente.")
        elif ext == ".docx":
            content = self.extract_text_from_docx(file_path)
        elif ext in [".jpg", ".jpeg", ".png"]:
            content = self.process_with_gemini(file_path, "Descreva esta imagem detalhadamente para um caso jurídico. Se for um documento ou print, extraia todo o texto visível.")

        if content:
            self.db.save_document_cache(file_hash, os.path.basename(file_path), content, commit=commit)
            return content
        return "[CONTEÚDO NÃO EXTRAÍDO]"
