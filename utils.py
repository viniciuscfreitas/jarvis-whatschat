import re
import logging
from datetime import datetime

# Patterns
COMBINED_PATTERN = re.compile(
    r"^(?:\[(?P<ios_date>\d{2}/\d{2}/\d{4}, \d{1,2}:\d{2}:\d{2}.*?)\] (?P<ios_author>.*?): |"
    r"(?P<and_date>\d{1,2}/\d{1,2}/\d{2,4}, \d{1,2}:\d{2}) - (?P<and_author>.*?): )"
    r"(?P<content>.*)"
)

ATTACHMENT_PATTERN = re.compile(r"(?:<anexado: |)(?P<filename>.*?\.(?:opus|ogg|m4a|wav|pdf|docx|jpg|jpeg|png))(?:>| \(file attached\))")

FILE_TIMESTAMP_PATTERNS = [
    re.compile(r"PTT-(?P<date>\d{8})-WA"),
    re.compile(r"WhatsApp Ptt (?P<date>\d{4}-\d{2}-\d{2}) at (?P<h>\d{2})\.(?P<m>\d{2})"),
    re.compile(r"IMG-(?P<date>\d{8})-WA"),
    re.compile(r"DOC-(?P<date>\d{8})-WA"),
    re.compile(r"VID-(?P<date>\d{8})-WA")
]

DATE_FORMATS = [
    "%d/%m/%Y, %I:%M:%S %p",
    "%d/%m/%Y, %H:%M:%S",
    "%m/%d/%y, %H:%M",
    "%d/%m/%y, %H:%M",
    "%Y-%m-%d %H:%M:%S",
    "%d.%m.%y %H:%M:%S",
    "%d/%m/%Y %H:%M"
]

def sanitize_line(line):
    if not line: return ""
    # Remove caracteres invisíveis do WhatsApp (LTR/RTL marks) e normaliza espaços
    line = line.replace('\u200e', '').replace('\u200f', '').replace('\u202f', ' ').replace('\xa0', ' ')
    return line.strip()

def clean_extra_whitespace(text):
    if not text: return ""
    # Remove múltiplas quebras de linha e espaços sobrando
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()

def parse_log_date(date_str, detected_fmt=None):
    clean_date = sanitize_line(date_str)
    if detected_fmt:
        try:
            return datetime.strptime(clean_date, detected_fmt), detected_fmt
        except ValueError:
            pass

    for fmt in DATE_FORMATS:
        try:
            dt = datetime.strptime(clean_date, fmt)
            return dt, fmt
        except ValueError:
            continue
    return None, None
