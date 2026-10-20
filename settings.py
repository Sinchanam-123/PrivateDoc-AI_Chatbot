"""settings.py - read tunable values from settings.ini (with safe defaults)."""
import configparser
from dataclasses import dataclass
from pathlib import Path

SETTINGS_FILE = Path(__file__).parent / "settings.ini"


@dataclass(frozen=True)
class Settings:
    # models
    llm_model: str = "llama3.2:3b"
    embed_model: str = "nomic-embed-text"
    # chunking
    chunk_size: int = 800
    chunk_overlap: int = 100
    section_chars: int = 3000
    # retrieval
    candidates: int = 8
    top_k: int = 5
    max_distance: float = 0.55
    strong_distance: float = 0.40
    distance_margin: float = 0.12
    keyword_weight: float = 0.3
    # generation
    context_tokens: int = 3072
    max_answer_tokens: int = 450
    temperature: float = 0.0
    keep_alive: str = "15m"
    # logging
    logging_enabled: bool = True
    log_questions: bool = True
    # problems found while reading the file (shown at start-up)
    warnings: tuple = ()


# field name -> (section, key in settings.ini)
_SECTIONS = {
    "llm_model": ("models", "llm"),
    "embed_model": ("models", "embedding"),
    "chunk_size": ("chunking", "chunk_size"),
    "chunk_overlap": ("chunking", "chunk_overlap"),
    "section_chars": ("chunking", "section_chars"),
    "candidates": ("retrieval", "candidates"),
    "top_k": ("retrieval", "top_k"),
    "max_distance": ("retrieval", "max_distance"),
    "strong_distance": ("retrieval", "strong_distance"),
    "distance_margin": ("retrieval", "distance_margin"),
    "keyword_weight": ("retrieval", "keyword_weight"),
    "context_tokens": ("generation", "context_tokens"),
    "max_answer_tokens": ("generation", "max_answer_tokens"),
    "temperature": ("generation", "temperature"),
    "keep_alive": ("generation", "keep_alive"),
    "logging_enabled": ("logging", "enabled"),
    "log_questions": ("logging", "log_questions"),
}


def validate(s: Settings):
    """Return a list of human-readable problems with the values."""
    problems = []
    if s.chunk_overlap >= s.chunk_size:
        problems.append("chunk_overlap must be smaller than chunk_size.")
    if s.strong_distance > s.max_distance:
        problems.append("strong_distance must not be larger than max_distance.")
    if s.top_k < 1 or s.candidates < 1:
        problems.append("top_k and candidates must be at least 1.")
    if s.top_k > s.candidates:
        problems.append("top_k is larger than candidates, so some slots can never be filled.")
    if not 0 <= s.max_distance <= 2:
        problems.append("max_distance should be between 0 and 2 (cosine distance).")
    return problems


def load_settings(path=SETTINGS_FILE) -> Settings:
    parser = configparser.ConfigParser(inline_comment_prefixes=("#", ";"))
    notes = []
    path = Path(path)
    if path.exists():
        try:
            parser.read(path, encoding="utf-8")
        except configparser.Error as e:
            notes.append(f"settings.ini could not be read ({e}). Defaults are in use.")
    defaults = Settings()
    values = {}
    for key, (section, ini_key) in _SECTIONS.items():
        default = getattr(defaults, key)
        try:
            if isinstance(default, bool):
                values[key] = parser.getboolean(section, ini_key, fallback=default)
            elif isinstance(default, int):
                values[key] = parser.getint(section, ini_key, fallback=default)
            elif isinstance(default, float):
                values[key] = parser.getfloat(section, ini_key, fallback=default)
            else:
                values[key] = parser.get(section, ini_key, fallback=default).strip()
        except ValueError:
            notes.append(f"'{ini_key}' in settings.ini has an invalid value. The default ({default}) is in use.")
            values[key] = default
    s = Settings(**values)
    notes.extend(validate(s))
    return Settings(**{**values, "warnings": tuple(notes)})


cfg = load_settings()
