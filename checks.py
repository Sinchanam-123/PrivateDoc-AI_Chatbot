"""checks.py - start-up checks with friendly messages."""
import ollama

from settings import cfg


def installed_models():
    """Names of the models Ollama has downloaded (works across ollama versions)."""
    resp = ollama.list()
    models = getattr(resp, "models", None)
    if models is None:
        models = resp.get("models", [])
    names = []
    for m in models:
        if isinstance(m, dict):
            name = m.get("model") or m.get("name")
        else:
            name = getattr(m, "model", None) or getattr(m, "name", None)
        if name:
            names.append(name)
    return names


def model_present(required: str, installed) -> bool:
    """'nomic-embed-text' also matches 'nomic-embed-text:latest'."""
    for name in installed:
        if name == required:
            return True
        if ":" not in required and name.startswith(required + ":"):
            return True
    return False


def run_checks(s=cfg, lister=installed_models):
    """Return a list of {'title', 'fix'} problems. An empty list means all good."""
    problems = []
    try:
        installed = lister()
    except Exception:
        return [{
            "title": "Ollama is not running",
            "fix": "Open the Ollama app from the Start menu (or run: ollama serve), "
                   "wait a few seconds, then click Check again.",
        }]
    for model in (s.llm_model, s.embed_model):
        if not model_present(model, installed):
            problems.append({
                "title": f"The model '{model}' is not downloaded",
                "fix": f"Run this in PowerShell, then click Check again:\nollama pull {model}",
            })
    for note in s.warnings:
        problems.append({"title": "Check settings.ini", "fix": note})
    return problems
