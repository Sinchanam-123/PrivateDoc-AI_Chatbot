from dataclasses import replace

import checks
from settings import Settings

S = Settings()


def test_model_present_handles_latest_tag():
    assert checks.model_present("nomic-embed-text", ["nomic-embed-text:latest"])
    assert checks.model_present("llama3.2:3b", ["llama3.2:3b"])
    assert not checks.model_present("llama3.2:3b", ["llama3.2:1b"])
    assert not checks.model_present("nomic-embed-text", ["llama3.2:3b"])


def test_all_good_returns_no_problems():
    assert checks.run_checks(S, lambda: ["llama3.2:3b", "nomic-embed-text:latest"]) == []


def test_missing_model_gives_pull_command():
    problems = checks.run_checks(S, lambda: ["llama3.2:3b"])
    assert len(problems) == 1
    assert "ollama pull nomic-embed-text" in problems[0]["fix"]


def test_ollama_not_running():
    def boom():
        raise ConnectionError("refused")
    problems = checks.run_checks(S, boom)
    assert problems[0]["title"] == "Ollama is not running"


def test_settings_warnings_are_reported():
    s = replace(S, warnings=("top_k is larger than candidates.",))
    problems = checks.run_checks(s, lambda: ["llama3.2:3b", "nomic-embed-text"])
    assert problems[0]["title"] == "Check settings.ini"
