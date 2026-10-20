from dataclasses import fields

from settings import SETTINGS_FILE, Settings, load_settings, validate


def test_defaults_when_file_missing(tmp_path):
    s = load_settings(tmp_path / "nope.ini")
    assert s == Settings()


def test_values_are_read_and_inline_comments_ignored(tmp_path):
    f = tmp_path / "s.ini"
    f.write_text("[retrieval]\ntop_k = 3   # fewer chunks\nmax_distance = 0.5\n[models]\nllm = llama3.2:1b\n")
    s = load_settings(f)
    assert s.top_k == 3
    assert s.max_distance == 0.5
    assert s.llm_model == "llama3.2:1b"
    assert s.chunk_size == 800  # untouched values keep defaults


def test_invalid_number_falls_back_with_warning(tmp_path):
    f = tmp_path / "s.ini"
    f.write_text("[retrieval]\ntop_k = many\n")
    s = load_settings(f)
    assert s.top_k == Settings().top_k
    assert any("top_k" in w for w in s.warnings)


def test_validate_flags_bad_combinations():
    bad = Settings(chunk_size=100, chunk_overlap=100, strong_distance=0.9, max_distance=0.5, top_k=9, candidates=4)
    problems = validate(bad)
    assert len(problems) >= 3


def test_shipped_settings_ini_matches_defaults():
    """settings.ini and the code defaults must never drift apart."""
    shipped = load_settings(SETTINGS_FILE)
    defaults = Settings()
    for f in fields(Settings):
        if f.name == "warnings":
            continue
        assert getattr(shipped, f.name) == getattr(defaults, f.name), f.name
    assert shipped.warnings == ()
