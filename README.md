# Private Document AI

A question-answering chatbot that runs **entirely on your own computer**. Upload PDF, Word or text files, ask questions in plain language, and get answers written only from your documents. If the documents do not contain the answer, it says so instead of guessing.

- Answers come only from retrieved document passages (Retrieval-Augmented Generation).
- A **Sources** button shows which documents, pages and passages were used.
- No cloud AI, no web search, no telemetry. The interface is bound to `127.0.0.1`.

## Requirements

- Windows 10/11, Python 3.10 or newer
- [Ollama](https://ollama.com) installed
- About 3 GB of disk space for the models
- Designed for modest hardware (NVIDIA RTX 4050, 8 GB RAM)

## Install (once)

```powershell
# 1. Download the two models
ollama pull llama3.2:3b
ollama pull nomic-embed-text

# 2. Create the project environment
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

If PowerShell blocks activation, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` first. It affects only the current window.

## Run

Double-click **`start.bat`**. It activates the environment, starts Ollama if needed, and opens http://127.0.0.1:8501. Close the window to stop the app.

Manual alternative (these use `python -m` so they also work on laptops where Windows blocks the `pip.exe` and `streamlit.exe` launchers):

```powershell
.\venv\Scripts\Activate.ps1
python -m streamlit run app.py
```

## Use

1. Click **Documents**, add files, then click **Index uploaded files**.
2. Ask a question in the chat box.
3. Click **Sources** under an answer to check the evidence behind it.

After updating the code, open **Documents** and click **Rebuild index** so every file is re-read with the latest settings.

## Settings

All tunable values live in **`settings.ini`** (models, chunk sizes, retrieval thresholds, answer length, logging). Edit, save and restart the app. Missing or invalid values fall back to safe defaults, and the app shows a message if something looks wrong.

Common changes:

| Goal | Setting |
|---|---|
| Faster on a slow laptop | `llm = llama3.2:1b` |
| Good questions get refused | raise `max_distance` (for example 0.60) |
| Unrelated questions get answered | lower `strong_distance` (for example 0.35) |
| Shorter or longer answers | `max_answer_tokens` |
| A document has the exact words but is not found | keep `hybrid_search = true` (word search runs next to meaning search) |
| Test whether the injection defences matter | set `sanitize_passages` or `block_override_requests` to `false`, rerun `python evaluate.py` |

## Start-up checks

On launch the app checks that Ollama is running and that both models are downloaded. If not, it shows what to do (for example `ollama pull llama3.2:3b`) with a **Check again** button.

## Retrieval log

Each question writes one line to `logs/queries.jsonl`: the question, whether it was refused, the time taken, and for every candidate passage its file, page, distance, keyword overlap and what happened to it. **No document text is stored.** Turn it off with `enabled = false` in `settings.ini`, or hide questions with `log_questions = false`.

Read it with:

```powershell
python view_log.py        # last 10 questions
python view_log.py 25     # last 25
```

## Feedback buttons

Under every answer are 👍 and 👎 buttons. Clicks are saved to `logs/feedback.jsonl` on this computer. Run `python feedback_to_eval.py` to turn them into draft questions in `evaluation/candidates.csv`, which you can review and add to the evaluation set.

## Evaluation

`evaluation/questions.csv` holds questions with known answers: ones the documents answer, ones that must be refused, and prompt-injection tests. With Ollama running and the test documents indexed:

```powershell
python evaluate.py                    # run everything
python evaluate.py --type refuse      # only the must-refuse questions
python evaluate.py --only A01,A08     # only these ids
python evaluate.py --file evaluation\\questions_100.csv   # the larger 108-question set
```

It prints accuracy, the false-refusal rate, the correct-refusal rate, the injection pass rate, how often the right document was retrieved, and the time per question. Every failure is labelled with its cause (`retrieval_miss`, `generation_refusal`, `generation_error`, `unsupported_answer` or `injection_leak`). Each run is saved in `evaluation/results/` and compared with the previous run, so you can see exactly which questions a change fixed or broke.

Upload `evaluation/07_Injection_Test.txt` with the Documents button before running the injection questions. See `evaluation/README.md` for the question file format.

## Tests

```powershell
python -m pip install -r requirements-dev.txt
python -m pytest
```

The tests cover chunking, text extraction (TXT, DOCX, PDF), indexing and deleting, the evidence checks and re-ranking, refusal handling, start-up checks, settings, logging, feedback and the evaluation scoring. They use fake embeddings and a temporary database, so they do not need Ollama and do not touch your documents.

## Privacy

- Documents, vector database and model inference stay on this computer.
- ChromaDB and Streamlit telemetry are off. Fonts are system fonts, so nothing is downloaded.
- `data/` (documents and index), `logs/` and `evaluation/results/` are excluded from Git by `.gitignore`.

## Project layout

```
app.py             Streamlit interface
rag.py             retrieval, evidence checks, generation
ingest.py          extraction, chunking, embedding, indexing
settings.py        reads settings.ini
settings.ini       tunable values
checks.py          start-up checks
qlog.py            retrieval log
view_log.py        prints the log
evalkit.py         scoring logic for the evaluation (unit-tested)
evaluate.py        runs the evaluation set and saves the results
feedback_to_eval.py  turns thumbs up/down clicks into draft questions
evaluation/        question set, injection test document, results
start.bat          one-click launcher
tests/             unit tests
.streamlit/        Streamlit configuration (localhost, telemetry off)
data/              documents and vector database (not committed)
logs/              retrieval log (not committed)
```

## Troubleshooting

| Problem | Fix |
|---|---|
| "Ollama is not running" | Open the Ollama app, or run `ollama serve`, then click **Check again** |
| "Model ... is not downloaded" | Run the `ollama pull ...` command shown |
| Slow or sluggish | Set `llm = llama3.2:1b` in `settings.ini`; use **Unload model** in the Documents popup |
| Scanned PDF shows "no text found" | Scanned pages need OCR, which is not included yet |
| Answers wrong after updating | Click **Rebuild index** in the Documents popup |

## Limitations

A 3B model can occasionally phrase a detail imprecisely, so use **Sources** to verify. Thresholds were tuned on a small fictional test set and may need adjusting for other documents. DOCX and TXT files are cited by "section" because they have no fixed pages.
