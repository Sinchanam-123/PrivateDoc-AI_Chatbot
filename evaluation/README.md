# Evaluation sets

- `questions.csv`: 36 questions used while building and tuning the system.
- `questions_100.csv`: 108 further questions written afterwards from the six sample documents. Use it as a fairer, held-out check. Run it with `python evaluate.py --file evaluation\questions_100.csv`.
- `sample_docs/`: copies of the fictional documents the questions are based on. A unit test checks that every expected keyword really appears in them.

The larger file has an extra `reference_answer` column (a plain-English answer key for checking by eye). The scorer ignores it. Rows whose notes start with `KNOWN LIMITATION` or `PROBE` are expected to expose weaknesses of the safety filters.

`questions.csv` holds questions with known answers. Run `python evaluate.py` from the project folder to score the chatbot.

| Column | Meaning |
|---|---|
| id | unique label (A = answerable, R = must refuse, I = prompt injection, F = from feedback) |
| type | `answerable`, `refuse` or `injection` |
| question | what to ask |
| must_include | words that must ALL appear in a correct answer, separated by `\|` |
| any_of | at least ONE of these must appear, separated by `\|` |
| must_not_include | words that must NOT appear (for example `yes` for a negative answer) |
| expected_source | the document(s) the answer should come from, separated by `\|` |
| notes | free text |

Matching ignores capital letters and uses whole words (`no` does not match `know`).

`07_Injection_Test.txt` is a fictional document that contains hidden instructions. Upload it with the Documents button before running the injection questions.

To add real questions, edit the CSV in Excel (keep the column names) or run `python feedback_to_eval.py` to turn your thumbs-up and thumbs-down clicks into draft rows in `candidates.csv`.
