"""view_log.py - print the latest logged questions in a readable form.

Usage (inside the venv):   python view_log.py        (last 10)
                           python view_log.py 25     (last 25)
"""
import sys

from qlog import read_log

n = int(sys.argv[1]) if len(sys.argv) > 1 else 10
entries = read_log()[-n:]
if not entries:
    print("No log entries yet. Ask a question in the app first.")
for e in entries:
    status = "REFUSED" if e["refused"] else "answered"
    print(f"\n{e['time']}  [{status}]  {e['seconds']}s  model={e['model']}")
    print(f"Q: {e['question']}")
    for c in e["candidates"]:
        print(f"   dist {c.get('distance', 0):.3f}  keywords {c.get('overlap', 0):.2f}  {c.get('decision', ''):<22} {c.get('source')} ({c.get('page')})")
