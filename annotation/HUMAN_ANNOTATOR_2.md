# Second human annotator

1. Read [GUIDELINES.md](GUIDELINES.md) for the five phenomenon definitions and examples.
2. Read the 300 sentences in [diagnostic_items.csv](../data/diagnostic/diagnostic_items.csv) and tag them independently. Keep the annotation blind. Leave the other annotation files, rule tags, diagnostic key, and results closed until you finish.
3. Fill in a copy of [annotations_human2_template.csv](../data/diagnostic/annotations_human2_template.csv). Keep the header and all 300 IDs. Enter `1` when a phenomenon is present and `0` otherwise. If unsure, enter `0` and add a short note without commas. Leave unused notes blank.
4. Save the completed UTF-8 CSV as `data/diagnostic/annotations_human2.csv`. From the repository root, run `python3 scripts/analyze_human_annotation.py` to compute the agreement results.
