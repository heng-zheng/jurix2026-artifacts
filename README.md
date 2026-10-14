# JURIX 2026 Artifacts

Paper: LLM-assisted Stance Detection for US Supreme Court Oral Arguments Analysis

Code, prompts, and a redacted annotation dataset for SEC v. Jarkesy (No. 22-859). The dataset contains 212 utterances and 39 columns.

## Directory Structure

```text
jurix2026-artifacts/
  README.md
  dataset/utterance_annotations.csv
  pipeline/
  prompts/
  config/
  tests/
  requirements.txt
  .env.example
  .gitignore
```

## Code Versions

| Code tag | Used for |
| --- | --- |
| `v1.2.3` | Stage 1 |
| `v1.2.4` | Stages 2 and 3 |

## Source Materials

- [Oyez case metadata](https://api.oyez.org/cases/2023/22-859)
- [Oyez oral-argument JSON, ID 25500](https://api.oyez.org/case_media/oral_argument_audio/25500)
- Petitioner's merits brief: [20230828183013049_22-859tsUnitedStates.pdf](https://www.supremecourt.gov/DocketPDF/22/22-859/278330/20230828183013049_22-859tsUnitedStates.pdf)
- Petitioner's reply brief: [20231113164043876_22-859 SEC v. Jarkesy.pdf](https://www.supremecourt.gov/DocketPDF/22/22-859/289467/20231113164043876_22-859%20SEC%20v.%20Jarkesy.pdf)

Target candidates comprise all heading levels under “Argument” in the two briefs' tables of contents: 19 merits headings and 8 reply headings. Codes such as `merits:I.A` and `reply:I.B` identify the brief and heading path.

The pipeline numbers all turns in the Oyez oral-argument JSON linked above (ID 25500), in transcript order starting at 1. Utterance IDs follow the format `25500__turn_NNNNNN__utterance`, with a six-digit turn number. These IDs can be reproduced using the same JSON and pipeline.

## Dataset Columns

Each row represents one Justice utterance.

In the column patterns below, `{source}` is `gpt`, `gemini`, `consolidated`, `student_a`, `student_b`, or `human_final`; `{model}` is `gpt` or `gemini`; `{n}` is `1`, `2`, or `3`.

| Column or pattern | Meaning |
| --- | --- |
| `docket_number` | Supreme Court docket number. |
| `oyez_argument_id` | Oyez oral-argument ID. |
| `utterance_id` | Pipeline-generated utterance identifier. |
| `target_justice` | Justice speaking. |
| `start_time`, `stop_time` | Utterance boundaries in seconds from the start of the audio. |
| `stage1_{source}_present` | Whether a question function is present (`true` / `false`). |
| `stage1_{source}_purposes` | Question-purpose labels, stored as a JSON array. |
| `stage2_{source}_targets` | Selected terminal heading codes, stored as a JSON array. |
| `stage3_{source}_stance` | Bundle-level stance: `Favor`, `Against`, `Neutral`, or `Mixed`. |
| `stage{n}_sent_to_llm_adjudicator` | Whether the utterance was routed to LLM adjudication at that stage. |
| `stage{n}_{model}_valid` | Pipeline validation status of the saved effective model result. |

`gpt` = GPT-5.2; `gemini` = Gemini 2.5 Pro; `consolidated` = consolidated LLM result; `student_a` / `student_b` = student annotators; `human_final` = legal expert's final decision. LLM adjudicator: Claude Opus 4.5.

Stage 2 lists only selected terminal nodes, without repeating their ancestors. Stage 3 judgments concern the Stage 2 human-final bundle.

`not applicable` marks an unexecuted stage or a purposes field whose corresponding `present` value is `false`. `[]` records an empty target selection.

For the saved model result, `true` means pipeline validation passed and `false` means it failed.

## Contact

- Heng Zheng: `hengzheng@uky.edu`
- Alex Zhang: `xz328@duke.edu`
