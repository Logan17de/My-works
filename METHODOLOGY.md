# How this catalog was audited

This repository is an evidence-based map of the local project archive, not a claim that every experiment reached production or proved its research hypothesis.

## Scope

- Audit date: **2026-07-31**
- Source: 39 sibling source folders under the local archive root, excluding this `My-works` catalog repository
- Inspected evidence: READMEs, design notes, representative source files, configurations, dependency manifests, tests, checkpoints, generated outputs, logs, and local modification dates
- Excluded from publication: source repositories, credentials, private datasets, model weights, virtual environments, dependency trees, logs, and generated trading artifacts

Folders that only hold datasets, tokenizer files, documents, or an empty placeholder remain in the catalog because they explain how the surrounding projects evolved. They are labeled as support material rather than presented as standalone accomplishments.

Paths listed as key evidence are provenance pointers to the audited local archive. They are not public links or a substitute for publishing sanitized, reproducible artifacts.

## Artifact classifications

| Classification | Meaning |
|---|---|
| **Measured** | A quantitative comparison or result summary assesses at least part of the experiment. The main hypothesis may still remain untested. |
| **Executed** | Checkpoints, logs, or generated outputs show that a substantial path ran, but they do not establish that the hypothesis outperformed a baseline. |
| **Implemented** | The central architecture or product flow exists in code, but durable run evidence or evaluation is missing. |
| **Fragment** | The folder is a partial sketch, isolated component, duplicate snapshot, broken scaffold, or empty placeholder. |
| **Support** | The folder stores datasets, tokenizer assets, notes, or design documents rather than a standalone project. |

“What the artifacts establish” is deliberately narrower than “what the idea intended to prove.” A checkpoint establishes that training reached a save point; it does not by itself establish model quality. A working UI establishes an interaction flow; it does not establish production readiness.

## Status language

- **Current** means recent local activity and a coherent continuation path were visible at audit time.
- **Paused** means meaningful implementation exists but recent continuation evidence, evaluation, or required infrastructure is missing.
- **Archived** means the folder is primarily a historical branch, superseded snapshot, or completed learning exercise.
- **Placeholder** means there is too little material to assess.

The “why paused” statements are reasoned inferences from the artifacts—not quotes from the author. They use qualified language whenever the folder does not contain an explicit decision record.

## Security boundary

The audit detected unsafe credential storage in several local prototypes. No sensitive values, credential files, or affected source trees are copied here. Root ignore rules are defensive only; future source publication requires verified credential rotation and a history-aware secret scan.
