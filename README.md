# Folio Search — offline search for Śrīla Prabhupāda's words

Search the recorded teachings of His Divine Grace A.C. Bhaktivedanta Swami
Prabhupāda — books, lectures, conversations, letters — entirely offline, by
keyword or by meaning (semantic search). Every result is the original text
with its exact reference. **No AI-generated answers: the tool only finds and
displays; it never writes.**

> **Bring your own texts.** This repository contains *code only*. The texts
> are © The Bhaktivedanta Book Trust and are not distributed here. You need
> your own Folio infobase (the classic `.nfo` desktop library); the pipeline
> converts your own export into a searchable corpus on your machine.

## Quick start

1. **Install** Python 3.10+ (tick "Add Python to PATH"), then:
   `pip install fastembed numpy`
2. **Export your Folio infobase**: open it in Folio Views → File → Export →
   format **MS Word RTF**. (Use the Contents-tab checkboxes to export
   everything or just selected sections.)
3. **Convert** the export into the corpus:
   `python pipeline.py "YourExport.RTF" corpus/library`
   For very large exports (300 MB+), see the note below.
4. **Build the semantic index** (one-time, ~1–2 hours, resumable):
   double-click `Build Semantic Index.bat` — or `python build_index.py 999999`
5. **Search**: double-click `Start Folio Search.bat` — the browser opens
   automatically at `http://localhost:7777`.

Keyword search works immediately after step 3; step 4 adds meaning-based
search. The embedding model (bge-small-en-v1.5, ~65 MB) downloads once on the
first index build and is cached in `models/`; everything is offline after that.

## Features

- **Hybrid search** — exact keyword and semantic (by meaning) side by side
- **Section filters** — BG / SB / CC / Other Books / Lectures / Conversations / Letters
- **Diacritic-insensitive** — `krsna` matches Kṛṣṇa; full IAST preserved in display
- **Exact phrases** with `"double quotes"`
- **Verbatim only** — every hit shows the source reference; "not found" is an
  honest answer, never papered over by a language model
- **Full-record view** — open any complete lecture/letter/chapter with
  highlights (`open full ↗`)
- CLI mode: `python serve.py --in Conversations,Letters "your query"`

## How it works

`pipeline.py` streams the Folio RTF export, converts the Balaram diacritic
font encoding losslessly to Unicode IAST, uses Folio's hidden record markers
(plus tape codes like `720312MW-VRN`) as passage boundaries, and writes one
JSON line per passage: `{"ref": "Bg 9.22", "text": "..."}`.

`build_index.py` splits passages into ~1800-character windows and embeds them
with `bge-small-en-v1.5` (384-dim, ONNX, CPU). Resumable with checkpoints
every ~3 minutes.

`serve.py` is a dependency-light local web app (stdlib http.server): keyword
search over normalized text + cosine similarity over the embedding matrix,
merged and returned as verbatim passages.

## Converting very large exports

The parser is streaming, but for multi-hundred-MB RTFs you may want to split
at `\pard` boundaries and convert in segments — see the comments in
`pipeline.py`. A 362 MB export (≈34,000 passages) converts in under a minute
on a modern machine.

## Copyright

All code: MIT (see LICENSE). The texts you index remain the property of
their copyright holder (The Bhaktivedanta Book Trust) — this tool is a
personal study aid for content you already own. Please support the BBT by
purchasing Śrīla Prabhupāda's books.

Hare Kṛṣṇa.
