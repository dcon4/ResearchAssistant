# Research Assistant

Ask questions about your own documents and get answers that name the
file and page they came from. Optionally search the internet for
questions your documents cannot answer.

## What it does

- Reads documents (PDF, EPUB, TXT, HTML, Markdown) from three folders
  you choose (set them in `settings.json` under `index_folders`;
  subfolders are included).
- ZIP archives are read as well. Every document inside is indexed, and
  an answer names both the archive and the file within it, for example
  `bundle.zip - notes/summary.txt`. A zip inside a zip is left alone,
  and one file cannot be larger than 60 MB.
- Builds a searchable index on this computer's hard drive. The
  documents themselves are never moved or copied.
- Answers your questions with an AI model from the provider of your
  choice (OpenRouter, Groq, NVIDIA NIM, OpenCode Zen, and others),
  listing which file (and page) each answer came from.

## How to start it

```
cd ~/research-assistant
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt -r requirements-dev.txt
.venv/bin/python -m research_assistant.cli serve
```

Then open the address it prints (for example
`http://localhost:8642`) in a browser, on this computer or any
device on your network.

## The pages

- **Ask** - type a question, get an answer with sources.
- **Index status** - which folders are being indexed, how far along.
- **Settings** - choose local (free) or OpenRouter (a few dollars)
  embedding, turn web search on or off, turn verbose logging on or
  off, turn the daily rescan on or off.
- **Download debug log** - saves the current log file, for sending to
  whoever helps you.

## Keeping the index up to date

- Every day at 5:17 AM the assistant rescans the three folders on its
  own and picks up new or changed files. It never deletes anything.
- You can also press **Scan now** on the Index status page, or run
  `.venv/bin/python -m research_assistant.cli index` from a terminal.
- Only one scan can run at a time; a second one is politely refused.

## Your API key

Copy `.env.example` to `.env` and paste the key for the provider you
want to use. The key is needed for answering questions (and for
embedding only if you switch Settings to OpenRouter mode). Pick the
provider and model in Settings. The `.env` file is never committed
anywhere.

## Costs

- Local embedding (default): free. Uses this computer's processor.
- OpenRouter embedding: a few dollars one-time for the whole corpus.
- Answers: free models cost nothing; paid models cost per use.
- Web search: billed per search when enabled.

## If something goes wrong

Use **Download debug log** on any page and send the file to your
helper. Turn on verbose logging in Settings first to get more detail.
