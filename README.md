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

## Use it from the A.R.Y.A voice assistant

A.R.Y.A (Adaptive Real-time Yielding Assistant) is a voice assistant
app for Android, written in Flutter. Source code:
https://github.com/dcon4/A.R.Y.A

Its **Local search** feature asks questions of this assistant on your
computer and speaks the answer back to you, naming the folder and
page where it was found. The phone and the computer must be on the
same wifi, and this assistant must be running.

How it works once enabled:

- Say or type **"local search"** followed by your question, for
  example "local search where is my passport". **"ask my documents"**
  works exactly the same way.
- Saying only the command makes A.R.Y.A ask what you want to search
  for and listen for your question.
- After a search, the next three questions continue as local searches
  without repeating the command, so you can ask follow-ups such as
  "what about his other books?" and A.R.Y.A understands words like
  "his" and "it". Say **"new conversation"** to end that earlier.
- A.R.Y.A reads the answer aloud, then names where it came from.

To turn it on: open A.R.Y.A's **Settings**, then **Local Search**.
Switch it on, set the computer address to this assistant's address
(the phone cannot use `localhost`, so use the computer's network
address, for example `http://192.168.1.5:8642` - this assistant
prints the exact address when it starts), and pick which model to
use. The model chooser lists the free models from each provider this
assistant supports.

A.R.Y.A talks to the JSON API described in the next section, so any
other app can do the same.

## JSON API for other apps

Two endpoints are available. They work on the address the server
prints (with or without the `/assistant` prefix), need no password on
your own network, and accept and return JSON.

**Check it is running**

```
GET /api/status
```

Example reply:

```json
{
  "ok": true,
  "chunks": 82028,
  "files": 48,
  "folders": ["/path/to/your/folders"],
  "index_ready": true,
  "model": "space-bunny-free",
  "vectors": 82028,
  "web_search": false
}
```

`vectors` smaller than `chunks` means the index is still building.

**Ask a question**

```
POST /api/ask
Content-Type: application/json
```

Request body:

```json
{
  "question": "Where was the passport mentioned?",
  "model": "space-bunny-free",
  "provider": "zen",
  "history": [
    {"question": "first question", "answer": "its answer"}
  ]
}
```

Only `question` is required. `model` picks a model (its provider is
worked out automatically), `provider` forces a specific provider,
and `history` carries up to three earlier question-and-answer pairs
so follow-ups make sense. Allow up to 120 seconds for the reply; a
normal answer takes 5 to 15 seconds.

Example reply:

```json
{
  "ok": true,
  "question": "Where was the passport mentioned?",
  "answer": "The passport is in the safe [1].",
  "model": "space-bunny-free",
  "provider": "zen",
  "history": [],
  "sources": [
    {
      "file": "/path/notes.txt",
      "folder": "Documents",
      "title": "notes",
      "location": "page 2",
      "text": "a 400-character excerpt...",
      "score": 0.81
    }
  ],
  "error": null
}
```

When something goes wrong, `ok` is `false` and `error` holds a
plain-language message meant to be spoken or shown as-is. An empty
question gets HTTP 400. There is no need to send any key from the
calling app; the keys live in this assistant's own `.env`.

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
