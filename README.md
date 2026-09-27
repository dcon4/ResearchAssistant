# Research Assistant

Ask questions about your own documents and get answers that name the
file and page they came from. Optionally search the internet for
questions your documents cannot answer.

## What it does

- Reads documents (PDF, EPUB, TXT, HTML) from three folders on your
  NAS share: `opencode/Ebooks`, `opencode/Documents`, `opencode/reports`
  (including new subfolders).
- Builds a searchable index on this computer's hard drive. The
  documents themselves are never moved or copied.
- Answers your questions with an AI model from OpenRouter, listing
  which file (and page) each answer came from.

## How to start it

```
cd ~/research-assistant
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt -r requirements-dev.txt
.venv/bin/python -m research_assistant.cli serve
```

Then open the address it prints (for example
`http://192.168.0.x:8642`) in a browser, on this computer or any
device on your network.

## Using it away from home

The same pages are also reachable from outside your network, through
the ngrok tunnel that already runs on this computer:

```
https://detail-online-exemplary.ngrok-free.dev/assistant/
```

Three things happen when you open that link:

1. **ngrok's warning page** appears first, because the free ngrok plan
   always shows one. Press its "Visit Site" button. You only see this
   once every 7 days.
2. **A username and password box** appears next, asking for the same
   login the opencode site uses. It is stored in
   `~/.config/ngrok/ngrok.yml`.
3. Then the assistant loads, at `/assistant/` on that address.

Nothing you can see is reachable without that login: without it the
address returns only ngrok's warning page or a plain "unauthorized"
message.

It starts on its own at boot, so it comes back after a restart without
anyone doing anything.

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

Copy `.env.example` to `.env` and paste your OpenRouter key into it.
The key is needed for answering questions (and for embedding only if
you switch Settings to OpenRouter mode). The `.env` file is never
committed anywhere.

## Costs

- Local embedding (default): free. Uses this computer's processor.
- OpenRouter embedding: a few dollars one-time for the whole corpus.
- Answers: free models cost nothing; paid models cost per use.
- Web search: billed per search when enabled.

## If something goes wrong

Use **Download debug log** on any page and send the file to your
helper. Turn on verbose logging in Settings first to get more detail.
