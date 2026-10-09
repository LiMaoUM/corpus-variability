# Agent room

Durable record of exchanges between Claude Code and Codex on this project: requests, replies,
evidence, disagreements and dispositions.

## Layout

- `threads/YYYY-MM-DD-topic/` holds one topic.
- Each message is one Markdown file named `YYYYMMDDTHHMMSSffffffZ-author-kind.md` (UTC; add a
  unique suffix on collision). Files are created exclusively and never edited; corrections are new
  messages.

## Message fields

Author, recipient, UTC timestamp, kind (request, reply, disposition), status (pending, answered,
closed), the filename it replies to, the task, the authorized edit scope, relevant paths and
revision, the substantive content, and the next action. Status is resolved by later replies.

## Conventions

- Save a request before sending it; give the recipient the absolute request path and the room
  path. Save the full reply in the same thread.
- The requesting agent records which recommendations were accepted, deferred or rejected and
  why, and distinguishes an agent recommendation from a decision Mao approved (see
  `DECISIONS.md`).
- English only. Source quotations verbatim. No private reasoning traces, credentials or
  unrelated transcripts.

## Resuming a thread

Read this README, then the latest message in the thread directory, and reply with a new file.
