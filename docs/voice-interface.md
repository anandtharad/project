# Voice interface

People can talk to the companion and hear it reply, or keep typing and
reading; both modalities work in both directions on the same conversation.
Speech goes over a WebSocket beside the text API. The spoken reply is the same
validated reply the text API returns: a spoken turn goes through the normal
turn lifecycle (safety, decisions, memory, knowledge, output validation), and
only the finished reply is turned into speech.

(docs/voice.md is about how replies *read*, the companion prompt's tone; this
page is about speech in and out.)

## Status

- **Half duplex.** One side speaks at a time. While the companion is
  transcribing, thinking or speaking, the person's audio is dropped and typed
  messages are refused with `busy`. Full duplex with barge-in is prepared for
  (see [Barge-in later](#barge-in-later)) but not enabled: `VOICE_DUPLEX`
  accepts only `half`.
- **Providers:** Deepgram, Sarvam and Gnani, each for speech-to-text (STT)
  and text-to-speech (TTS), all over their streaming WebSocket APIs. The
  adapters follow each provider's published reference as read on
  2026-10-16. **None has been run against the real service yet** (no keys).
  Everything was tested with local servers that speak each provider's
  documented protocol, and end to end with the development providers
  (`fake`) through the real database, orchestrator and a browser. Expect
  small corrections on the first live run; the open questions per provider
  are listed below.
- Off by default (`VOICE_ENABLED=false`).

## Setup

In `.env`:

```
VOICE_ENABLED=true
STT_PROVIDER=deepgram        # deepgram | sarvam | gnani | fake | none
TTS_PROVIDER=sarvam          # deepgram | sarvam | gnani | fake | none
DEEPGRAM_API_KEY=...
SARVAM_API_KEY=...
VOICE_LANGUAGE=en-IN         # BCP-47: en-IN, hi-IN, en-US, ...
```

Only the keys of the selected providers are needed; startup fails with a
clear message when one is missing. STT and TTS can come from different
providers. `STT_MODEL`, `TTS_MODEL` and `TTS_VOICE` override the defaults
below. `fake` providers are development stubs (a scripted transcript, a tone
per word); `APP_ENV=production` refuses them.

Then restart the app and open `http://localhost:8000/voice`: paste a
development token, choose text and/or audio output, connect, and hold the
button (or the space bar) to talk. The page captures the microphone, sends
16 kHz PCM and plays the reply. Browsers allow the microphone only on
`localhost` or HTTPS.

Keys stay on the server. The browser never talks to a speech provider: it
streams audio to this app, which holds the provider connections. (Gnani's
STT takes its settings only as upgrade headers, which browsers cannot set, so
a server-side connection is required for it anyway.)

## Settings

| Variable | Default | Meaning |
|---|---|---|
| `VOICE_ENABLED` | `false` | turns the voice endpoints on |
| `STT_PROVIDER`, `TTS_PROVIDER` | `none` | `deepgram`, `sarvam`, `gnani`, `fake` or `none`; with STT `none` the socket takes text only, with TTS `none` replies are text only |
| `STT_MODEL`, `TTS_MODEL`, `TTS_VOICE` | provider default | see the provider sections |
| `VOICE_LANGUAGE` | `en-IN` | the conversation's language, passed to both providers |
| `VOICE_INPUT_SAMPLE_RATE` | `16000` | the rate of the PCM the client sends (8000-48000; Sarvam takes 8000 or 16000, Gnani 8000, 16000, 44100 or 48000) |
| `VOICE_DUPLEX` | `half` | one side at a time (`full` is not available yet) |
| `VOICE_END_OF_TURN` | `client` | `client`: push-to-talk, the turn ends when the person lets go. `provider`: hands-free, the STT provider's end-of-utterance ends it |
| `VOICE_MAX_UTTERANCE_SECONDS` | `60` | an utterance is ended after this much audio |
| `VOICE_TICKET_TTL_SECONDS` | `60` | how long a connection ticket is valid |
| `DEEPGRAM_API_KEY`, `SARVAM_API_KEY`, `GNANI_API_KEY` | | the selected providers' keys |

## How a session works

```
POST /v1/conversations/{id}/voice/tickets  (bearer token)  -> {"ticket", "expires_in", "path"}
WS   /v1/voice?ticket=<ticket>
```

A browser cannot put a bearer token on a WebSocket, so the client first asks
for a ticket with its normal credentials. A ticket is single use, expires
after `VOICE_TICKET_TTL_SECONDS`, is bound to the user and the conversation,
and is checked for ownership when issued. A refused connection is accepted
and then closed with code `4401` (bad or expired ticket) or `4503` (voice
disabled), because a refused handshake reaches the page only as an
unexplained failure.

**Client to server**

| Frame | Meaning |
|---|---|
| binary | the person's audio: 16-bit little-endian mono PCM at `VOICE_INPUT_SAMPLE_RATE`, under 64 KB and an even length (100 ms frames are fine) |
| `{"type": "start", "output": ["text", "audio"]}` | optional, first: which outputs to send (text is always sent) |
| `{"type": "end_of_speech"}` | push-to-talk released |
| `{"type": "text", "content": "..."}` | a typed message |
| `{"type": "playback_done"}` | the reply's audio has finished playing |
| `{"type": "stop"}` | end the session |

**Server to client**

| Event | Meaning |
|---|---|
| `ready` | `duplex`, `input` and `output` modalities, `input_audio` and `output_audio` formats, `language` |
| `state` | `listening`, `transcribing`, `thinking`, `speaking` |
| `transcript` | `{"final": bool, "text"}`; interim results only from providers that send them |
| `reply` | the reply text, message ids, `safety` notice and cited `sources`, exactly as the text API |
| binary | the reply's audio, PCM in the `output_audio` format |
| `audio_end` | all audio for this reply was sent (`seconds`) |
| `no_speech` | the utterance had no recognisable speech |
| `interrupted` | full duplex only (not yet) |
| `error` | `code` and `message`: `busy`, `stt_unavailable`, `tts_unavailable`, `bad_audio_frame`, `bad_message`, or a turn error such as `too_many_messages` |

**States.** `listening` → (audio, then `end_of_speech`) `transcribing` →
`thinking` → `speaking` → `listening`. A typed message goes from `listening`
straight to `thinking`. In half duplex the server returns to `listening` only
after the client reports `playback_done` (or the audio's length plus five
seconds, if it never does), so the microphone does not pick up the reply.

**Failure.** If speech recognition fails, the person gets one
`stt_unavailable` error, the rest of that utterance is dropped, and typing
still works. If speech synthesis fails, the reply has already been sent as
text, and the person gets `tts_unavailable`. A turn error (for example the
per-user message limit) arrives as `error` and the session returns to
`listening`.

**What is stored.** This app does not store audio; what a speech provider
keeps is governed by its own terms. A spoken message is stored as its transcript, with
`generation_metadata.input = {"modality": "voice", "channel": "voice_socket",
"stt": {"provider", "model"}}`; typed messages store `{"modality": "text"}`.
Transcripts are personal data, exactly like typed messages: they are covered
by conversation deletion, retention and memory rules.

**What is spoken.** `speakable()` removes citation markers, Markdown and
leftover separators, so `[1]` or `**` are not read aloud; the sources stay in
the `reply` event. The reply goes to the TTS provider sentence by sentence.

**Connections.** A speech-to-text connection is opened for each utterance
(on its first audio frame) and closed after it, and a text-to-speech
connection for each reply. This costs a connection setup per turn but avoids the providers' idle timeouts while the
other side is talking (Deepgram closes an STT socket after about 10 s without
audio; Sarvam's TTS socket closes after about 60 s idle). Keeping connections open with
keep-alives is a later optimisation.

## Providers

The facts below come from each provider's API reference, guides and official
SDK as read on 2026-10-16. Where the provider's documents disagree, the
adapter takes the safer reading and the disagreement is listed.

### Deepgram

- **STT:** `wss://api.deepgram.com/v1/listen`, `Authorization: Token <key>`,
  settings in the query. Default model `nova-3`; `en-IN` is passed as is,
  `hi-IN` as `hi`. Audio goes up as binary frames. Interim and final results,
  `SpeechStarted` and `UtteranceEnd` (1000 ms gap) events. Push-to-talk ends
  with `Finalize` then `CloseStream`.
- **TTS:** `wss://api.deepgram.com/v1/speak`, default voice
  `aura-2-thalia-en` (the voice is part of the model name; `TTS_VOICE` or
  `TTS_MODEL` sets it), raw linear16 at 24 kHz as binary frames. `Speak`
  messages (at most 2000 characters each), `Flush` (acknowledged by
  `Flushed`), and `Clear` cancels in-progress synthesis (`Cleared`), which is
  the barge-in hook.
- **Limits:** no Hindi TTS voice, and no Indian-accent English voice on
  `/v1/speak` (Indian-accent English exists only on Deepgram's newer Flux TTS,
  `/v2/speak`, a different protocol not implemented here). For Hindi or
  Indian English speech output, use Sarvam or Gnani for TTS. Deepgram documents
  limits of 20 flushes and 2400 characters of TTS per minute; India data
  residency uses `api.in.deepgram.com`.

### Sarvam

- **STT:** the realtime WebSocket
  `wss://api.sarvam.ai/speech-to-text-realtime/ws`,
  `Api-Subscription-Key` header, settings in the query (`language_code`,
  `model`, `encoding=linear16`, `sample_rate` 8000 or 16000,
  `stream_type=balanced`). Default model `saaras:v3-realtime`. All messages
  are JSON; audio goes up base64 in `audio_input`. Push-to-talk uses
  `endpointing=manual` with `speech_start` / `speech_end`; hands-free uses the
  server's VAD (`vad.speech_end`). Partial and final transcripts.
- **TTS:** `wss://api.sarvam.ai/text-to-speech/ws?model=bulbul:v4-flash`,
  a `config` message first (language, speaker, 24 kHz, `linear16`), then
  `text`, `flush`, base64 `audio` chunks and a `final` event. Default speaker
  `simran_en_customer` (an example persona id from Sarvam's v4-flash guide;
  v4-flash takes persona ids, not v3 names). No cancel message: barge-in
  closes the socket, as Sarvam documents.
- **Open questions:**
  - One guide calls the realtime STT endpoint beta, and its close code 4000
    can mean the account is not enabled for it. If a key gets 4000, ask
    Sarvam to enable it.
  - The TTS reference says streaming returns MP3 only; the guides list
    linear16 and recommend it. The adapter asks for linear16 and refuses
    compressed audio with `tts_unavailable` instead of playing noise; if your
    account returns MP3, Sarvam TTS will not work until MP3 decoding is
    added.
  - The default TTS model differs between documents, so the model is always
    sent explicitly.

### Gnani

- **STT:** `wss://api.vachana.ai/stt/v3/stream`. All settings are upgrade
  headers (`x-api-key-id`, `lang_code`, `x-sample-rate`, `x-format`, VAD
  headers), fixed for the connection. Audio must be binary frames of exactly
  1024 bytes at real-time pace; the adapter re-frames whatever the client
  sends. Final transcripts only (one per VAD segment, announced by
  `processing`); no interim results. There is no flush or end message, so to
  end a push-to-talk turn the adapter sends 0.9 s of paced silence (the
  server's minimum silence is set to 600 ms), which adds about a second to
  each spoken turn. Sessions are capped at 15 minutes (each utterance gets
  its own connection here).
- **TTS:** `wss://api.vachana.ai/api/v1/tts`, `X-API-Key-ID` header, model
  `timbre-v2.5`, voices `Kaveri` (en-IN) and `Nalini` (hi-IN) by default,
  raw 16-bit PCM at 24 kHz. Each request is one whole text (no incremental
  text), so the reply is sent sentence by sentence. No cancel message:
  barge-in closes the socket.
- **Open questions:** whether one TTS connection takes several requests (the
  API page says yes, the SDK opens a connection per request; the adapter
  reuses it and reconnects once if the server closed it); access may be
  gated per account (a search snippet mentioned a waitlist).

### Choosing

| Need | STT | TTS |
|---|---|---|
| English (India), lowest risk | Deepgram | Sarvam or Gnani |
| Hindi | Deepgram (`hi`), Sarvam or Gnani | Sarvam or Gnani (not Deepgram) |
| Interim transcripts on screen | Deepgram or Sarvam | |
| Barge-in later with a real cancel | | Deepgram (`Clear`) |

## Barge-in later

The pieces for full duplex are in place:

- `TextToSpeechStream.cancel()` stops synthesis (Deepgram `Clear`; Sarvam
  and Gnani close the socket);
- `VoiceSession._barge_in()` cancels the reply, bumps a turn counter so the
  interrupted turn does not change the state of the next one, sends
  `interrupted` and returns to `listening`;
- `on_audio` calls it when audio arrives while `speaking` in full duplex;
- the client already stops playback on `interrupted`.

What remains before `VOICE_DUPLEX=full` can be allowed:

1. **Echo cancellation.** Without it the microphone hears the reply and
   interrupts it. The browser's `echoCancellation` constraint helps but
   needs testing on real devices; otherwise use headphones or a
   provider-side solution.
2. **A barge-in signal that is not noise.** Interrupt on a provider's speech
   start (`SpeechStarted`, `vad.speech_start`) or on an interim transcript
   of two or more words, as Deepgram and Sarvam recommend, rather than on
   any audio frame.
3. **Keeping STT open while speaking**, with keep-alives (Deepgram
   `KeepAlive`, Sarvam `ping`, silence for Gnani), instead of one connection
   per utterance.
4. **Recording what was actually heard.** An interrupted reply is stored in
   full today; the stored turn should note the interruption (and ideally how
   much was played).
5. Tests for each, then allow `full` in `VOICE_DUPLEX`.

## Code

- `app/voice/base.py`: audio format, transcript events and the provider
  protocols (`SpeechToText`, `TextToSpeech` and their streams).
- `app/voice/session.py`: `VoiceSession`, the half-duplex state machine,
  `speakable()` and sentence splitting.
- `app/voice/providers/`: `deepgram.py`, `sarvam.py`, `gnani.py`, the shared
  WebSocket helpers (`_ws.py`) and the development stubs (`fake.py`).
- `app/voice/factory.py`: providers from settings.
- `app/voice/tickets.py`: single-use connection tickets.
- `app/api/voice.py`: the ticket endpoint, the WebSocket and the `/voice`
  page (`app/static/voice.html`).
- Tests: `tests/unit/test_voice_session.py` (state machine, half duplex,
  failures, the barge-in seam), `tests/unit/test_voice_providers.py` (each
  adapter against a local server speaking the provider's documented
  protocol), `tests/integration/test_voice.py` (tickets, ownership, a typed
  and a spoken turn through the real app and database).

To add a provider, implement the two protocols in `app/voice/base.py`, add
it to `app/voice/factory.py` and the `STT_PROVIDER` / `TTS_PROVIDER` choices,
and test it against a local server like the others.
