# RB-Y1 AI Voice GUI (V3)

`AI_gui/` is an independent copy of Web v2. The original `gui/`, `web_gui/`,
robot, and LeRobot source files are not modified by V3.

V3 keeps all three live D405 views during policy execution and replaces the
BOWL/CUP selection buttons with a voice interface driven by the laptop browser
microphone. Korean and English are both accepted: the language of each spoken
command is detected automatically, and the console answers in that language.

## Where each piece runs

As before, speech is handled on the operator's laptop and only the resulting
task command travels to the robot:

| Piece | Machine | Port |
| --- | --- | --- |
| `stt_service.py` — Whisper recognition | operator laptop (x86 + NVIDIA GPU) | 8002 |
| `server.py` — cameras, robot client, GUI | Jetson | 8001 |
| browser — microphone, VAD, speech output | operator laptop | — |

The browser posts each utterance straight to the laptop's own recognition
service, so microphone audio never crosses the network.

Recognition deliberately does **not** run on the Jetson: the aarch64
`ctranslate2` wheels are built without CUDA, so Whisper would fall back to the
ARM CPU and take seconds per command — far too slow for `멈춰` / `stop`.

## Speech recognition service (laptop)

Install once into the laptop's `lerobot` environment:

```bash
conda activate lerobot
pip install faster-whisper nvidia-cublas-cu12
```

`nvidia-cublas-cu12` is only needed because this environment's torch ships
CUDA 13 while ctranslate2 links against the CUDA 12 cuBLAS. `whisper_stt.py`
loads those libraries itself, so no `LD_LIBRARY_PATH` wrapper is required. If
the GPU is unavailable the engine falls back to CPU automatically and logs a
warning; recognition still works, but takes about a second instead of ~45 ms.

Then start it, leaving it running for the session:

```bash
conda activate lerobot
./AI_gui/run_stt_service.sh
```

`--check` loads the model and exits. `--host 0.0.0.0` serves other devices on
the LAN, needed only if the GUI is opened from a different machine than the one
running this service. The model is configured under `stt` in `config.json`
(`small` on `cuda`, downloaded on first use); `stt.endpoint` is the address the
browser will post to. The **Speech Recognition** field in the GUI shows the
loaded model and device, or an error if the service is not reachable.

## Start on the Jetson

Stop any running `web_gui` server first because only one process can own each
RealSense camera. Then run:

```bash
cd ~/rby1-lerobot/rby1-lerobot
conda activate lerobot
./AI_gui/run_AI_gui.sh
```

AI GUI V3 uses port `8001`, leaving Web v2's port `8000` unchanged.

## Open from the laptop (recommended)

Browser microphone features work most reliably from a secure context. The
easiest development setup is to forward Jetson port 8001 and open it as
localhost on the laptop.

### VS Code Remote-SSH

1. Open the **Ports** panel.
2. Forward port `8001`.
3. Open the forwarded address, normally `http://localhost:8001`.

### Normal SSH

On the laptop:

```bash
ssh -L 8001:localhost:8001 nvidia@<JETSON-IP>
```

Then open:

```text
http://localhost:8001
```

Direct LAN access (`http://<JETSON-IP>:8001`) displays the GUI, but Chrome
blocks microphone access on a non-secure remote HTTP origin and the orb will
report that it needs a secure context. Either use the port forwarding above, or
allow the origin explicitly in `chrome://flags/#unsafely-treat-insecure-origin-as-secure`.

## Voice operation

1. Start the inference server as usual.
2. Click the animated orb once.
3. Allow microphone access in the laptop browser.
4. Speak a command in either language:

```text
그릇을 집어줘          Pick up the bowl
컵을 집어줘            Pick up the cup
멈춰                   Stop
종료해줘               Terminate
```

Nothing needs to be switched between the two. Each utterance is recognized on
its own, and the reply, the orb text, and the whole interface follow whichever
language was just spoken. The `AUTO · KO` chip in the header shows the language
currently in use; click it to pin `KO` or `EN` if detection misfires during a
demo, and click again to return to `AUTO`.

Clicking the orb arms voice policy execution mode. The browser detects the start
and end of each utterance, posts it to the local recognition service as 16 kHz
mono audio, and Whisper returns both the text and its language. The first two
commands select the
matching task and immediately start `run_lerobot_robot_client.sh` through the
GUI-owned V3 wrapper. The spoken response no longer delays policy
startup. `멈춰` pauses policy actions while keeping the robot client, cameras,
and inference connection alive. A new DISH/CUP command updates the task and
resumes the existing client. Phrases containing `종료` send SIGINT and fully
stop the robot client as before.

On a cold start the GUI tracks `robot_preparing`, `policy_loading`, and
`executing` from the real client log. It immediately says `네, 정책을 실행합니다.`
and, when policy loading is still active after ten seconds, announces
`현재 작업을 수행하기 위한 정책을 업로드하고 있습니다.` Commands other than
pause/terminate received during execution get the response
`현재 작업을 실행하고 있습니다.`

The orb changes appearance for idle, listening, processing, speaking, policy
execution, and error states. Clicking the orb again disables continuous voice
listening.

Recognition is fully local, so the laptop no longer needs internet access for
speech. Utterance audio goes to the laptop's own recognition service, is
transcribed in memory, and discarded; it never reaches the Jetson and nothing is
written to disk.

Because the microphone stays open, the console stays quiet unless something in
the utterance was aimed at it. Only phrases that mention a known object or
action word get a spoken reply.

## Commands and responses

Voice keywords, policy instructions, spoken responses, and the microphone
sensitivity are all configured in `config.json`. Keyword lists are keyed by
language and every language is matched regardless of what was detected, so a
Korean speaker saying "cup" still works; only the reply language follows the
detected one. V3 currently maps:

- `보울`, `그릇`, `사발` / `bowl` plus an action word -> BOWL
- `컵`, `잔`, `커피잔` / `cup`, `mug`, `glass` plus an action word -> CUP
- `멈춰`, `일시정지`, `그만` / `stop`, `pause`, `hold on` -> pause policy actions and hold the current pose
- `종료`, `끝내` / `terminate`, `shut down`, `quit` -> fully stop the robot client

`voice.vad` tunes the microphone: raise `min_rms` if the lab's background noise
keeps opening the microphone, and raise `end_ms` if long pauses mid-sentence cut
commands in half.

The policy instruction sent to the inference server (`RBY1_WEB_TASK`) stays
English in both languages, since the policies were trained on English prompts.

## Offline check

This does not open cameras, start the robot client, or bind a network port:

```bash
./AI_gui/run_AI_gui.sh --check
```

## Network safety

This development server has no user authentication. Use it only on a trusted
robot network, use SSH port forwarding where practical, and never expose port
8001 to the public internet.
