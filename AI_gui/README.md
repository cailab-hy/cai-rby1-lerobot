# RB-Y1 AI Voice GUI (V3)

`AI_gui/` is an independent copy of Web v2. The original `gui/`, `web_gui/`,
robot, and LeRobot source files are not modified by V3.

V3 keeps all three live D405 views during policy execution and replaces the
DISH/CUP selection buttons with a Korean voice interface driven by the laptop
browser microphone.

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

Direct LAN access (`http://<JETSON-IP>:8001`) displays the GUI, but Chrome may
reject microphone access on a non-secure remote HTTP origin.

## Voice operation

1. Start the inference server as usual.
2. Click the animated orb once.
3. Allow microphone access in the laptop browser.
4. Speak a command:

```text
접시를 집어줘
컵을 집어줘
멈춰
종료해줘
```

Clicking the orb arms voice policy execution mode. The first two commands merge
short recognition fragments (for example `접시를` + `집어줘`), select the
matching task, and immediately start `run_lerobot_robot_client.sh` through the
GUI-owned V3 wrapper. The spoken Korean response no longer delays policy
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

Chrome/Edge speech recognition may use the browser vendor's online speech
service, so the laptop may need internet access. No microphone audio is handled
or stored by the Jetson web server.

## Commands and responses

Voice keywords, exact policy instructions, and spoken responses are configured
in `config.json`. V3 currently maps:

- `접시`, `그릇`, `dish` plus an action phrase -> DISH
- `컵`, `잔`, `cup` plus an action phrase -> CUP
- `멈춰`, `일시정지`, `스톱`, `stop` -> pause policy actions and hold the current pose
- any phrase containing `종료` -> fully stop the robot client

## Offline check

This does not open cameras, start the robot client, or bind a network port:

```bash
./AI_gui/run_AI_gui.sh --check
```

## Network safety

This development server has no user authentication. Use it only on a trusted
robot network, use SSH port forwarding where practical, and never expose port
8001 to the public internet.
