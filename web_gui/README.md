# RB-Y1 Browser GUI (Web v2)

This directory is independent from the existing RB-Y1 and LeRobot source. It
opens the three D405 cameras and invokes `../run_lerobot_robot_client.sh`
without modifying it.

## Start

On the Jetson, including from a VS Code Remote-SSH terminal:

```bash
cd ~/rby1-lerobot/rby1-lerobot
conda activate lerobot
./web_gui/run_web_gui.sh
```

Then open the following address in a browser on a computer connected to the
same network:

```text
http://<JETSON-IP>:8000
```

Find the Jetson IP with:

```bash
hostname -I
```

Stop the web server with `Ctrl+C` in its terminal. The shutdown handler sends
SIGINT to a running robot client and releases the cameras.

## Task buttons

Select `1. DISH` or `2. CUP` before pressing the robot-client start button.
Web v2 replaces the shell script's existing `--task` argument inside the
GUI-owned client wrapper, so the original shell script stays unchanged.

Task labels and the exact English policy instructions are configured in
`config.json`. Task selection is locked while the client is starting, running,
or stopping.

## Offline check

This does not open cameras, start the robot client, or bind a network port:

```bash
./web_gui/run_web_gui.sh --check
```

## Camera behavior in Web v2

The browser shows all three cameras before and during policy execution. The
existing policy script currently owns `front` and `left`. A GUI-owned command
shim launches the original client and copies those clients' latest buffered
frames into a local memory-mapped relay without consuming the policy's camera
events. The `right` camera is display-only and remains owned by the web server.

```text
front + left: LeRobot client -> policy + Web v2 relay
right:        Web server -> browser only
```

When the client exits, `front` and `left` return to direct web preview mode.
The original `run_lerobot_robot_client.sh` and LeRobot/RB-Y1 source files are
not modified.

## Network safety

The server listens on the local network and has no user authentication. Run it
only on a trusted robot network and do not expose port 8000 to the public
internet.
