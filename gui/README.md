# RB-Y1 VLA Control GUI

This directory is standalone. It does not modify the existing robot or
LeRobot source files. The **Run robot client** button invokes
`../run_lerobot_robot_client.sh` exactly as it currently exists.

## Run

From the repository directory:

```bash
conda activate lerobot
./gui/run_gui.sh
```

When launched from an SSH or text-only terminal, `run_gui.sh` automatically
uses the logged-in Jetson desktop on display `:0` when that display is
available. Otherwise, run it from a desktop terminal or reconnect with X11
forwarding (`ssh -X`).

To validate dependencies and paths without opening a window or cameras:

```bash
./gui/run_gui.sh --check
```

## Camera behavior

The GUI opens the left-arm, head, and right-arm D405 cameras when it starts.
Before launching the robot client, it releases all camera pipelines so the
client can acquire them. When the client exits, the GUI reconnects its camera
previews automatically.

Camera serial numbers and the script path are stored in `config.json`.

## Current scope

The task and model fields are read-only because those values are hard-coded in
the existing `run_lerobot_robot_client.sh`. This prevents the GUI from showing
a selection that the launched script would ignore. They can be made selectable
later without changing existing source by adding a separate GUI-owned launch
configuration.
