const cameraSummary = document.querySelector("#cameraSummary");
const clientSummary = document.querySelector("#clientSummary");
const taskValue = document.querySelector("#taskValue");
const taskButtons = document.querySelector("#taskButtons");
const modelValue = document.querySelector("#modelValue");
const startButton = document.querySelector("#startButton");
const stopButton = document.querySelector("#stopButton");
const reconnectButton = document.querySelector("#reconnectButton");
const logOutput = document.querySelector("#logOutput");
const apiMessage = document.querySelector("#apiMessage");

let lastLogId = 0;
let requestBusy = false;
let latestStatus = null;

async function request(path, options = {}) {
  const response = await fetch(path, { cache: "no-store", ...options });
  let payload = {};
  try { payload = await response.json(); } catch (_) { /* ignore malformed error bodies */ }
  if (!response.ok) throw new Error(payload.message || `HTTP ${response.status}`);
  return payload;
}

function updateCamera(camera) {
  const card = document.querySelector(`[data-camera="${camera.key}"]`);
  if (!card) return;
  const overlay = card.querySelector(".camera-overlay");
  const meta = card.querySelector(".camera-meta");
  card.classList.toggle("connected", camera.state === "connected");
  card.classList.toggle("error", camera.state === "error");
  overlay.textContent = camera.message;
  meta.textContent = `${camera.serial} · ${camera.message}`;
}

function updateTasks(status, clientActive) {
  const existingIds = Array.from(taskButtons.children).map((button) => button.dataset.taskId);
  const incomingIds = status.tasks.map((task) => task.id);
  if (existingIds.join(",") !== incomingIds.join(",")) {
    taskButtons.replaceChildren();
    for (const task of status.tasks) {
      const button = document.createElement("button");
      button.type = "button";
      button.className = "task-button";
      button.dataset.taskId = task.id;
      button.textContent = `${task.number}. ${task.label}`;
      button.addEventListener("click", () => post(`/api/task/select?id=${encodeURIComponent(task.id)}`));
      taskButtons.appendChild(button);
    }
  }
  for (const button of taskButtons.children) {
    button.classList.toggle("selected", button.dataset.taskId === status.selected_task_id);
    button.disabled = clientActive || requestBusy;
  }
}

async function refreshStatus() {
  try {
    const status = await request("/api/status");
    latestStatus = status;
    const activeStates = ["starting", "running", "stopping"];
    const clientActive = activeStates.includes(status.client.state);
    const connected = status.cameras.filter((camera) => camera.state === "connected").length;
    status.cameras.forEach(updateCamera);
    updateTasks(status, clientActive);

    cameraSummary.textContent = `Cameras ${connected}/3`;
    cameraSummary.classList.toggle("good", connected === 3);
    clientSummary.textContent = `Client ${status.client.state}`;
    clientSummary.classList.toggle("busy", clientActive);
    taskValue.textContent = status.task;
    modelValue.textContent = status.model;

    startButton.disabled = clientActive || requestBusy;
    stopButton.disabled = !clientActive || status.client.state === "stopping" || requestBusy;
    reconnectButton.disabled = clientActive || requestBusy;
    apiMessage.textContent = "";
  } catch (error) {
    apiMessage.textContent = `서버 연결 오류: ${error.message}`;
  }
}

async function refreshLogs() {
  try {
    const payload = await request(`/api/logs?after=${lastLogId}`);
    for (const item of payload.items) {
      const line = document.createElement("span");
      line.textContent = `[${item.time}] ${item.message}\n`;
      if (item.level === "error") line.className = "log-error";
      logOutput.appendChild(line);
      lastLogId = Math.max(lastLogId, item.id);
    }
    if (payload.items.length) logOutput.scrollTop = logOutput.scrollHeight;
  } catch (_) {
    // Status polling displays connection failures; keep the log quiet.
  }
}

async function post(path) {
  requestBusy = true;
  await refreshStatus();
  try {
    const payload = await request(path, { method: "POST", headers: { "X-RBY1-Control": "web-v2" } });
    apiMessage.textContent = payload.message || "";
  } catch (error) {
    apiMessage.textContent = error.message;
  } finally {
    requestBusy = false;
    await refreshStatus();
    await refreshLogs();
  }
}

startButton.addEventListener("click", () => {
  const selected = latestStatus?.tasks?.find((task) => task.id === latestStatus.selected_task_id);
  const taskText = selected ? `${selected.number}. ${selected.label}\n${selected.instruction}\n\n` : "";
  const confirmed = window.confirm(
    `${taskText}기존 run_lerobot_robot_client.sh를 실행합니다.\n추론 서버와 로봇 주변 안전을 확인했습니까?`
  );
  if (confirmed) post("/api/client/start");
});
stopButton.addEventListener("click", () => post("/api/client/stop"));
reconnectButton.addEventListener("click", () => post("/api/cameras/restart"));

refreshStatus();
refreshLogs();
setInterval(refreshStatus, 700);
setInterval(refreshLogs, 700);
