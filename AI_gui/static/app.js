const cameraSummary = document.querySelector("#cameraSummary");
const clientSummary = document.querySelector("#clientSummary");
const taskValue = document.querySelector("#taskValue");
const selectedTaskLabel = document.querySelector("#selectedTaskLabel");
const modelValue = document.querySelector("#modelValue");
const startButton = document.querySelector("#startButton");
const stopButton = document.querySelector("#stopButton");
const reconnectButton = document.querySelector("#reconnectButton");
const logOutput = document.querySelector("#logOutput");
const apiMessage = document.querySelector("#apiMessage");
const voiceOrb = document.querySelector("#voiceOrb");
const voiceState = document.querySelector("#voiceState");
const voiceTranscript = document.querySelector("#voiceTranscript");
const microphoneStatus = document.querySelector("#microphoneStatus");

const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
const PAUSE_WORDS = ["멈춰", "멈춰줘", "일시정지", "잠깐 멈춰", "스톱", "stop"];
const TERMINATE_WORDS = ["종료"];
const ACTION_WORDS = ["집어", "잡아", "들어", "옮겨", "시작", "실행", "해줘", "해주세요", "pick"];
const POLICY_LOADING_NOTICE_DELAY_MS = 10000;

let lastLogId = 0;
let requestBusy = false;
let latestStatus = null;
let recognition = null;
let recognitionRunning = false;
let voiceEnabled = false;
let speaking = false;
let commandInFlight = false;
let restartTimer = null;
let lastCommand = "";
let lastCommandAt = 0;
let pendingTranscript = "";
let pendingTranscriptTimer = null;
let policyLoadingNoticeArmed = false;
let policyLoadingNoticeStartedAt = 0;
let policyLoadingNoticeSpoken = false;

async function request(path, options = {}) {
  const response = await fetch(path, { cache: "no-store", ...options });
  let payload = {};
  try { payload = await response.json(); } catch (_) { /* ignore malformed error bodies */ }
  if (!response.ok) throw new Error(payload.message || `HTTP ${response.status}`);
  return payload;
}

function clientIsActive(status = latestStatus) {
  return ["starting", "running", "paused", "stopping"].includes(status?.client?.state);
}

function clientIsPaused(status = latestStatus) {
  return status?.client?.state === "paused";
}

function clientIsExecuting(status = latestStatus) {
  return ["starting", "running", "stopping"].includes(status?.client?.state);
}

function clientPhase(status = latestStatus) {
  return status?.client?.phase || status?.client?.state || "stopped";
}

function selectedTask(status = latestStatus) {
  return status?.tasks?.find((task) => task.id === status.selected_task_id) || null;
}

function setOrb(state, title, transcript) {
  voiceOrb.dataset.state = state;
  if (title) voiceState.textContent = title;
  if (transcript !== undefined) voiceTranscript.textContent = transcript;
}

function setMicrophone(text, active = false) {
  microphoneStatus.textContent = text;
  microphoneStatus.classList.toggle("active", active);
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

function updateVoiceFromRobot(status) {
  const task = selectedTask(status);
  if (task) {
    selectedTaskLabel.textContent = task.label;
    taskValue.textContent = task.instruction;
  }

  if (speaking || commandInFlight) return;
  if (clientIsPaused(status)) {
    setOrb("paused", `${task?.label || "VLA"} 작업 일시정지`, "새 작업을 말하면 연결을 유지한 채 다시 실행합니다.");
  } else if (clientIsActive(status)) {
    const activeTask = status.tasks.find((item) => item.id === status.client.active_task_id) || task;
    const phase = clientPhase(status);
    if (["launching", "robot_preparing"].includes(phase)) {
      setOrb("processing", `${activeTask?.label || "VLA"} 작업 준비 중`, "로봇과 카메라를 정책 실행 상태로 준비하고 있습니다.");
    } else if (phase === "policy_loading") {
      setOrb("processing", `${activeTask?.label || "VLA"} 정책 업로드 중`, "현재 작업을 수행하기 위한 정책을 업로드하고 있습니다.");
    } else {
      setOrb("executing", `${activeTask?.label || "VLA"} 작업 실행 중`, "“멈춰”는 일시정지, “종료해줘”는 완전 종료입니다.");
    }
  } else if (voiceEnabled && recognitionRunning) {
    setOrb("listening", "명령을 듣고 있습니다", "“접시를 집어줘” 또는 “컵을 집어줘”");
  } else if (!voiceEnabled) {
    setOrb("idle", "오브를 눌러 음성 정책 실행 모드로 전환하세요", "“접시를 집어줘” 또는 “컵을 집어줘”");
  }
}

async function refreshStatus() {
  try {
    const status = await request("/api/status");
    latestStatus = status;
    const active = clientIsActive(status);
    const connected = status.cameras.filter((camera) => camera.state === "connected").length;
    status.cameras.forEach(updateCamera);

    cameraSummary.textContent = `Cameras ${connected}/3`;
    cameraSummary.classList.toggle("good", connected === 3);
    clientSummary.textContent = status.client.state === "running"
      ? `Client ${clientPhase(status)}`
      : `Client ${status.client.state}`;
    clientSummary.classList.toggle("busy", active);
    modelValue.textContent = status.model;
    startButton.disabled = clientIsExecuting(status) || requestBusy || commandInFlight;
    stopButton.disabled = !active || status.client.state === "stopping" || requestBusy;
    reconnectButton.disabled = active || requestBusy;
    apiMessage.textContent = "";
    updateVoiceFromRobot(status);
    maybeAnnouncePolicyLoading(status);
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

async function controlPost(path) {
  const payload = await request(path, {
    method: "POST",
    headers: { "X-RBY1-Control": "ai-v3" },
  });
  apiMessage.textContent = payload.message || "";
  return payload;
}

async function post(path) {
  requestBusy = true;
  await refreshStatus();
  try {
    return await controlPost(path);
  } catch (error) {
    apiMessage.textContent = error.message;
    throw error;
  } finally {
    requestBusy = false;
    await refreshStatus();
    await refreshLogs();
  }
}

// Keep every policy launch on the same path as the proven Web GUI flow.
// Voice recognition only chooses the task; selection and execution use the
// same request wrapper as the manual Execute control below.
async function selectPolicyTask(task) {
  return post(`/api/task/select?id=${encodeURIComponent(task.id)}`);
}

async function startSelectedPolicy() {
  return post("/api/client/start");
}

async function pausePolicyClient() {
  return post("/api/client/pause");
}

async function stopPolicyClient() {
  return post("/api/client/stop");
}

function stopRecognition() {
  clearTimeout(restartTimer);
  if (recognition && recognitionRunning) {
    try { recognition.abort(); } catch (_) { /* already stopping */ }
  }
}

function scheduleRecognitionRestart(delay = 350) {
  clearTimeout(restartTimer);
  if (!voiceEnabled || speaking || commandInFlight) return;
  restartTimer = window.setTimeout(() => {
    if (!recognitionRunning) startRecognition();
  }, delay);
}

function startRecognition() {
  if (!recognition || recognitionRunning || !voiceEnabled || speaking || commandInFlight) return;
  try {
    recognition.start();
  } catch (error) {
    if (error.name !== "InvalidStateError") {
      setOrb("error", "마이크를 시작하지 못했습니다", error.message);
    }
  }
}

function koreanVoice() {
  if (!("speechSynthesis" in window)) return null;
  const voices = window.speechSynthesis.getVoices();
  return voices.find((voice) => voice.lang.toLowerCase().startsWith("ko")) || null;
}

function speak(text, onDone) {
  stopRecognition();
  if (!("speechSynthesis" in window)) {
    if (onDone) onDone();
    else scheduleRecognitionRestart();
    return;
  }

  window.speechSynthesis.cancel();
  const utterance = new SpeechSynthesisUtterance(text);
  utterance.lang = "ko-KR";
  utterance.rate = 0.96;
  utterance.pitch = 0.94;
  const voice = koreanVoice();
  if (voice) utterance.voice = voice;
  speaking = true;
  setOrb("speaking", "응답하고 있습니다", text);
  const finish = () => {
    if (!speaking) return;
    speaking = false;
    if (onDone) onDone();
    else scheduleRecognitionRestart(250);
  };
  utterance.onend = finish;
  utterance.onerror = finish;
  window.speechSynthesis.speak(utterance);
}

function armPolicyLoadingNotice() {
  policyLoadingNoticeArmed = true;
  policyLoadingNoticeStartedAt = Date.now();
  policyLoadingNoticeSpoken = false;
}

function clearPolicyLoadingNotice() {
  policyLoadingNoticeArmed = false;
  policyLoadingNoticeStartedAt = 0;
  policyLoadingNoticeSpoken = false;
}

function maybeAnnouncePolicyLoading(status) {
  if (!policyLoadingNoticeArmed || policyLoadingNoticeSpoken) return;
  if (["stopped", "stopping", "paused"].includes(status.client.state)) {
    clearPolicyLoadingNotice();
    return;
  }
  if (clientPhase(status) === "executing") {
    clearPolicyLoadingNotice();
    return;
  }
  const waitedLongEnough = Date.now() - policyLoadingNoticeStartedAt >= POLICY_LOADING_NOTICE_DELAY_MS;
  if (clientPhase(status) === "policy_loading" && waitedLongEnough && !speaking && !commandInFlight) {
    policyLoadingNoticeSpoken = true;
    speak("현재 작업을 수행하기 위한 정책을 업로드하고 있습니다.");
  }
}

function normalizeSpeech(text) {
  return text.toLowerCase().replace(/[\s.,!?~]+/g, "");
}

function findSpokenTask(transcript) {
  const normalized = normalizeSpeech(transcript);
  const hasAction = ACTION_WORDS.some((word) => normalized.includes(normalizeSpeech(word)));
  if (!hasAction) return null;
  return latestStatus?.tasks?.find((task) =>
    task.voice_keywords.some((keyword) => normalized.includes(normalizeSpeech(keyword)))
  ) || null;
}

function isPauseCommand(transcript) {
  const normalized = normalizeSpeech(transcript);
  return PAUSE_WORDS.some((word) => normalized.includes(normalizeSpeech(word)));
}

function isTerminateCommand(transcript) {
  const normalized = normalizeSpeech(transcript);
  return TERMINATE_WORDS.some((word) => normalized.includes(normalizeSpeech(word)));
}

async function pauseByVoice() {
  commandInFlight = true;
  stopRecognition();
  setOrb("processing", "일시정지 명령을 처리합니다", "연결은 유지하고 정책 액션만 멈추는 중입니다.");
  try {
    if (latestStatus?.client?.state === "running" && clientPhase() === "executing") {
      await pausePolicyClient();
      speak("로봇 동작을 일시정지합니다.");
    } else if (latestStatus?.client?.state === "running") {
      speak("현재 작업을 수행하기 위한 정책을 준비하고 있습니다.");
    } else if (clientIsPaused()) {
      speak("현재 로봇 동작은 이미 일시정지되어 있습니다.");
    } else {
      speak("현재 실행 중인 작업이 없습니다.");
    }
  } catch (error) {
    setOrb("error", "일시정지 요청에 실패했습니다", error.message);
    speak("일시정지 요청에 실패했습니다.");
  } finally {
    commandInFlight = false;
    await refreshStatus();
  }
}

async function terminateByVoice() {
  commandInFlight = true;
  stopRecognition();
  setOrb("processing", "종료 명령을 처리합니다", "로봇 클라이언트와 정책 연결을 종료하는 중입니다.");
  try {
    if (clientIsActive()) {
      await stopPolicyClient();
      clearPolicyLoadingNotice();
      speak("로봇 클라이언트를 종료합니다.");
    } else {
      speak("현재 종료할 작업이 없습니다.");
    }
  } catch (error) {
    setOrb("error", "종료 요청에 실패했습니다", error.message);
    speak("종료 요청에 실패했습니다.");
  } finally {
    commandInFlight = false;
    await refreshStatus();
  }
}

async function executeVoiceTask(task) {
  if (clientIsExecuting()) {
    speak("현재 작업을 실행하고 있습니다. 먼저 멈춰라고 말해 주세요.");
    return;
  }

  const resumingPausedPolicy = clientIsPaused();
  commandInFlight = true;
  stopRecognition();
  setOrb("processing", `${task.label} 명령을 확인했습니다`, task.instruction);
  try {
    if (!resumingPausedPolicy) armPolicyLoadingNotice();
    await selectPolicyTask(task);
    setOrb("processing", `${task.label} 정책을 시작합니다`, "카메라를 정책 클라이언트에 연결하는 중입니다.");
    await startSelectedPolicy();
    commandInFlight = false;
    setOrb("executing", `${task.label} 작업 실행 중`, "“멈춰”는 일시정지, “종료해줘”는 완전 종료입니다.");
    speak(task.voice_response);
    await refreshStatus();
    await refreshLogs();
  } catch (error) {
    if (!resumingPausedPolicy) clearPolicyLoadingNotice();
    commandInFlight = false;
    setOrb("error", "음성 명령 처리에 실패했습니다", error.message);
    speak("명령 처리에 실패했습니다.");
  }
}

function flushPendingTranscript() {
  clearTimeout(pendingTranscriptTimer);
  const transcript = pendingTranscript.trim();
  pendingTranscript = "";
  if (transcript && !commandInFlight) handleFinalTranscript(transcript);
}

function queueFinalTranscript(transcript) {
  pendingTranscript = `${pendingTranscript} ${transcript}`.trim();
  voiceTranscript.textContent = `“${pendingTranscript}”`;
  clearTimeout(pendingTranscriptTimer);
  // Pause/terminate words should react quickly. Object commands get a short window so
  // Chrome can combine results such as "접시를" + "집어줘".
  const delay = (isPauseCommand(pendingTranscript) || isTerminateCommand(pendingTranscript)) ? 120 : 700;
  pendingTranscriptTimer = window.setTimeout(flushPendingTranscript, delay);
}

function handleFinalTranscript(transcript) {
  const now = Date.now();
  const normalized = normalizeSpeech(transcript);
  if (normalized === lastCommand && now - lastCommandAt < 3000) return;
  lastCommand = normalized;
  lastCommandAt = now;
  voiceTranscript.textContent = `“${transcript}”`;

  if (isTerminateCommand(transcript)) {
    terminateByVoice();
    return;
  }
  if (isPauseCommand(transcript)) {
    pauseByVoice();
    return;
  }
  if (clientIsExecuting()) {
    if (clientPhase() === "executing") {
      speak("현재 작업을 실행하고 있습니다.");
    } else {
      speak("현재 작업을 수행하기 위한 정책을 준비하고 있습니다.");
    }
    return;
  }
  const task = findSpokenTask(transcript);
  if (task) {
    executeVoiceTask(task);
    return;
  }
  speak("명령을 이해하지 못했습니다. 접시를 집어줘, 컵을 집어줘, 멈춰, 또는 종료해줘라고 말해 주세요.");
}

function initializeRecognition() {
  if (!SpeechRecognition) {
    setMicrophone("Unsupported browser");
    setOrb("error", "이 브라우저는 음성인식을 지원하지 않습니다", "노트북의 Chrome 또는 Edge 브라우저를 사용해 주세요.");
    voiceOrb.disabled = true;
    return;
  }

  recognition = new SpeechRecognition();
  recognition.lang = "ko-KR";
  recognition.continuous = true;
  recognition.interimResults = true;
  recognition.maxAlternatives = 3;

  recognition.onstart = () => {
    recognitionRunning = true;
    setMicrophone("Listening · ko-KR", true);
    if (!clientIsActive() && !speaking && !commandInFlight) {
      setOrb("listening", "명령을 듣고 있습니다", "말씀하세요…");
    }
  };
  recognition.onspeechstart = () => {
    if (!clientIsActive() && !commandInFlight) setOrb("listening", "음성을 감지했습니다", "듣고 있습니다…");
  };
  recognition.onresult = (event) => {
    let interim = "";
    const finals = [];
    for (let index = event.resultIndex; index < event.results.length; index += 1) {
      const result = event.results[index];
      if (result.isFinal) finals.push(result[0].transcript.trim());
      else interim += result[0].transcript;
    }
    if (interim && !commandInFlight) voiceTranscript.textContent = interim;
    if (finals.length && !commandInFlight) queueFinalTranscript(finals.join(" "));
  };
  recognition.onerror = (event) => {
    if (event.error === "aborted" || event.error === "no-speech") return;
    if (event.error === "not-allowed" || event.error === "service-not-allowed") {
      voiceEnabled = false;
      setMicrophone("Permission denied");
      setOrb("error", "마이크 권한이 필요합니다", "브라우저 주소창의 마이크 권한을 허용해 주세요.");
      return;
    }
    setMicrophone(`Error · ${event.error}`);
    if (!clientIsActive()) setOrb("error", "음성인식 오류", event.error);
  };
  recognition.onend = () => {
    recognitionRunning = false;
    if (!voiceEnabled) setMicrophone("Voice control off");
    scheduleRecognitionRestart();
  };
}

voiceOrb.addEventListener("click", () => {
  if (!recognition) return;
  voiceEnabled = !voiceEnabled;
  if (voiceEnabled) {
    setMicrophone("Starting…", true);
    setOrb("processing", "음성 정책 실행 모드로 전환합니다", "잠시 후 명령을 말씀하세요.");
    startRecognition();
  } else {
    clearTimeout(pendingTranscriptTimer);
    pendingTranscript = "";
    stopRecognition();
    setMicrophone("Voice control off");
    setOrb("idle", "음성 제어가 꺼졌습니다", "오브를 누르면 다시 시작합니다.");
  }
});

startButton.addEventListener("click", () => {
  const task = selectedTask();
  const taskText = task ? `${task.number}. ${task.label}\n${task.instruction}\n\n` : "";
  const confirmed = window.confirm(
    `${taskText}선택한 정책을 수동 실행합니다.\n추론 서버와 로봇 주변 안전을 확인했습니까?`
  );
  if (confirmed) startSelectedPolicy().catch(() => {});
});
stopButton.addEventListener("click", () => stopPolicyClient().catch(() => {}));
reconnectButton.addEventListener("click", () => post("/api/cameras/restart").catch(() => {}));

initializeRecognition();
refreshStatus();
refreshLogs();
setInterval(refreshStatus, 700);
setInterval(refreshLogs, 700);
