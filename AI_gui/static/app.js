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
const sttStatus = document.querySelector("#sttStatus");
const langChip = document.querySelector("#langChip");

// Whisper reports which language each utterance was spoken in, and every
// operator-facing string is looked up by key so the console can answer a
// Korean speaker in Korean and an English speaker in English. Field labels
// that were already English stay English in both languages.
const MESSAGES = {
  ko: {
    "camera.idle": "대기 중",
    "camera.connecting": "카메라 연결 중",
    "camera.connected": "연결됨 · {fps} FPS",
    "camera.failed": "연결 실패: {error}",
    "camera.stopped": "미리보기 중지됨",
    "camera.relay_live": "정책 프레임 공유 중 · {fps} FPS",
    "camera.relay_wait": "정책 프레임 대기 중",

    "orb.ariaLabel": "음성 인식 시작",
    "orb.idle.title": "오브를 눌러 음성 정책 실행 모드로 전환하세요",
    "orb.hint.examples": "“그릇을 집어줘” 또는 “컵을 집어줘”",
    "orb.off.title": "음성 제어가 꺼졌습니다",
    "orb.off.hint": "오브를 누르면 다시 시작합니다.",
    "orb.starting.title": "음성 정책 실행 모드로 전환합니다",
    "orb.starting.hint": "잠시 후 명령을 말씀하세요.",
    "orb.listening.title": "명령을 듣고 있습니다",
    "orb.listening.hint": "말씀하세요…",
    "orb.speech.title": "음성을 감지했습니다",
    "orb.speech.hint": "듣고 있습니다…",
    "orb.transcribing.title": "음성을 인식하는 중입니다",
    "orb.transcribing.hint": "잠시만 기다려 주세요…",
    "orb.speaking.title": "응답하고 있습니다",
    "orb.confirmed.title": "{label} 명령을 확인했습니다",
    "orb.startingPolicy.title": "{label} 정책을 시작합니다",
    "orb.startingPolicy.hint": "카메라를 정책 클라이언트에 연결하는 중입니다.",
    "orb.executing.title": "{label} 작업 실행 중",
    "orb.executing.hint": "“멈춰”는 일시정지, “종료해줘”는 완전 종료입니다.",
    "orb.paused.title": "{label} 작업 일시정지",
    "orb.paused.hint": "새 작업을 말하면 연결을 유지한 채 다시 실행합니다.",
    "orb.preparing.title": "{label} 작업 준비 중",
    "orb.preparing.hint": "로봇과 카메라를 정책 실행 상태로 준비하고 있습니다.",
    "orb.policyLoading.title": "{label} 정책 업로드 중",
    "orb.policyLoading.hint": "현재 작업을 수행하기 위한 정책을 업로드하고 있습니다.",
    "orb.pauseCommand.title": "일시정지 명령을 처리합니다",
    "orb.pauseCommand.hint": "연결은 유지하고 정책 액션만 멈추는 중입니다.",
    "orb.terminateCommand.title": "종료 명령을 처리합니다",
    "orb.terminateCommand.hint": "로봇 클라이언트와 정책 연결을 종료하는 중입니다.",
    "orb.error.pause": "일시정지 요청에 실패했습니다",
    "orb.error.terminate": "종료 요청에 실패했습니다",
    "orb.error.command": "음성 명령 처리에 실패했습니다",
    "orb.error.mic": "마이크를 시작하지 못했습니다",
    "orb.error.permission.title": "마이크 권한이 필요합니다",
    "orb.error.permission.hint": "브라우저 주소창의 마이크 권한을 허용해 주세요.",
    "orb.error.insecure.title": "보안 컨텍스트가 아니어서 마이크를 열 수 없습니다",
    "orb.error.insecure.hint":
      "localhost로 접속하거나, Chrome의 안전하지 않은 원본 허용 설정에 이 주소를 추가해 주세요.",
    "orb.error.unsupported.title": "이 브라우저는 음성 입력을 지원하지 않습니다",
    "orb.error.unsupported.hint": "노트북의 Chrome 또는 Edge 브라우저를 사용해 주세요.",

    "speak.pausing": "로봇 동작을 일시정지합니다.",
    "speak.policyPreparing": "현재 작업을 수행하기 위한 정책을 준비하고 있습니다.",
    "speak.alreadyPaused": "현재 로봇 동작은 이미 일시정지되어 있습니다.",
    "speak.nothingRunning": "현재 실행 중인 작업이 없습니다.",
    "speak.pauseFailed": "일시정지 요청에 실패했습니다.",
    "speak.terminating": "로봇 클라이언트를 종료합니다.",
    "speak.nothingToTerminate": "현재 종료할 작업이 없습니다.",
    "speak.terminateFailed": "종료 요청에 실패했습니다.",
    "speak.busy": "현재 작업을 실행하고 있습니다. 먼저 멈춰라고 말해 주세요.",
    "speak.executing": "현재 작업을 실행하고 있습니다.",
    "speak.policyUploading": "현재 작업을 수행하기 위한 정책을 업로드하고 있습니다.",
    "speak.commandFailed": "명령 처리에 실패했습니다.",
    "speak.notUnderstood":
      "명령을 이해하지 못했습니다. 그릇을 집어줘, 컵을 집어줘, 멈춰, 또는 종료해줘라고 말해 주세요.",

    "example.bowl": "그릇을 집어줘",
    "example.cup": "컵을 집어줘",
    "example.pause": "멈춰 · 일시정지",
    "example.terminate": "종료해줘 · 완전 종료",

    "ui.note": "“멈춰”는 연결을 유지한 일시정지, “종료해줘”는 로봇 클라이언트 완전 종료입니다.",
    "ui.reconnect": "↻ 카메라 다시 연결",
    "ui.manualRun": "▶ 선택 작업 수동 실행",
    "ui.stopClient": "■ 클라이언트 정지",
    "ui.confirm": "{task}선택한 정책을 수동 실행합니다.\n추론 서버와 로봇 주변 안전을 확인했습니까?",
    "ui.serverError": "서버 연결 오류: {error}",
  },
  en: {
    "camera.idle": "Idle",
    "camera.connecting": "Connecting camera",
    "camera.connected": "Connected · {fps} FPS",
    "camera.failed": "Connection failed: {error}",
    "camera.stopped": "Preview stopped",
    "camera.relay_live": "Sharing policy frames · {fps} FPS",
    "camera.relay_wait": "Waiting for policy frames",

    "orb.ariaLabel": "Start voice recognition",
    "orb.idle.title": "Tap the orb to switch to voice policy control",
    "orb.hint.examples": "“Pick up the bowl” or “Pick up the cup”",
    "orb.off.title": "Voice control is off",
    "orb.off.hint": "Tap the orb to start again.",
    "orb.starting.title": "Switching to voice policy control",
    "orb.starting.hint": "You can speak in a moment.",
    "orb.listening.title": "Listening for a command",
    "orb.listening.hint": "Go ahead…",
    "orb.speech.title": "Speech detected",
    "orb.speech.hint": "Listening…",
    "orb.transcribing.title": "Recognizing speech",
    "orb.transcribing.hint": "One moment…",
    "orb.speaking.title": "Responding",
    "orb.confirmed.title": "{label} command confirmed",
    "orb.startingPolicy.title": "Starting the {label} policy",
    "orb.startingPolicy.hint": "Handing the cameras over to the policy client.",
    "orb.executing.title": "Running task {label}",
    "orb.executing.hint": "Say “stop” to pause, “terminate” to shut down.",
    "orb.paused.title": "Task {label} paused",
    "orb.paused.hint": "Say a new task to run again without reconnecting.",
    "orb.preparing.title": "Preparing task {label}",
    "orb.preparing.hint": "Getting the robot and cameras ready to run the policy.",
    "orb.policyLoading.title": "Uploading the {label} policy",
    "orb.policyLoading.hint": "Uploading the policy for the current task.",
    "orb.pauseCommand.title": "Handling the pause command",
    "orb.pauseCommand.hint": "Keeping the connection and stopping policy actions only.",
    "orb.terminateCommand.title": "Handling the shutdown command",
    "orb.terminateCommand.hint": "Shutting down the robot client and the policy connection.",
    "orb.error.pause": "The pause request failed",
    "orb.error.terminate": "The shutdown request failed",
    "orb.error.command": "The voice command failed",
    "orb.error.mic": "Could not start the microphone",
    "orb.error.permission.title": "Microphone permission is required",
    "orb.error.permission.hint": "Allow microphone access from the browser address bar.",
    "orb.error.insecure.title": "The microphone needs a secure context",
    "orb.error.insecure.hint":
      "Open this page on localhost, or add this address to Chrome's insecure-origins setting.",
    "orb.error.unsupported.title": "This browser does not support voice input",
    "orb.error.unsupported.hint": "Please use Chrome or Edge on a laptop.",

    "speak.pausing": "Pausing the robot.",
    "speak.policyPreparing": "I am still preparing the policy for the current task.",
    "speak.alreadyPaused": "The robot is already paused.",
    "speak.nothingRunning": "There is no task running right now.",
    "speak.pauseFailed": "The pause request failed.",
    "speak.terminating": "Shutting down the robot client.",
    "speak.nothingToTerminate": "There is nothing to shut down right now.",
    "speak.terminateFailed": "The shutdown request failed.",
    "speak.busy": "A task is already running. Please say stop first.",
    "speak.executing": "A task is running right now.",
    "speak.policyUploading": "I am uploading the policy for the current task.",
    "speak.commandFailed": "The command failed.",
    "speak.notUnderstood":
      "I did not catch that. Say pick up the bowl, pick up the cup, stop, or terminate.",

    "example.bowl": "Pick up the bowl",
    "example.cup": "Pick up the cup",
    "example.pause": "Stop · pause",
    "example.terminate": "Terminate · full shutdown",

    "ui.note":
      "“Stop” pauses while staying connected; “terminate” shuts the robot client down completely.",
    "ui.reconnect": "↻ Reconnect cameras",
    "ui.manualRun": "▶ Run selected task manually",
    "ui.stopClient": "■ Stop client",
    "ui.confirm":
      "{task}This runs the selected policy manually.\nHave you checked the inference server and the area around the robot?",
    "ui.serverError": "Server connection error: {error}",
  },
};

// Status-panel readouts stay in English in both languages, matching the
// English field labels they sit under.
const NEUTRAL_MESSAGES = {
  "mic.notActivated": "Not activated",
  "mic.off": "Voice control off",
  "mic.starting": "Starting…",
  "mic.listening": "Listening · {mode}",
  "mic.speech": "Speech · {mode}",
  "mic.denied": "Permission denied",
  "mic.unsupported": "Unsupported browser",
  "mic.insecure": "Insecure origin",
  "mic.error": "Error · {error}",
  "stt.idle": "Idle",
  "stt.loading": "Loading {model}…",
  "stt.ready": "{model} · {device}",
  "stt.error": "Error · {detail}",
  "stt.disabled": "Disabled",
};

// Used until /api/status delivers the configured word lists.
const FALLBACK_WORDS = {
  action_words: {
    ko: ["집어", "잡아", "들어", "옮겨", "시작", "실행", "해줘", "해주세요"],
    en: ["pick", "pick up", "grab", "take", "move", "start", "run", "execute", "please"],
  },
  pause_words: {
    ko: ["멈춰", "멈춰줘", "일시정지", "잠깐 멈춰", "그만", "스톱"],
    en: ["stop", "pause", "hold on", "wait"],
  },
  terminate_words: {
    ko: ["종료", "끝내"],
    en: ["terminate", "shut down", "shutdown", "quit", "exit", "end it"],
  },
};

const DEFAULT_VAD = {
  min_rms: 0.012,
  noise_multiplier: 3.0,
  start_ms: 96,
  end_ms: 700,
  preroll_ms: 384,
  min_seconds: 0.25,
  max_seconds: 12,
};
const FRAME_SAMPLES = 512;
const CAPTURE_SAMPLE_RATE = 16000;
const POLICY_LOADING_NOTICE_DELAY_MS = 10000;
const SPEECH_TAIL_GUARD_MS = 300;

let currentLang = "ko";
let langMode = "auto";
let availableLanguages = ["ko", "en"];
let sttEndpoint = "http://localhost:8002";
let sttHealth = { state: "idle" };

let lastLogId = 0;
let requestBusy = false;
let latestStatus = null;
let voiceEnabled = false;
let speaking = false;
let transcribing = false;
let commandInFlight = false;
let lastCommand = "";
let lastCommandAt = 0;
let policyLoadingNoticeArmed = false;
let policyLoadingNoticeStartedAt = 0;
let policyLoadingNoticeSpoken = false;

let audioContext = null;
let mediaStream = null;
let sourceNode = null;
let workletNode = null;
let captureReady = false;
let vadSettings = { ...DEFAULT_VAD };
let noiseFloor = 0.004;
let loudFrames = 0;
let quietFrames = 0;
let collecting = false;
let preRollFrames = [];
let utteranceFrames = [];
let micGateUntil = 0;

let orbView = { state: "idle", titleKey: "orb.idle.title", hintKey: "orb.hint.examples", hintText: null, params: {} };
let micView = { key: "mic.notActivated", params: {}, active: false };

/* ---------------------------------------------------------------- i18n --- */

function t(key, params = {}) {
  const table = MESSAGES[currentLang] || MESSAGES.ko;
  const template = table[key] ?? MESSAGES.ko[key] ?? NEUTRAL_MESSAGES[key] ?? key;
  return template.replace(/\{(\w+)\}/g, (match, name) =>
    params[name] === undefined ? match : String(params[name]),
  );
}

function localized(value) {
  if (!value) return "";
  if (typeof value === "string") return value;
  return value[currentLang] || value.ko || value.en || Object.values(value)[0] || "";
}

function applyStaticTranslations() {
  document.querySelectorAll("[data-i18n]").forEach((node) => {
    node.textContent = t(node.dataset.i18n);
  });
  document.querySelectorAll("[data-i18n-aria]").forEach((node) => {
    node.setAttribute("aria-label", t(node.dataset.i18nAria));
  });
}

function setLanguage(language) {
  const next = MESSAGES[language] ? language : "ko";
  if (next === currentLang) return;
  currentLang = next;
  document.documentElement.lang = next;
  applyStaticTranslations();
  renderOrb();
  renderMicrophone();
  renderLanguageChip();
  if (latestStatus) (latestStatus.cameras || []).forEach(updateCamera);
  renderSttStatus();
}

function languageModeLabel() {
  return langMode === "auto" ? `auto (${availableLanguages.join("/")})` : langMode;
}

function renderLanguageChip() {
  langChip.textContent = langMode === "auto" ? `AUTO · ${currentLang.toUpperCase()}` : currentLang.toUpperCase();
  langChip.classList.toggle("pinned", langMode !== "auto");
}

/* ------------------------------------------------------------- rendering --- */

function setOrb(state, titleKey, hintKey, params = {}) {
  orbView = { state, titleKey, hintKey, hintText: null, params };
  renderOrb();
}

function setOrbText(state, titleKey, hintText, params = {}) {
  orbView = { state, titleKey, hintKey: null, hintText, params };
  renderOrb();
}

function renderOrb() {
  voiceOrb.dataset.state = orbView.state;
  voiceState.textContent = t(orbView.titleKey, orbView.params);
  if (orbView.hintText !== null) voiceTranscript.textContent = orbView.hintText;
  else if (orbView.hintKey) voiceTranscript.textContent = t(orbView.hintKey, orbView.params);
}

function setMicrophone(key, params = {}, active = false) {
  micView = { key, params, active };
  renderMicrophone();
}

function renderMicrophone() {
  microphoneStatus.textContent = t(micView.key, micView.params);
  microphoneStatus.classList.toggle("active", micView.active);
}

function renderSttStatus() {
  const keys = {
    ready: "stt.ready",
    loading: "stt.loading",
    error: "stt.error",
    disabled: "stt.disabled",
    idle: "stt.idle",
  };
  sttStatus.textContent = t(keys[sttHealth.state] || "stt.idle", {
    model: sttHealth.model || "whisper",
    device: sttHealth.device || "?",
    detail: sttHealth.detail || "",
  });
  sttStatus.classList.toggle("active", sttHealth.state === "ready");
}

/* --------------------------------------------------------------- status --- */

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

function updateCamera(camera) {
  const card = document.querySelector(`[data-camera="${camera.key}"]`);
  if (!card) return;
  const overlay = card.querySelector(".camera-overlay");
  const meta = card.querySelector(".camera-meta");
  const message = t(camera.message_key || "camera.idle", camera.message_params || {});
  card.classList.toggle("connected", camera.state === "connected");
  card.classList.toggle("error", camera.state === "error");
  overlay.textContent = message;
  meta.textContent = `${camera.serial} · ${message}`;
}

function updateVoiceFromRobot(status) {
  const task = selectedTask(status);
  if (task) {
    selectedTaskLabel.textContent = task.label;
    taskValue.textContent = task.instruction;
  }

  if (speaking || commandInFlight || transcribing) return;
  if (clientIsPaused(status)) {
    setOrb("paused", "orb.paused.title", "orb.paused.hint", { label: task?.label || "VLA" });
  } else if (clientIsActive(status)) {
    const activeTask = status.tasks.find((item) => item.id === status.client.active_task_id) || task;
    const label = activeTask?.label || "VLA";
    const phase = clientPhase(status);
    if (["launching", "robot_preparing"].includes(phase)) {
      setOrb("processing", "orb.preparing.title", "orb.preparing.hint", { label });
    } else if (phase === "policy_loading") {
      setOrb("processing", "orb.policyLoading.title", "orb.policyLoading.hint", { label });
    } else {
      setOrb("executing", "orb.executing.title", "orb.executing.hint", { label });
    }
  } else if (voiceEnabled && captureReady) {
    setOrb("listening", "orb.listening.title", "orb.hint.examples");
  } else if (!voiceEnabled) {
    setOrb("idle", "orb.idle.title", "orb.hint.examples");
  }
}

async function refreshStatus() {
  try {
    const status = await request("/api/status");
    latestStatus = status;
    if (Array.isArray(status.languages) && status.languages.length) {
      availableLanguages = status.languages;
    }
    if (status.voice && status.voice.vad) {
      vadSettings = { ...DEFAULT_VAD, ...status.voice.vad };
    }
    if (status.stt_endpoint) sttEndpoint = status.stt_endpoint.replace(/\/+$/, "");

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
    apiMessage.textContent = t("ui.serverError", { error: error.message });
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
// Speech recognition only chooses the task; selection and execution use the
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

/* ------------------------------------------------------------------ TTS --- */

function voiceForLanguage(language) {
  if (!("speechSynthesis" in window)) return null;
  const prefix = language === "en" ? "en" : "ko";
  const voices = window.speechSynthesis.getVoices() || [];
  return voices.find((voice) => voice.lang.toLowerCase().replace("_", "-").startsWith(prefix)) || null;
}

function speak(text, onDone) {
  resetUtterance();
  if (!text || !("speechSynthesis" in window)) {
    if (onDone) onDone();
    return;
  }

  window.speechSynthesis.cancel();
  const utterance = new SpeechSynthesisUtterance(text);
  utterance.lang = currentLang === "en" ? "en-US" : "ko-KR";
  utterance.rate = 0.96;
  utterance.pitch = 0.94;
  const voice = voiceForLanguage(currentLang);
  if (voice) utterance.voice = voice;
  speaking = true;
  setOrbText("speaking", "orb.speaking.title", text);
  const finish = () => {
    if (!speaking) return;
    speaking = false;
    // Do not let the tail of our own answer open a new utterance.
    micGateUntil = Date.now() + SPEECH_TAIL_GUARD_MS;
    resetUtterance();
    if (onDone) onDone();
    else restoreIdleOrb();
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
    speak(t("speak.policyUploading"));
  }
}

/* -------------------------------------------------------------- matching --- */

function normalizeCompact(text) {
  return text.toLowerCase().replace(/[\s.,!?~"'`‘’“”·…\-()]+/g, "");
}

function normalizeWords(text) {
  return ` ${text.toLowerCase().replace(/[^a-z0-9]+/g, " ").trim()} `;
}

// Korean terms match on the compacted string because spacing and particles
// vary ("컵을 집어줘" / "컵 집어"). Latin terms match on word boundaries so
// "stop" does not fire inside an unrelated word.
function matchesTerm(text, term) {
  if (/[가-힣]/.test(term)) return normalizeCompact(text).includes(normalizeCompact(term));
  const needle = term.toLowerCase().replace(/[^a-z0-9]+/g, " ").trim();
  return needle ? normalizeWords(text).includes(` ${needle} `) : false;
}

function matchesAny(text, groups) {
  if (!groups) return false;
  const lists = Array.isArray(groups) ? [groups] : Object.values(groups);
  return lists.some((list) => Array.isArray(list) && list.some((term) => matchesTerm(text, term)));
}

function wordGroup(name) {
  return latestStatus?.voice?.[name] || FALLBACK_WORDS[name];
}

// Keywords from every configured language are checked, so a Korean speaker
// saying "cup" and an English speaker saying "bowl" both land correctly; only
// the reply language follows what Whisper detected.
function taskByKeyword(text) {
  return (latestStatus?.tasks || []).find((task) => matchesAny(text, task.voice_keywords)) || null;
}

function findSpokenTask(text) {
  const task = taskByKeyword(text);
  if (!task) return null;
  return matchesAny(text, wordGroup("action_words")) ? task : null;
}

/* -------------------------------------------------------------- commands --- */

function restoreIdleOrb() {
  if (speaking || commandInFlight || transcribing) return;
  if (clientIsActive()) {
    updateVoiceFromRobot(latestStatus);
    return;
  }
  if (voiceEnabled) setOrb("listening", "orb.listening.title", "orb.hint.examples");
  else setOrb("idle", "orb.idle.title", "orb.hint.examples");
}

async function pauseByVoice() {
  commandInFlight = true;
  resetUtterance();
  setOrb("processing", "orb.pauseCommand.title", "orb.pauseCommand.hint");
  try {
    if (latestStatus?.client?.state === "running" && clientPhase() === "executing") {
      await pausePolicyClient();
      speak(t("speak.pausing"));
    } else if (latestStatus?.client?.state === "running") {
      speak(t("speak.policyPreparing"));
    } else if (clientIsPaused()) {
      speak(t("speak.alreadyPaused"));
    } else {
      speak(t("speak.nothingRunning"));
    }
  } catch (error) {
    setOrbText("error", "orb.error.pause", error.message);
    speak(t("speak.pauseFailed"));
  } finally {
    commandInFlight = false;
    await refreshStatus();
  }
}

async function terminateByVoice() {
  commandInFlight = true;
  resetUtterance();
  setOrb("processing", "orb.terminateCommand.title", "orb.terminateCommand.hint");
  try {
    if (clientIsActive()) {
      await stopPolicyClient();
      clearPolicyLoadingNotice();
      speak(t("speak.terminating"));
    } else {
      speak(t("speak.nothingToTerminate"));
    }
  } catch (error) {
    setOrbText("error", "orb.error.terminate", error.message);
    speak(t("speak.terminateFailed"));
  } finally {
    commandInFlight = false;
    await refreshStatus();
  }
}

async function executeVoiceTask(task) {
  if (clientIsExecuting()) {
    speak(t("speak.busy"));
    return;
  }

  const resumingPausedPolicy = clientIsPaused();
  commandInFlight = true;
  resetUtterance();
  setOrbText("processing", "orb.confirmed.title", task.instruction, { label: task.label });
  try {
    if (!resumingPausedPolicy) armPolicyLoadingNotice();
    await selectPolicyTask(task);
    setOrb("processing", "orb.startingPolicy.title", "orb.startingPolicy.hint", { label: task.label });
    await startSelectedPolicy();
    commandInFlight = false;
    setOrb("executing", "orb.executing.title", "orb.executing.hint", { label: task.label });
    speak(localized(task.voice_response));
    await refreshStatus();
    await refreshLogs();
  } catch (error) {
    if (!resumingPausedPolicy) clearPolicyLoadingNotice();
    commandInFlight = false;
    setOrbText("error", "orb.error.command", error.message);
    speak(t("speak.commandFailed"));
  }
}

function handleTranscript(text, language) {
  setLanguage(langMode === "auto" ? language : langMode);

  const now = Date.now();
  const fingerprint = normalizeCompact(text);
  if (fingerprint === lastCommand && now - lastCommandAt < 3000) {
    restoreIdleOrb();
    return;
  }
  lastCommand = fingerprint;
  lastCommandAt = now;
  setOrbText(orbView.state, orbView.titleKey, `“${text}”`, orbView.params);

  if (matchesAny(text, wordGroup("terminate_words"))) {
    terminateByVoice();
    return;
  }
  if (matchesAny(text, wordGroup("pause_words"))) {
    pauseByVoice();
    return;
  }

  const task = findSpokenTask(text);
  const mentionsObject = Boolean(taskByKeyword(text));
  const mentionsAction = matchesAny(text, wordGroup("action_words"));

  if (clientIsExecuting()) {
    if (task || mentionsObject || mentionsAction) {
      speak(clientPhase() === "executing" ? t("speak.executing") : t("speak.policyPreparing"));
    } else {
      restoreIdleOrb();
    }
    return;
  }
  if (task) {
    executeVoiceTask(task);
    return;
  }
  // The microphone is always open, so answering every stray sentence would
  // make the robot talk over the room. Only prompt when something in the
  // utterance was clearly aimed at it.
  if (mentionsObject || mentionsAction) speak(t("speak.notUnderstood"));
  else restoreIdleOrb();
}

/* ----------------------------------------------------------- audio input --- */

function sampleRate() {
  return audioContext ? audioContext.sampleRate : CAPTURE_SAMPLE_RATE;
}

function frameMilliseconds() {
  return (FRAME_SAMPLES / sampleRate()) * 1000;
}

function framesToSeconds(count) {
  return (count * FRAME_SAMPLES) / sampleRate();
}

function resetUtterance() {
  collecting = false;
  utteranceFrames = [];
  preRollFrames = [];
  loudFrames = 0;
  quietFrames = 0;
}

function encodeWav(frames) {
  const total = frames.reduce((sum, frame) => sum + frame.length, 0);
  const rate = sampleRate();
  const buffer = new ArrayBuffer(44 + total * 2);
  const view = new DataView(buffer);
  const ascii = (offset, text) => {
    for (let index = 0; index < text.length; index += 1) view.setUint8(offset + index, text.charCodeAt(index));
  };

  ascii(0, "RIFF");
  view.setUint32(4, 36 + total * 2, true);
  ascii(8, "WAVE");
  ascii(12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 1, true);
  view.setUint32(24, rate, true);
  view.setUint32(28, rate * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  ascii(36, "data");
  view.setUint32(40, total * 2, true);

  let offset = 44;
  for (const frame of frames) {
    for (let index = 0; index < frame.length; index += 1, offset += 2) {
      const sample = Math.max(-1, Math.min(1, frame[index]));
      view.setInt16(offset, sample < 0 ? sample * 0x8000 : sample * 0x7fff, true);
    }
  }
  return new Blob([buffer], { type: "audio/wav" });
}

// Speech recognition runs next to the microphone, not on the robot host, so
// the audio never leaves this machine; only the resulting task command is sent
// to the GUI server.
async function transcribeAudio(blob) {
  const query = langMode === "auto" ? "" : `?lang=${encodeURIComponent(langMode)}`;
  const response = await fetch(`${sttEndpoint}/transcribe${query}`, {
    method: "POST",
    cache: "no-store",
    headers: { "X-RBY1-Control": "ai-v3", "Content-Type": "audio/wav" },
    body: blob,
  });
  let payload = {};
  try { payload = await response.json(); } catch (_) { /* ignore malformed error bodies */ }
  if (!response.ok) throw new Error(payload.message || `HTTP ${response.status}`);
  return payload;
}

async function refreshSttHealth() {
  try {
    const response = await fetch(`${sttEndpoint}/health`, { cache: "no-store" });
    sttHealth = response.ok ? await response.json() : { state: "error", detail: `HTTP ${response.status}` };
  } catch (error) {
    sttHealth = { state: "error", detail: "unreachable" };
  }
  renderSttStatus();
}

async function submitUtterance(frames) {
  if (framesToSeconds(frames.length) < vadSettings.min_seconds) {
    restoreIdleOrb();
    return;
  }
  transcribing = true;
  setOrb("processing", "orb.transcribing.title", "orb.transcribing.hint");
  try {
    const result = await transcribeAudio(encodeWav(frames));
    transcribing = false;
    if (result && result.ok && result.text) handleTranscript(result.text, result.language);
    else restoreIdleOrb();
  } catch (error) {
    transcribing = false;
    setMicrophone("mic.error", { error: error.message }, false);
    restoreIdleOrb();
  }
}

function handleAudioFrame(rms, samples) {
  if (!voiceEnabled) return;
  if (speaking || commandInFlight || transcribing || Date.now() < micGateUntil) {
    resetUtterance();
    return;
  }

  if (!collecting) {
    // Track the room's noise floor only while nobody is speaking, so the
    // threshold adapts to the lab without drifting up during an utterance.
    noiseFloor = noiseFloor * 0.995 + rms * 0.005;
  }
  const threshold = Math.max(noiseFloor * vadSettings.noise_multiplier, vadSettings.min_rms);
  const loud = rms >= threshold;
  const frameMs = frameMilliseconds();
  const preRollLimit = Math.max(1, Math.round(vadSettings.preroll_ms / frameMs));
  const startLimit = Math.max(1, Math.round(vadSettings.start_ms / frameMs));
  const endLimit = Math.max(1, Math.round(vadSettings.end_ms / frameMs));

  if (!collecting) {
    preRollFrames.push(samples);
    if (preRollFrames.length > preRollLimit) preRollFrames.shift();
    loudFrames = loud ? loudFrames + 1 : 0;
    if (loudFrames >= startLimit) {
      // The pre-roll carries the onset that triggered detection, so short
      // words like "멈춰" are not clipped at the front.
      collecting = true;
      utteranceFrames = preRollFrames.slice();
      preRollFrames = [];
      quietFrames = 0;
      if (!clientIsActive()) setOrb("listening", "orb.speech.title", "orb.speech.hint");
      setMicrophone("mic.speech", { mode: languageModeLabel() }, true);
    }
    return;
  }

  utteranceFrames.push(samples);
  quietFrames = loud ? 0 : quietFrames + 1;
  if (quietFrames >= endLimit || framesToSeconds(utteranceFrames.length) >= vadSettings.max_seconds) {
    // Trim most of the closing silence. Whisper gains nothing from it, and
    // leaving it in would make even a stray cough long enough to clear the
    // min_seconds guard below.
    const tailKeep = Math.max(2, Math.round(120 / frameMs));
    const end = Math.max(1, utteranceFrames.length - Math.max(0, quietFrames - tailKeep));
    const frames = utteranceFrames.slice(0, end);
    resetUtterance();
    setMicrophone("mic.listening", { mode: languageModeLabel() }, true);
    submitUtterance(frames);
  }
}

async function startCapture() {
  if (captureReady) {
    if (audioContext.state === "suspended") await audioContext.resume();
    return true;
  }
  if (!navigator.mediaDevices?.getUserMedia || !window.AudioWorkletNode) {
    setMicrophone("mic.unsupported");
    setOrb("error", "orb.error.unsupported.title", "orb.error.unsupported.hint");
    voiceOrb.disabled = true;
    return false;
  }
  if (!window.isSecureContext) {
    // getUserMedia is blocked on plain http:// origins other than localhost.
    setMicrophone("mic.insecure");
    setOrb("error", "orb.error.insecure.title", "orb.error.insecure.hint");
    return false;
  }

  try {
    mediaStream = await navigator.mediaDevices.getUserMedia({
      audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true, autoGainControl: true },
    });
  } catch (error) {
    if (["NotAllowedError", "SecurityError"].includes(error.name)) {
      setMicrophone("mic.denied");
      setOrb("error", "orb.error.permission.title", "orb.error.permission.hint");
    } else {
      setMicrophone("mic.error", { error: error.name || error.message });
      setOrbText("error", "orb.error.mic", error.message);
    }
    return false;
  }

  try {
    try {
      audioContext = new AudioContext({ sampleRate: CAPTURE_SAMPLE_RATE });
    } catch (_) {
      // Some browsers refuse a forced rate; the server resamples whatever we send.
      audioContext = new AudioContext();
    }
    await audioContext.audioWorklet.addModule("/vad-worklet.js");
    sourceNode = audioContext.createMediaStreamSource(mediaStream);
    workletNode = new AudioWorkletNode(audioContext, "capture-processor");
    workletNode.port.onmessage = (event) => handleAudioFrame(event.data.rms, event.data.samples);
    sourceNode.connect(workletNode);
    // A silent sink keeps the graph running without echoing the mic to the speakers.
    const sink = audioContext.createGain();
    sink.gain.value = 0;
    workletNode.connect(sink);
    sink.connect(audioContext.destination);
    if (audioContext.state === "suspended") await audioContext.resume();
  } catch (error) {
    stopCapture();
    setMicrophone("mic.error", { error: error.message });
    setOrbText("error", "orb.error.mic", error.message);
    return false;
  }

  captureReady = true;
  return true;
}

function stopCapture() {
  if (workletNode) {
    workletNode.port.onmessage = null;
    workletNode.disconnect();
    workletNode = null;
  }
  if (sourceNode) {
    sourceNode.disconnect();
    sourceNode = null;
  }
  if (mediaStream) {
    mediaStream.getTracks().forEach((track) => track.stop());
    mediaStream = null;
  }
  if (audioContext) {
    audioContext.close().catch(() => {});
    audioContext = null;
  }
  captureReady = false;
  resetUtterance();
}

/* --------------------------------------------------------------- wiring --- */

voiceOrb.addEventListener("click", async () => {
  if (voiceEnabled) {
    voiceEnabled = false;
    window.speechSynthesis?.cancel();
    speaking = false;
    stopCapture();
    setMicrophone("mic.off");
    setOrb("idle", "orb.off.title", "orb.off.hint");
    return;
  }

  setMicrophone("mic.starting", {}, true);
  setOrb("processing", "orb.starting.title", "orb.starting.hint");
  const started = await startCapture();
  if (!started) return;

  voiceEnabled = true;
  noiseFloor = 0.004;
  micGateUntil = 0;
  resetUtterance();
  setMicrophone("mic.listening", { mode: languageModeLabel() }, true);
  restoreIdleOrb();
});

langChip.addEventListener("click", () => {
  const order = ["auto", ...availableLanguages];
  langMode = order[(order.indexOf(langMode) + 1) % order.length];
  if (langMode !== "auto") setLanguage(langMode);
  renderLanguageChip();
  if (voiceEnabled) setMicrophone("mic.listening", { mode: languageModeLabel() }, true);
});

startButton.addEventListener("click", () => {
  const task = selectedTask();
  const taskText = task ? `${task.number}. ${task.label}\n${task.instruction}\n\n` : "";
  if (window.confirm(t("ui.confirm", { task: taskText }))) startSelectedPolicy().catch(() => {});
});
stopButton.addEventListener("click", () => stopPolicyClient().catch(() => {}));
reconnectButton.addEventListener("click", () => post("/api/cameras/restart").catch(() => {}));

if ("speechSynthesis" in window) {
  // Chrome populates the voice list asynchronously.
  window.speechSynthesis.addEventListener("voiceschanged", () => voiceForLanguage(currentLang));
}

document.documentElement.lang = currentLang;
applyStaticTranslations();
renderLanguageChip();
renderOrb();
renderMicrophone();
renderSttStatus();
refreshStatus().then(refreshSttHealth);
refreshLogs();
setInterval(refreshStatus, 700);
setInterval(refreshLogs, 700);
setInterval(refreshSttHealth, 3000);
