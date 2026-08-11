"use strict";

// ── 공통 ────────────────────────────────────────────────────────────
const $ = (sel, root = document) => root.querySelector(sel);
const esc = (value) =>
  String(value ?? "").replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c])
  );

let toastTimer = null;
function toast(message, ms = 3500) {
  const el = $("#toast");
  el.textContent = message;
  el.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => (el.hidden = true), ms);
}

async function api(url, options = {}) {
  const response = await fetch(url, options);
  const isJson = (response.headers.get("content-type") || "").includes("json");
  const body = isJson ? await response.json() : await response.text();
  if (!response.ok) throw new Error((body && body.error) || response.statusText);
  return body;
}

const jsonPost = (url, payload, method = "POST") =>
  api(url, {
    method,
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });

function hhmmss(seconds) {
  const total = Math.max(0, Math.floor(seconds || 0));
  const pad = (n) => String(n).padStart(2, "0");
  return `${pad(Math.floor(total / 3600))}:${pad(Math.floor((total % 3600) / 60))}:${pad(total % 60)}`;
}

async function showHealth() {
  const el = $("#health");
  if (!el) return;
  try {
    const info = await api("/api/health");
    const problems = [];
    if (!info.hf_token) problems.push("HF_TOKEN 없음");
    if (!info.ffmpeg) problems.push("ffmpeg 없음");

    const where =
      info.asr_device === "cuda"
        ? `GPU: ${info.gpu || "cuda"}`
        : `CPU 모드 (${info.compute_type})` +
          (info.resolved_device === "mps" ? " · 화자분리 MPS" : "");
    el.textContent = problems.length
      ? `⚠ ${problems.join(" · ")} · ${where}`
      : `${where} · ${info.whisper_model}`;
    el.classList.toggle("bad", problems.length > 0);
  } catch {
    el.textContent = "";
  }
}

// ── 업로드 페이지 ────────────────────────────────────────────────────
function initIndex() {
  const dropzone = $("#dropzone");
  const fileInput = $("#file");
  const filename = $("#filename");

  const setFile = (file) => {
    if (!file) return;
    const transfer = new DataTransfer();
    transfer.items.add(file);
    fileInput.files = transfer.files;
    filename.textContent = `${file.name} (${(file.size / 1048576).toFixed(1)} MB)`;
  };

  $("#pick").addEventListener("click", () => fileInput.click());
  dropzone.addEventListener("click", (event) => {
    if (event.target === dropzone || event.target.tagName === "P") fileInput.click();
  });
  fileInput.addEventListener("change", () => setFile(fileInput.files[0]));

  ["dragenter", "dragover"].forEach((type) =>
    dropzone.addEventListener(type, (event) => {
      event.preventDefault();
      dropzone.classList.add("over");
    })
  );
  ["dragleave", "drop"].forEach((type) =>
    dropzone.addEventListener(type, (event) => {
      event.preventDefault();
      dropzone.classList.remove("over");
    })
  );
  dropzone.addEventListener("drop", (event) => setFile(event.dataTransfer.files[0]));

  $("#upload-form").addEventListener("submit", async (event) => {
    event.preventDefault();
    if (!fileInput.files.length) return toast("파일을 선택하세요.");

    const button = $("#submit");
    button.disabled = true;
    button.textContent = "업로드 중…";

    const form = new FormData();
    form.append("file", fileInput.files[0]);
    form.append("name", $("#name").value.trim());
    form.append("language", $("#language").value);
    form.append("initial_prompt", $("#initial_prompt").value);
    form.append("use_prompt", $("#use_prompt").checked ? "1" : "0");
    form.append("min_speakers", $("#min_speakers").value);
    form.append("max_speakers", $("#max_speakers").value);

    try {
      const result = await api("/api/jobs", { method: "POST", body: form });
      toast(`작업 등록됨: ${result.name}`);
      fileInput.value = "";
      filename.textContent = "";
      refreshJobs();
    } catch (error) {
      toast(`업로드 실패: ${error.message}`, 8000);
    } finally {
      button.disabled = false;
      button.textContent = "전사 시작";
    }
  });

  refreshJobs();
  refreshResults();
  setInterval(refreshJobs, 1500);
}

let lastActiveCount = 0;

// 재시도 시 건너뛸 수 있는 단계 이름
const STAGE_NAMES = {
  audio: "오디오 변환",
  transcribe: "음성 인식",
  align: "단어 정렬",
  diarize: "화자 분리",
};

async function refreshJobs() {
  const container = $("#jobs");
  if (!container) return;
  let data;
  try {
    data = await api("/api/jobs?limit=12");
  } catch {
    return;
  }

  const jobs = data.jobs || [];
  if (!jobs.length) {
    container.innerHTML = '<p class="muted">작업 없음</p>';
    return;
  }

  container.innerHTML = jobs
    .map((job) => {
      const cls = job.status === "done" ? "done" : job.status === "error" ? "error" : "";
      const label =
        { queued: "대기", running: "진행 중", done: "완료", error: "오류" }[job.status] ||
        job.status;
      const link =
        job.status === "done"
          ? `<a href="/result/${encodeURIComponent(job.name)}">결과 보기</a>`
          : "";
      const error = job.error
        ? `<div class="job-error">${esc(job.error)}</div>`
        : "";

      let retry = "";
      if (job.status === "error") {
        const kept = (job.cached_stages || []).map((s) => STAGE_NAMES[s] || s);
        const hint = kept.length
          ? `${kept.join(" · ")} 재사용`
          : "처음부터";
        retry = `<button data-retry="${job.id}" title="${esc(hint)}">이어서 재시도</button>
                 <span class="job-stage">${esc(hint)}</span>`;
      }

      return `<div class="job">
        <span class="job-name">${esc(job.name)}</span>
        <span class="status ${cls}">${label}</span>
        <span class="job-stage">${esc(job.stage || "")}</span>
        <span class="bar"><i style="width:${Math.round(job.progress || 0)}%"></i></span>
        ${link}${retry}${error}
      </div>`;
    })
    .join("");

  container.querySelectorAll("[data-retry]").forEach((button) =>
    button.addEventListener("click", async () => {
      button.disabled = true;
      try {
        const result = await api(`/api/jobs/${button.dataset.retry}/retry`, {
          method: "POST",
        });
        const kept = (result.resumed_from || []).map((s) => STAGE_NAMES[s] || s);
        toast(
          kept.length
            ? `재시도 시작. ${kept.join(" · ")} 단계는 다시 계산하지 않습니다.`
            : "재시도 시작."
        );
        refreshJobs();
      } catch (error) {
        toast(`재시도 실패: ${error.message}`, 8000);
        button.disabled = false;
      }
    })
  );

  const active = jobs.filter((job) => job.status === "queued" || job.status === "running").length;
  if (active === 0 && lastActiveCount > 0) refreshResults();
  lastActiveCount = active;
}

async function refreshResults() {
  const container = $("#results");
  if (!container) return;
  let results;
  try {
    ({ results } = await api("/api/results"));
  } catch (error) {
    container.innerHTML = `<p class="muted">목록을 불러오지 못했습니다: ${esc(error.message)}</p>`;
    return;
  }
  if (!results.length) {
    container.innerHTML = '<p class="muted">아직 결과가 없습니다.</p>';
    return;
  }
  container.innerHTML = results
    .map(
      (item) => `<div class="result-row">
        <a href="/result/${encodeURIComponent(item.name)}">${esc(item.name)}</a>
        <span class="chips">${item.speakers
          .map((name) => `<span class="chip">${esc(name)}</span>`)
          .join("")}</span>
        <span class="result-meta">${esc(item.created_at.replace("T", " "))} · ${hhmmss(
        item.duration
      )} · ${esc(item.language)}</span>
        <button class="danger" data-del="${esc(item.name)}">삭제</button>
      </div>`
    )
    .join("");

  container.querySelectorAll("[data-del]").forEach((button) =>
    button.addEventListener("click", async () => {
      const name = button.dataset.del;
      if (!confirm(`'${name}' 결과를 삭제할까요? (txt·json 파일이 지워집니다)`)) return;
      await api(`/api/results/${encodeURIComponent(name)}`, { method: "DELETE" });
      toast("삭제됨");
      refreshResults();
    })
  );
}

// ── 결과 페이지 ─────────────────────────────────────────────────────
let resultData = null;

async function initResult() {
  const name = document.body.dataset.name;
  resultData = await api(`/api/results/${encodeURIComponent(name)}`);

  const detection = resultData.language_detection || {};
  const lang = detection.auto
    ? `언어 ${resultData.language} (자동 감지, 확신도 ${detection.confidence ?? "?"})`
    : `언어 ${resultData.language} (고정)`;
  $("#result-meta").textContent =
    `${(resultData.created_at || "").replace("T", " ")} · ${hhmmss(resultData.duration)} · ` +
    `${lang} · 원본 ${resultData.source_file}`;

  const warnings = $("#warnings");
  warnings.innerHTML = (resultData.warnings || [])
    .map((text) => `<div class="notice warn">${esc(text)}</div>`)
    .join("");

  $("#show-time").addEventListener("change", renderTranscript);
  $("#save-speakers").addEventListener("click", saveSpeakers);

  renderSpeakerPanel();
  renderTranscript();
}

function renderSpeakerPanel() {
  const names = (resultData.registered || []).map((person) => person.name);
  const datalist = `<datalist id="speaker-names">${names
    .map((n) => `<option value="${esc(n)}"></option>`)
    .join("")}</datalist>`;

  const rows = Object.entries(resultData.speakers || {})
    .map(([label, info]) => {
      const known = info.speaker_id ? info.display : "";
      let badge = `<span class="badge">미지정</span>`;
      if (info.manual) badge = `<span class="badge auto">직접 지정</span>`;
      else if (info.matched)
        badge = `<span class="badge auto">자동 인식 · 유사도 ${(info.score || 0).toFixed(2)}</span>`;
      else if (info.reason) badge = `<span class="badge" title="${esc(info.reason)}">${esc(info.reason)}</span>`;

      return `<div class="sp-row" data-label="${esc(label)}">
        <span class="sp-label">${esc(info.display)}</span>
        <span class="sp-stat">발화 ${hhmmss(info.total_speech)} · ${esc(label)}</span>
        ${badge}
        <input type="text" list="speaker-names" value="${esc(known)}"
               placeholder="이름 입력 (비우면 지정 해제)">
      </div>`;
    })
    .join("");

  $("#speaker-panel").innerHTML = datalist + rows;
}

function renderTranscript() {
  const showTime = $("#show-time").checked;
  $("#transcript").innerHTML = (resultData.lines || [])
    .map(
      (line) => `<div class="line">
        <button class="who" data-jump="${esc(line.speaker || "")}">${esc(line.name)}</button>
        ${showTime ? `<span class="time">${hhmmss(line.start)}</span>` : ""}
        <span class="said">${esc(line.text)}</span>
      </div>`
    )
    .join("");

  document.querySelectorAll("[data-jump]").forEach((button) =>
    button.addEventListener("click", () => {
      const row = $(`.sp-row[data-label="${CSS.escape(button.dataset.jump)}"]`);
      if (!row) return;
      row.scrollIntoView({ behavior: "smooth", block: "center" });
      row.classList.add("flash");
      const input = $("input", row);
      if (input) input.focus();
      setTimeout(() => row.classList.remove("flash"), 1600);
    })
  );
}

async function saveSpeakers() {
  const button = $("#save-speakers");
  button.disabled = true;
  button.textContent = "저장 중…";

  const assignments = [...document.querySelectorAll(".sp-row")].map((row) => ({
    label: row.dataset.label,
    name: $("input", row).value.trim(),
  }));

  try {
    const response = await jsonPost(
      `/api/results/${encodeURIComponent(resultData.name)}/speakers`,
      { assignments }
    );
    resultData.speakers = response.speakers;
    resultData.lines = response.lines;
    resultData.registered = (await api("/api/speakers")).speakers;
    renderSpeakerPanel();
    renderTranscript();
    toast(
      response.enrolled.length
        ? `저장 완료. 목소리 등록: ${response.enrolled.join(", ")} — 다음 파일부터 자동 인식됩니다.`
        : "저장 완료."
    );
  } catch (error) {
    toast(`저장 실패: ${error.message}`, 8000);
  } finally {
    button.disabled = false;
    button.textContent = "화자 저장 · 텍스트 다시 만들기";
  }
}

// ── 화자 관리 페이지 ─────────────────────────────────────────────────
function initSpeakers() {
  $("#new-speaker").addEventListener("submit", async (event) => {
    event.preventDefault();
    const input = $("#new-name");
    const name = input.value.trim();
    if (!name) return;
    try {
      await jsonPost("/api/speakers", { name });
      input.value = "";
      toast(`'${name}' 추가됨`);
      renderSpeakers();
    } catch (error) {
      toast(error.message, 6000);
    }
  });
  renderSpeakers();
}

async function renderSpeakers() {
  const container = $("#speaker-list");
  const { speakers } = await api("/api/speakers");
  if (!speakers.length) {
    container.innerHTML = '<p class="muted">등록된 화자가 없습니다.</p>';
    return;
  }

  container.innerHTML = speakers
    .map(
      (person) => `<div class="sp-card" data-id="${person.id}">
        <div class="sp-head">
          <strong>${esc(person.name)}</strong>
          <span class="badge">보이스프린트 ${person.voiceprint_count}개 · 학습 음성 ${hhmmss(
        person.total_speech
      )}</span>
          <span class="spacer"></span>
          <label class="toggle"><input type="file" accept="audio/*,video/*" data-sample="${person.id}" hidden>
            <button type="button" data-pick="${person.id}">샘플 음성 추가</button></label>
          <button type="button" data-rename="${person.id}">이름 변경</button>
          <button type="button" class="danger" data-remove="${person.id}">삭제</button>
        </div>
        ${
          person.voiceprints.length
            ? `<ul class="vp-list">${person.voiceprints
                .map(
                  (vp) =>
                    `<li>${esc(vp.source || "출처 없음")} · ${hhmmss(vp.speech_sec)} · ${esc(
                      (vp.created_at || "").replace("T", " ")
                    )}</li>`
                )
                .join("")}</ul>`
            : '<p class="muted">보이스프린트 없음 — 샘플을 올리거나 전사 결과에서 이름을 지정하세요.</p>'
        }
      </div>`
    )
    .join("");

  container.querySelectorAll("[data-pick]").forEach((button) =>
    button.addEventListener("click", () =>
      $(`[data-sample="${button.dataset.pick}"]`).click()
    )
  );

  container.querySelectorAll("[data-sample]").forEach((input) =>
    input.addEventListener("change", async () => {
      if (!input.files.length) return;
      const form = new FormData();
      form.append("file", input.files[0]);
      toast("샘플 분석 중… (모델 로딩에 시간이 걸릴 수 있습니다)", 15000);
      try {
        const result = await api(`/api/speakers/${input.dataset.sample}/samples`, {
          method: "POST",
          body: form,
        });
        toast(`등록 완료 · 학습 음성 ${result.speech_sec.toFixed(1)}초`);
        renderSpeakers();
      } catch (error) {
        toast(`등록 실패: ${error.message}`, 10000);
      } finally {
        input.value = "";
      }
    })
  );

  container.querySelectorAll("[data-rename]").forEach((button) =>
    button.addEventListener("click", async () => {
      const card = button.closest(".sp-card");
      const current = $("strong", card).textContent;
      const next = prompt("새 이름", current);
      if (!next || next.trim() === current) return;
      const updateResults = confirm("기존 결과 파일의 이름도 함께 바꿀까요?");
      try {
        const result = await jsonPost(
          `/api/speakers/${button.dataset.rename}`,
          { name: next.trim(), update_results: updateResults },
          "PATCH"
        );
        toast(`변경됨. 갱신된 결과 ${result.updated_results}건`);
        renderSpeakers();
      } catch (error) {
        toast(error.message, 6000);
      }
    })
  );

  container.querySelectorAll("[data-remove]").forEach((button) =>
    button.addEventListener("click", async () => {
      const name = $("strong", button.closest(".sp-card")).textContent;
      if (!confirm(`'${name}' 화자와 등록된 목소리를 모두 삭제할까요?`)) return;
      await api(`/api/speakers/${button.dataset.remove}`, { method: "DELETE" });
      toast("삭제됨");
      renderSpeakers();
    })
  );
}

// ── 부팅 ────────────────────────────────────────────────────────────
document.addEventListener("DOMContentLoaded", () => {
  showHealth();
  const page = document.body.dataset.page;
  const boot = { index: initIndex, result: initResult, speakers: initSpeakers }[page];
  if (boot) {
    Promise.resolve()
      .then(boot)
      .catch((error) => toast(`오류: ${error.message}`, 8000));
  }
});
