"use strict";

// ── Shared ──────────────────────────────────────────────────────────
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
    if (!info.hf_token) problems.push("HF_TOKEN missing");
    if (!info.ffmpeg) problems.push("ffmpeg missing");

    const where =
      info.asr_device === "cuda"
        ? `GPU: ${info.gpu || "cuda"}`
        : `CPU mode (${info.compute_type})` +
          (info.resolved_device === "mps" ? " · diarization on MPS" : "");
    el.textContent = problems.length
      ? `⚠ ${problems.join(" · ")} · ${where}`
      : `${where} · ${info.whisper_model}`;
    el.classList.toggle("bad", problems.length > 0);
  } catch {
    el.textContent = "";
  }
}

// ── Upload page ─────────────────────────────────────────────────────
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
    if (!fileInput.files.length) return toast("Please choose a file.");

    const button = $("#submit");
    button.disabled = true;
    button.textContent = "Uploading…";

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
      toast(`Job queued: ${result.name}`);
      fileInput.value = "";
      filename.textContent = "";
      refreshJobs();
    } catch (error) {
      toast(`Upload failed: ${error.message}`, 8000);
    } finally {
      button.disabled = false;
      button.textContent = "Start transcription";
    }
  });

  refreshJobs();
  refreshResults();
  setInterval(refreshJobs, 1500);
}

let lastActiveCount = 0;

// Stage names that a retry can skip
const STAGE_NAMES = {
  audio: "audio conversion",
  transcribe: "transcription",
  align: "word alignment",
  diarize: "diarization",
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
    container.innerHTML = '<p class="muted">No jobs</p>';
    return;
  }

  container.innerHTML = jobs
    .map((job) => {
      const cls = job.status === "done" ? "done" : job.status === "error" ? "error" : "";
      const label =
        { queued: "queued", running: "running", done: "done", error: "error" }[job.status] ||
        job.status;
      const link =
        job.status === "done"
          ? `<a href="/result/${encodeURIComponent(job.name)}">View result</a>`
          : "";
      const error = job.error
        ? `<div class="job-error">${esc(job.error)}</div>`
        : "";

      let retry = "";
      if (job.status === "error") {
        const kept = (job.cached_stages || []).map((s) => STAGE_NAMES[s] || s);
        const hint = kept.length
          ? `reuses ${kept.join(" · ")}`
          : "starts over";
        retry = `<button data-retry="${job.id}" title="${esc(hint)}">Resume</button>
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
            ? `Retry started. These stages will not be recomputed: ${kept.join(" · ")}.`
            : "Retry started."
        );
        refreshJobs();
      } catch (error) {
        toast(`Retry failed: ${error.message}`, 8000);
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
    container.innerHTML = `<p class="muted">Could not load the list: ${esc(error.message)}</p>`;
    return;
  }
  if (!results.length) {
    container.innerHTML = '<p class="muted">Nothing here yet.</p>';
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
        <button class="danger" data-del="${esc(item.name)}">Delete</button>
      </div>`
    )
    .join("");

  container.querySelectorAll("[data-del]").forEach((button) =>
    button.addEventListener("click", async () => {
      const name = button.dataset.del;
      if (!confirm(`Delete the result '${name}'? Its txt and json files will be removed.`)) return;
      await api(`/api/results/${encodeURIComponent(name)}`, { method: "DELETE" });
      toast("Deleted");
      refreshResults();
    })
  );
}

// ── Result page ─────────────────────────────────────────────────────
let resultData = null;

async function initResult() {
  const name = document.body.dataset.name;
  resultData = await api(`/api/results/${encodeURIComponent(name)}`);

  const detection = resultData.language_detection || {};
  const lang = detection.auto
    ? `Language ${resultData.language} (auto-detected, confidence ${detection.confidence ?? "?"})`
    : `Language ${resultData.language} (pinned)`;
  // Even with silence cut out or the file processed in chunks, timestamps stay
  // on the original clock, so playback positions are unaffected
  const trim = resultData.trim || {};
  const notes = [];
  if (trim.enabled) notes.push(`transcribed with ${hhmmss(trim.removed)} of silence removed`);
  if ((resultData.chunks || []).length > 1)
    notes.push(`processed in ${resultData.chunks.length} chunks`);
  if ((resultData.dropped || []).length)
    notes.push(`${resultData.dropped.length} hallucination(s) removed`);
  $("#result-meta").textContent =
    `${(resultData.created_at || "").replace("T", " ")} · ${hhmmss(resultData.duration)} · ` +
    `${lang} · source ${resultData.source_file}` +
    (notes.length ? ` · ${notes.join(" · ")}` : "");

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
      let badge = `<span class="badge">unassigned</span>`;
      if (info.manual) badge = `<span class="badge auto">set manually</span>`;
      else if (info.matched)
        badge = `<span class="badge auto">auto-recognized · similarity ${(info.score || 0).toFixed(2)}</span>`;
      else if (info.reason) badge = `<span class="badge" title="${esc(info.reason)}">${esc(info.reason)}</span>`;

      return `<div class="sp-row" data-label="${esc(label)}">
        <span class="sp-label">${esc(info.display)}</span>
        <span class="sp-stat">${hhmmss(info.total_speech)} spoken · ${esc(label)}</span>
        ${badge}
        <input type="text" list="speaker-names" value="${esc(known)}"
               placeholder="Enter a name (blank to unassign)">
      </div>`;
    })
    .join("");

  $("#speaker-panel").innerHTML = datalist + rows;
}

// Clicking a transcript line plays just that span. The server cuts the clip, so
// even a 10-hour recording starts instantly (the whole wav would be 1.1GB).
let player = null;
let playingLine = null;

function stopPlayback() {
  if (player) {
    player.pause();
    player.removeAttribute("src");
    player.load();
  }
  if (playingLine) {
    playingLine.classList.remove("playing");
    playingLine = null;
  }
}

function playLine(row) {
  if (playingLine === row) return stopPlayback(); // clicking the same line stops it
  stopPlayback();

  if (!player) {
    player = new Audio();
    player.addEventListener("ended", stopPlayback);
    player.addEventListener("error", () => {
      if (playingLine) playingLine.classList.add("failed");
      stopPlayback();
    });
  }
  const name = encodeURIComponent(document.body.dataset.name);
  const { start, end } = row.dataset;
  player.src = `/api/results/${name}/clip?start=${start}&end=${end}`;
  row.classList.remove("failed");
  row.classList.add("playing");
  playingLine = row;
  player.play().catch(() => stopPlayback());
}

function renderTranscript() {
  stopPlayback();
  const showTime = $("#show-time").checked;
  $("#transcript").innerHTML = (resultData.lines || [])
    .map(
      (line) => `<div class="line" data-start="${line.start}" data-end="${line.end}">
        <button class="who" data-jump="${esc(line.speaker || "")}">${esc(line.name)}</button>
        ${showTime ? `<span class="time">${hhmmss(line.start)}</span>` : ""}
        <button class="said" title="Click to hear just this part">${esc(line.text)}</button>
      </div>`
    )
    .join("");

  document.querySelectorAll(".transcript .said").forEach((button) =>
    button.addEventListener("click", () => playLine(button.closest(".line")))
  );

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
  button.textContent = "Saving…";

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
        ? `Saved. Enrolled: ${response.enrolled.join(", ")} — they will be recognized automatically from now on.`
        : "Saved."
    );
  } catch (error) {
    toast(`Save failed: ${error.message}`, 8000);
  } finally {
    button.disabled = false;
    button.textContent = "Save speakers · rebuild text";
  }
}

// ── Speakers page ───────────────────────────────────────────────────
function initSpeakers() {
  $("#new-speaker").addEventListener("submit", async (event) => {
    event.preventDefault();
    const input = $("#new-name");
    const name = input.value.trim();
    if (!name) return;
    try {
      await jsonPost("/api/speakers", { name });
      input.value = "";
      toast(`Added '${name}'`);
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
    container.innerHTML = '<p class="muted">No speakers enrolled yet.</p>';
    return;
  }

  container.innerHTML = speakers
    .map(
      (person) => `<div class="sp-card" data-id="${person.id}">
        <div class="sp-head">
          <strong>${esc(person.name)}</strong>
          <span class="badge">${person.voiceprint_count} voiceprint(s) · ${hhmmss(
        person.total_speech
      )}</span>
          <span class="spacer"></span>
          <label class="toggle"><input type="file" accept="audio/*,video/*" data-sample="${person.id}" hidden>
            <button type="button" data-pick="${person.id}">Add voice sample</button></label>
          <button type="button" data-rename="${person.id}">Rename</button>
          <button type="button" class="danger" data-remove="${person.id}">Delete</button>
        </div>
        ${
          person.voiceprints.length
            ? `<ul class="vp-list">${person.voiceprints
                .map(
                  (vp) =>
                    `<li>${esc(vp.source || "no source")} · ${hhmmss(vp.speech_sec)} · ${esc(
                      (vp.created_at || "").replace("T", " ")
                    )}</li>`
                )
                .join("")}</ul>`
            : '<p class="muted">No voiceprints yet — upload a sample or assign a name on a result page.</p>'
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
      toast("Analyzing the sample… (loading the model can take a while)", 15000);
      try {
        const result = await api(`/api/speakers/${input.dataset.sample}/samples`, {
          method: "POST",
          body: form,
        });
        toast(`Enrolled · ${result.speech_sec.toFixed(1)}s of voice`);
        renderSpeakers();
      } catch (error) {
        toast(`Enrollment failed: ${error.message}`, 10000);
      } finally {
        input.value = "";
      }
    })
  );

  container.querySelectorAll("[data-rename]").forEach((button) =>
    button.addEventListener("click", async () => {
      const card = button.closest(".sp-card");
      const current = $("strong", card).textContent;
      const next = prompt("New name", current);
      if (!next || next.trim() === current) return;
      const updateResults = confirm("Also update the name in existing result files?");
      try {
        const result = await jsonPost(
          `/api/speakers/${button.dataset.rename}`,
          { name: next.trim(), update_results: updateResults },
          "PATCH"
        );
        toast(`Renamed. ${result.updated_results} result(s) updated`);
        renderSpeakers();
      } catch (error) {
        toast(error.message, 6000);
      }
    })
  );

  container.querySelectorAll("[data-remove]").forEach((button) =>
    button.addEventListener("click", async () => {
      const name = $("strong", button.closest(".sp-card")).textContent;
      if (!confirm(`Delete the speaker '${name}' and every enrolled voiceprint?`)) return;
      await api(`/api/speakers/${button.dataset.remove}`, { method: "DELETE" });
      toast("Deleted");
      renderSpeakers();
    })
  );
}

// ── Boot ────────────────────────────────────────────────────────────
document.addEventListener("DOMContentLoaded", () => {
  showHealth();
  const page = document.body.dataset.page;
  const boot = { index: initIndex, result: initResult, speakers: initSpeakers }[page];
  if (boot) {
    Promise.resolve()
      .then(boot)
      .catch((error) => toast(`Error: ${error.message}`, 8000));
  }
});
