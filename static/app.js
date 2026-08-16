const API = "";

let currentSessionId = null;
let repsByDayChart = null;
let setsByExerciseChart = null;

// ---------- View switching ----------
document.querySelectorAll(".nav-link").forEach((btn) => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".nav-link").forEach((b) => b.classList.remove("active"));
    document.querySelectorAll(".view").forEach((v) => v.classList.remove("active"));
    btn.classList.add("active");
    document.getElementById(`view-${btn.dataset.view}`).classList.add("active");
    if (btn.dataset.view === "history") loadHistory();
    if (btn.dataset.view === "analytics") loadAnalytics();
  });
});

// ---------- Populate exercise dropdown from the model's known classes ----------
async function loadExercises() {
  try {
    const res = await fetch(`${API}/api/exercises`);
    const data = await res.json();
    const select = document.getElementById("intendedExercise");
    data.exercises.forEach((ex) => {
      const opt = document.createElement("option");
      opt.value = ex;
      opt.textContent = ex[0].toUpperCase() + ex.slice(1);
      select.appendChild(opt);
    });
  } catch (e) {
    console.error("Could not load exercises", e);
  }
}

// ---------- Session creation ----------
document.getElementById("newSessionBtn").addEventListener("click", async () => {
  const label = document.getElementById("sessionLabel").value || "Synthetic test batch";
  try {
    const res = await fetch(`${API}/api/sessions`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ label }),
    });
    const data = await res.json();
    currentSessionId = data.session_id;
    const hint = document.getElementById("sessionHint");
    hint.textContent = `Active session #${data.session_id}: "${data.label}"`;
    hint.classList.add("ok");
    refreshStats();
    updateSubmitState();
  } catch (e) {
    document.getElementById("formError").textContent = "Could not create session. Is the server running?";
  }
});

// ---------- Form state ----------
const exerciseSelect = document.getElementById("intendedExercise");
const numSetsInput = document.getElementById("numSets");
const targetRepsInput = document.getElementById("targetReps");
const genPreviewText = document.getElementById("genPreviewText");

function updatePreviewText() {
  const ex = exerciseSelect.value;
  const sets = numSetsInput.value;
  const reps = targetRepsInput.value;
  if (ex) {
    genPreviewText.textContent = `Will generate ${sets} set(s) of ${ex} at ${reps} reps each, then classify each one.`;
  } else {
    genPreviewText.textContent = "Pick an exercise, sets, and reps, then generate.";
  }
  updateSubmitState();
}
[exerciseSelect, numSetsInput, targetRepsInput].forEach((el) =>
  el.addEventListener("input", updatePreviewText)
);

function updateSubmitState() {
  document.getElementById("submitBtn").disabled = !(currentSessionId && exerciseSelect.value);
}

// ---------- Submit / generate + predict ----------
document.getElementById("logForm").addEventListener("submit", async (e) => {
  e.preventDefault();
  const errorEl = document.getElementById("formError");
  errorEl.textContent = "";

  if (!currentSessionId) {
    errorEl.textContent = "Create a session first.";
    return;
  }
  if (!exerciseSelect.value) {
    errorEl.textContent = "Choose an exercise.";
    return;
  }

  const submitBtn = document.getElementById("submitBtn");
  submitBtn.disabled = true;
  submitBtn.textContent = "Generating & analyzing…";

  const payload = {
    session_id: currentSessionId,
    exercise: exerciseSelect.value,
    num_sets: parseInt(numSetsInput.value || 1),
    reps_per_set: parseInt(targetRepsInput.value || 1),
  };

  try {
    const res = await fetch(`${API}/api/generate-and-log`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || "Generation failed.");
    }
    const data = await res.json();
    renderBatchResult(data.generated, payload.exercise);
    refreshStats();
  } catch (err) {
    errorEl.textContent = err.message;
  } finally {
    submitBtn.disabled = false;
    submitBtn.textContent = "Generate & run through model";
    updateSubmitState();
  }
});

function renderBatchResult(generated, intendedExercise) {
  document.getElementById("resultEmpty").hidden = true;
  const body = document.getElementById("resultBody");
  body.hidden = false;

  const avgReps = generated.reduce((a, s) => a + s.predicted_reps, 0) / generated.length;
  const avgConf = generated.reduce((a, s) => a + s.confidence, 0) / generated.length;
  const numCorrect = generated.filter((s) => s.match).length;

  document.getElementById("resultEyebrow").textContent = `${generated.length} set(s) generated`;
  document.getElementById("resultExercise").textContent = intendedExercise;
  document.getElementById("resultReps").textContent = avgReps.toFixed(1);
  document.getElementById("resultConfidence").textContent = `${Math.round(avgConf * 100)}%`;
  document.getElementById("resultEpochs").textContent = `${numCorrect} / ${generated.length}`;

  const fill = document.getElementById("confidenceBarFill");
  requestAnimationFrame(() => { fill.style.width = `${Math.round(avgConf * 100)}%`; });

  const badge = document.getElementById("matchBadge");
  badge.hidden = false;
  const allMatch = numCorrect === generated.length;
  badge.textContent = allMatch ? "All classified correctly" : `${generated.length - numCorrect} misclassified`;
  badge.className = "match-badge " + (allMatch ? "match-yes" : "match-no");

  const setBySet = document.getElementById("setBySet");
  setBySet.innerHTML = generated.map((s) => `
    <div class="set-row ${s.match ? "set-row-ok" : "set-row-bad"}">
      <span>Set ${s.set_number}</span>
      <span class="tag-exercise">${s.predicted_exercise}</span>
      <span>${s.predicted_reps} reps</span>
      <span>${Math.round(s.confidence * 100)}%</span>
    </div>
  `).join("");
}

// ---------- Stats strip ----------
async function refreshStats() {
  try {
    const res = await fetch(`${API}/api/analytics`);
    const data = await res.json();
    document.getElementById("statSets").textContent = data.total_sets;
    document.getElementById("statReps").textContent = data.total_reps;
    document.getElementById("statSessions").textContent = data.total_sessions;
    document.getElementById("statAccuracy").textContent =
      data.intended_vs_predicted_accuracy === null
        ? "—"
        : `${Math.round(data.intended_vs_predicted_accuracy * 100)}%`;
  } catch (e) {
    console.error(e);
  }
}

// ---------- History ----------
async function loadHistory() {
  const tbody = document.getElementById("historyBody");
  tbody.innerHTML = `<tr><td colspan="8" class="empty-row">Loading…</td></tr>`;
  try {
    const res = await fetch(`${API}/api/history`);
    const data = await res.json();
    if (!data.sets.length) {
      tbody.innerHTML = `<tr><td colspan="8" class="empty-row">No sets logged yet.</td></tr>`;
      return;
    }
    tbody.innerHTML = data.sets.map((s) => {
      let matchCell = '<span class="tag-match-na">—</span>';
      if (s.match === 1) matchCell = '<span class="tag-match-yes">Match</span>';
      if (s.match === 0) matchCell = '<span class="tag-match-no">Different</span>';
      const logged = new Date(s.created_at).toLocaleString();
      return `<tr>
        <td>${logged}</td>
        <td>${s.session_label || "—"}</td>
        <td>${s.set_number ?? "—"}</td>
        <td class="tag-exercise">${s.intended_exercise || "—"}</td>
        <td class="tag-exercise">${s.predicted_exercise}</td>
        <td>${s.predicted_reps}</td>
        <td>${Math.round(s.confidence * 100)}%</td>
        <td>${matchCell}</td>
      </tr>`;
    }).join("");
  } catch (e) {
    tbody.innerHTML = `<tr><td colspan="8" class="empty-row">Could not load history.</td></tr>`;
  }
}

// ---------- Analytics ----------
async function loadAnalytics() {
  try {
    const res = await fetch(`${API}/api/analytics`);
    const data = await res.json();

    const dayLabels = data.by_day.map((d) => d.day);
    const dayReps = data.by_day.map((d) => d.reps || 0);
    const ctx1 = document.getElementById("repsByDayChart");
    if (repsByDayChart) repsByDayChart.destroy();
    repsByDayChart = new Chart(ctx1, {
      type: "line",
      data: {
        labels: dayLabels.length ? dayLabels : ["No data"],
        datasets: [{
          label: "Reps",
          data: dayReps.length ? dayReps : [0],
          borderColor: "#B8FF3D",
          backgroundColor: "rgba(184,255,61,0.12)",
          fill: true,
          tension: 0.35,
          pointRadius: 3,
          pointBackgroundColor: "#B8FF3D",
        }],
      },
      options: chartOptions(),
    });

    const exLabels = data.by_exercise.map((e) => e.exercise);
    const exSets = data.by_exercise.map((e) => e.sets);
    const ctx2 = document.getElementById("setsByExerciseChart");
    if (setsByExerciseChart) setsByExerciseChart.destroy();
    setsByExerciseChart = new Chart(ctx2, {
      type: "bar",
      data: {
        labels: exLabels.length ? exLabels : ["No data"],
        datasets: [{
          label: "Sets",
          data: exSets.length ? exSets : [0],
          backgroundColor: "#5B8CFF",
          borderRadius: 6,
        }],
      },
      options: chartOptions(),
    });

    const tbody = document.getElementById("exerciseBreakdownBody");
    if (!data.by_exercise.length) {
      tbody.innerHTML = `<tr><td colspan="4" class="empty-row">Nothing logged yet.</td></tr>`;
    } else {
      tbody.innerHTML = data.by_exercise.map((e) => `
        <tr>
          <td class="tag-exercise">${e.exercise}</td>
          <td>${e.sets}</td>
          <td>${e.reps || 0}</td>
          <td>${Math.round((e.avg_confidence || 0) * 100)}%</td>
        </tr>
      `).join("");
    }
  } catch (e) {
    console.error(e);
  }
}

function chartOptions() {
  return {
    responsive: true,
    plugins: { legend: { display: false } },
    scales: {
      x: { ticks: { color: "#8A93A6" }, grid: { color: "#2A3040" } },
      y: { ticks: { color: "#8A93A6" }, grid: { color: "#2A3040" }, beginAtZero: true },
    },
  };
}

// ---------- Init ----------
loadExercises();
refreshStats();
updatePreviewText();
