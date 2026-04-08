function toTable(title, rows) {
  if (!rows || rows.length === 0) return `<h4>${title}</h4><p>No rows.</p>`;
  const headers = Object.keys(rows[0]);
<<<<<<< HEAD
  const thead = `<tr>${headers.map((h) => `<th>${h}</th>`).join('')}</tr>`;
  const body = rows.map((r) => `<tr>${headers.map((h) => `<td>${r[h] ?? ''}</td>`).join('')}</tr>`).join('');
=======
  const thead = `<tr>${headers.map(h => `<th>${h}</th>`).join('')}</tr>`;
  const body = rows.map(r => `<tr>${headers.map(h => `<td>${r[h] ?? ''}</td>`).join('')}</tr>`).join('');
>>>>>>> origin/main
  return `<h4>${title}</h4><table>${thead}${body}</table>`;
}

const analyzeForm = document.getElementById('analyze-form');
<<<<<<< HEAD
const jobForm = document.getElementById('job-form');
const statusEl = document.getElementById('job-status');
const logsEl = document.getElementById('job-logs');

let currentJobId = null;
let logCursor = 0;
let timer = null;

function stopPolling() {
  if (timer) {
    clearInterval(timer);
    timer = null;
  }
}

async function pollJob() {
  if (!currentJobId) return;

  const statusRes = await fetch(`/api/jobs/${currentJobId}`);
  const statusData = await statusRes.json();
  if (!statusRes.ok) {
    statusEl.innerText = `Error: ${statusData.detail || 'status fetch failed'}`;
    stopPolling();
    return;
  }

  const pct = Math.round((statusData.progress || 0) * 100);
  statusEl.innerText = `Job ${currentJobId} | status: ${statusData.status} | progress: ${pct}% | ${statusData.progress_text || ''}`;

  const logRes = await fetch(`/api/jobs/${currentJobId}/logs?since=${logCursor}`);
  const logData = await logRes.json();
  if (logRes.ok) {
    if (logData.logs.length > 0) {
      logsEl.textContent += `${logData.logs.join('\n')}\n`;
      logsEl.scrollTop = logsEl.scrollHeight;
    }
    logCursor = logData.next;
  }

  if (statusData.status === 'completed') {
    stopPolling();
    const resultRes = await fetch(`/api/jobs/${currentJobId}/result`);
    const resultData = await resultRes.json();
    document.getElementById('model-output').innerHTML = `
      <h4>Final Output</h4>
      <pre>${JSON.stringify(resultData.result, null, 2)}</pre>
      ${resultData.result.fairness ? toTable('Final Per-group Fairness Output', resultData.result.fairness) : ''}
      ${resultData.result.latent_images ? resultData.result.latent_images.map((src) => `<img src="${src}" style="max-width:100%;margin:8px 0;border:1px solid #ddd;border-radius:6px;"/>`).join('') : ''}
    `;
  }

  if (statusData.status === 'failed') {
    stopPolling();
  }
}
=======
const modelForm = document.getElementById('model-form');
>>>>>>> origin/main

analyzeForm.addEventListener('submit', async (e) => {
  e.preventDefault();
  const fd = new FormData(analyzeForm);
  const res = await fetch('/api/demo/analyze', { method: 'POST', body: fd });
  const data = await res.json();
<<<<<<< HEAD
  if (!res.ok) {
    alert(data.detail || 'Analysis failed');
    return;
  }

  jobForm.classList.remove('hidden');
  jobForm.querySelector('input[name="upload_id"]').value = data.upload_id;
=======
  if (!res.ok) { alert(data.detail || 'Analysis failed'); return; }

  modelForm.classList.remove('hidden');
  modelForm.querySelector('input[name="upload_id"]').value = data.upload_id;
>>>>>>> origin/main
  document.getElementById('analysis-output').innerHTML = `
    <p><strong>Upload id:</strong> ${data.upload_id}</p>
    <p><strong>Rows:</strong> ${data.analysis.row_count} | <strong>Baseline group:</strong> ${data.analysis.baseline_group}</p>
    ${toTable('Intermediate Fairness Metrics', data.analysis.fairness)}
    ${toTable('Odds Ratio by Group Terms', data.analysis.odds)}
  `;
});

<<<<<<< HEAD
jobForm.addEventListener('submit', async (e) => {
  e.preventDefault();
  stopPolling();
  logsEl.textContent = '';
  logCursor = 0;
  document.getElementById('model-output').innerHTML = '';

  const fd = new FormData(jobForm);
  const runLatentChecked = jobForm.querySelector('input[name="run_latent"]').checked;
  if (!runLatentChecked) fd.delete('run_latent');

  const res = await fetch('/api/jobs', { method: 'POST', body: fd });
  const data = await res.json();
  if (!res.ok) {
    alert(data.detail || 'Failed to start job');
    return;
  }

  currentJobId = data.job_id;
  statusEl.innerText = `Job ${currentJobId} | status: ${data.status}`;

  timer = setInterval(pollJob, 2500);
  pollJob();
=======
modelForm.addEventListener('submit', async (e) => {
  e.preventDefault();
  const fd = new FormData(modelForm);
  document.getElementById('model-output').innerHTML = '<p>Running model, please wait...</p>';
  const res = await fetch('/api/demo/run-model', { method: 'POST', body: fd });
  const data = await res.json();
  if (!res.ok) { alert(data.detail || 'Model run failed'); return; }

  document.getElementById('model-output').innerHTML = `
    <h4>Final Output</h4>
    <pre>${JSON.stringify(data, null, 2)}</pre>
    ${data.fairness ? toTable('Final Per-group Fairness Output', data.fairness) : ''}
  `;
>>>>>>> origin/main
});
