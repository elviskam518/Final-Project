function toTable(title, rows) {
  if (!rows || rows.length === 0) return `<h4>${title}</h4><p>No rows.</p>`;
  const headers = Object.keys(rows[0]);
  const thead = `<tr>${headers.map(h => `<th>${h}</th>`).join('')}</tr>`;
  const body = rows.map(r => `<tr>${headers.map(h => `<td>${r[h] ?? ''}</td>`).join('')}</tr>`).join('');
  return `<h4>${title}</h4><table>${thead}${body}</table>`;
}

const analyzeForm = document.getElementById('analyze-form');
const modelForm = document.getElementById('model-form');

analyzeForm.addEventListener('submit', async (e) => {
  e.preventDefault();
  const fd = new FormData(analyzeForm);
  const res = await fetch('/api/demo/analyze', { method: 'POST', body: fd });
  const data = await res.json();
  if (!res.ok) { alert(data.detail || 'Analysis failed'); return; }

  modelForm.classList.remove('hidden');
  modelForm.querySelector('input[name="upload_id"]').value = data.upload_id;
  document.getElementById('analysis-output').innerHTML = `
    <p><strong>Upload id:</strong> ${data.upload_id}</p>
    <p><strong>Rows:</strong> ${data.analysis.row_count} | <strong>Baseline group:</strong> ${data.analysis.baseline_group}</p>
    ${toTable('Intermediate Fairness Metrics', data.analysis.fairness)}
    ${toTable('Odds Ratio by Group Terms', data.analysis.odds)}
  `;
});

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
});
