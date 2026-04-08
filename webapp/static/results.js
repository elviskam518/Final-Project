function renderTable(el, rows) {
  if (!rows || rows.length === 0) { el.innerHTML = '<tr><td>No data available.</td></tr>'; return; }
  const headers = Object.keys(rows[0]);
  const thead = `<tr>${headers.map(h => `<th>${h}</th>`).join('')}</tr>`;
  const body = rows.map(r => `<tr>${headers.map(h => `<td>${r[h] ?? ''}</td>`).join('')}</tr>`).join('');
  el.innerHTML = thead + body;
}

fetch('/api/results').then(r => r.json()).then(data => {
  renderTable(document.getElementById('method-table'), data.method_comparison);
  renderTable(document.getElementById('di-table'), data.per_group_di);
  renderTable(document.getElementById('latent-table'), data.latent.rows || []);
  document.getElementById('latent-note').innerText = data.latent.note || '';

  const methodCtx = document.getElementById('method-chart');
  new Chart(methodCtx, {
    type: 'bar',
    data: {
      labels: data.method_comparison.map(x => x.method),
      datasets: [{ label: 'Min DI', data: data.method_comparison.map(x => x.min_di), backgroundColor: '#356fb3' }]
    }
  });

  if (data.per_group_di.length > 0) {
    const groups = data.per_group_di.map(x => x.Group);
    new Chart(document.getElementById('di-chart'), {
      type: 'bar',
      data: {
        labels: groups,
        datasets: [
          { label: 'Baseline_DI', data: data.per_group_di.map(x => x.Baseline_DI), backgroundColor: '#9db4d3' },
          { label: 'Gender_Adv_DI', data: data.per_group_di.map(x => x.Gender_Adv_DI), backgroundColor: '#4a78b3' },
          { label: 'Intersect_Adv_DI', data: data.per_group_di.map(x => x.Intersect_Adv_DI), backgroundColor: '#0b3a66' }
        ]
      }
    });
  }
});
