function renderTable(el, rows) {
  if (!rows || rows.length === 0) {
    el.innerHTML = '<tr><td>No data available.</td></tr>';
    return;
  }
  const headers = Object.keys(rows[0]);
  const thead = `<tr>${headers.map((h) => `<th>${h}</th>`).join('')}</tr>`;
  const body = rows
    .map((r) => `<tr>${headers.map((h) => `<td>${r[h] ?? 'N/A'}</td>`).join('')}</tr>`)
    .join('');
  el.innerHTML = thead + body;
}

fetch('/api/results')
  .then((r) => r.json())
  .then((data) => {
    renderTable(document.getElementById('method-table'), data.method_comparison);
    renderTable(document.getElementById('di-table'), data.per_group_di);
    renderTable(document.getElementById('latent-table'), data.latent.rows || []);
    document.getElementById('latent-note').innerText = data.latent.note || '';

    const usableMethods = data.method_comparison.filter((x) => x.min_di !== null && x.min_di !== undefined);
    new Chart(document.getElementById('method-chart'), {
      type: 'bar',
      data: {
        labels: usableMethods.map((x) => x.method),
        datasets: [{ label: 'Min DI', data: usableMethods.map((x) => x.min_di), backgroundColor: '#356fb3' }],
      },
    });

    if (data.per_group_di.length > 0) {
      const groups = data.per_group_di.map((x) => x.Group);
      new Chart(document.getElementById('di-chart'), {
        type: 'bar',
        data: {
          labels: groups,
          datasets: [
            { label: 'Baseline_DI', data: data.per_group_di.map((x) => x.Baseline_DI), backgroundColor: '#9db4d3' },
            { label: 'Gender_Adv_DI', data: data.per_group_di.map((x) => x.Gender_Adv_DI), backgroundColor: '#4a78b3' },
            { label: 'Intersect_Adv_DI', data: data.per_group_di.map((x) => x.Intersect_Adv_DI), backgroundColor: '#0b3a66' },
          ],
        },
      });
    }

    const imageWrap = document.getElementById('latent-images');
    if (data.latent.images && data.latent.images.length > 0) {
      imageWrap.innerHTML = data.latent.images
        .map((src) => `<img src="${src}" alt="latent visual" style="max-width:100%;margin:8px 0;border:1px solid #ddd;border-radius:6px;"/>`)
        .join('');
    }
  });
