// Chart chips inside a card
document.querySelectorAll('.chips').forEach(function(tb){
  var btns = tb.querySelectorAll('.chip'), panes = tb.parentNode.querySelectorAll('.cpane');
  btns.forEach(function(b, i){ b.addEventListener('click', function(){
    btns.forEach(function(x, j){ x.setAttribute('aria-selected', i === j); if (panes[j]) panes[j].hidden = i !== j; }); }); });
});
// Decision simulator: logistic model p = 1/(1+exp(-(b0 + sum b*x)))
document.querySelectorAll('.sim').forEach(function(box){
  var cfg = JSON.parse(box.dataset.cfg), ins = box.querySelectorAll('input'), thr = box.querySelector('.thr');
  function upd(){
    var z = cfg.b0;
    ins.forEach(function(i){ if (i.classList.contains('thr')) return; var v = parseFloat(i.value); z += cfg.coefs[i.dataset.k] * v;
      i.parentNode.querySelector('.v').textContent = v.toFixed(cfg.dec[i.dataset.k]) + cfg.unit[i.dataset.k]; });
    var p = 1 / (1 + Math.exp(-z)), t = parseFloat(thr.value) / 100;
    thr.parentNode.querySelector('.v').textContent = thr.value + '%';
    box.querySelector('.pv').textContent = (p * 100).toFixed(1) + '%';
    var d = box.querySelector('.dec'), act = p >= t; d.textContent = act ? cfg.yes : cfg.no; d.className = 'dec ' + (act ? 'stop' : 'go');
  }
  ins.forEach(function(i){ i.addEventListener('input', upd); }); upd();
});
// Sortable tables
document.querySelectorAll('table').forEach(function(tb){
  tb.querySelectorAll('th').forEach(function(th, c){ th.addEventListener('click', function(){
    var body = tb.tBodies[0]; if (!body) return; var rows = Array.from(body.rows), dir = th.dataset.d === 'a' ? -1 : 1; th.dataset.d = dir === 1 ? 'a' : 'd';
    var num = function(s){ var n = parseFloat(s.replace(/[^0-9.\-]/g, '')); return isNaN(n) ? s : n; };
    rows.sort(function(a, b){ var x = num(a.cells[c].textContent), y = num(b.cells[c].textContent); return (x > y ? 1 : x < y ? -1 : 0) * dir; });
    rows.forEach(function(r){ body.appendChild(r); }); }); });
});
// Interactive charts: specs written by Python (report.js_chart), drawn with Chart.js in the site theme
(function(){
  if (!window.Chart) return;
  var F = { pct0: function(v){ return (v * 100).toFixed(0) + '%'; }, pct1: function(v){ return (v * 100).toFixed(1) + '%'; },
            num2: function(v){ return (+v).toFixed(2); }, num1: function(v){ return (+v).toFixed(1); }, usd: function(v){ return '$' + Math.round(v).toLocaleString(); }, int: function(v){ return Math.round(v).toLocaleString(); } };
  var fmt = function(k){ return F[k] || function(v){ return v; }; };
  Chart.defaults.font.family = '"Plus Jakarta Sans", system-ui, sans-serif'; Chart.defaults.color = '#64748B'; Chart.defaults.font.size = 13;
  document.querySelectorAll('canvas[data-chart]').forEach(function(cv){
    var s = JSON.parse(cv.dataset.chart), fx = fmt(s.fmt_x), fy = fmt(s.fmt_y), scatter = s.type === 'scatter';
    var horiz = !!s.horizontal;
    var sets = s.datasets.map(function(d){
      var o = { label: d.label, borderColor: d.color, backgroundColor: d.color + (scatter ? 'CC' : (d.fill ? '33' : 'CC')),
        borderWidth: scatter ? 0 : 2, pointRadius: scatter ? 6 : 0, pointHoverRadius: scatter ? 9 : 4, tension: .25, fill: !!d.fill };
      if (d.kind) { o.type = d.kind; o.borderDash = [6, 4]; o.backgroundColor = 'transparent'; o.borderWidth = 1.5; }
      if (d.colors) { o.backgroundColor = d.colors; o.borderColor = d.colors; o.borderWidth = 0; }
      if (s.type === 'bar') { o.borderRadius = 6; o.borderWidth = 0; }
      o.data = d.points || d.data; return o; });
    var y = { title: { display: !!s.y_title, text: s.y_title }, ticks: { callback: function(v){ return fy(v); } }, grid: { color: 'rgba(100,116,139,.16)' } };
    var x = scatter
      ? { type: 'linear', title: { display: !!s.x_title, text: s.x_title }, grid: { color: 'rgba(100,116,139,.12)' },
          ticks: s.x_ticks ? { stepSize: 1, callback: function(v){ return s.x_ticks[v] || ''; } } : { callback: function(v){ return fx(v); } },
          min: s.x_ticks ? -0.5 : undefined, max: s.x_ticks ? s.x_ticks.length - 0.5 : undefined,
          afterBuildTicks: s.x_ticks ? function(ax){ ax.ticks = s.x_ticks.map(function(_, i){ return { value: i }; }); } : undefined }
      : { grid: { display: false }, ticks: { maxRotation: 0, maxTicksLimit: 8, callback: function(v){ var l = this.getLabelForValue(v); return String(l).slice(0, 7); } } };
    if (horiz) {
      x = { title: { display: !!s.x_title, text: s.x_title }, ticks: { callback: function(v){ return fx(v); } }, grid: { color: 'rgba(100,116,139,.16)' }, beginAtZero: true };
      y = { grid: { display: false }, ticks: { color: '#334155' } };
    }
    new Chart(cv, { type: s.type, data: { labels: s.labels, datasets: sets },
      options: { indexAxis: horiz ? 'y' : 'x', responsive: true, maintainAspectRatio: false, animation: false,
        interaction: scatter ? { mode: 'nearest', intersect: true } : { mode: 'index', intersect: false, axis: horiz ? 'y' : 'x' },
        plugins: { legend: { display: sets.length > 1, labels: { color: '#475569', usePointStyle: true, boxWidth: 8 } },
          tooltip: { backgroundColor: '#1B237A', borderColor: '#1B237A', borderWidth: 0, padding: 12, cornerRadius: 10, titleColor: '#C7D2FE', bodyColor: '#FFFFFF',
            callbacks: { label: function(c){ var r = c.raw;
              if (scatter) return (r.name ? r.name + ': ' : '') + (s.x_ticks ? '' : fx(r.x) + ', ') + fy(r.y);
              return c.dataset.label + ': ' + (horiz ? fx(c.parsed.x) : fy(c.parsed.y)); } } } },
        scales: { x: x, y: y } } });
  });
})();
// Dark mode toggle (remembered per browser)
(function(){
  var b = document.querySelector('.themet'); if (!b) return;
  var root = document.documentElement, set = function(d){ if (d) root.setAttribute('data-theme', 'dark'); else root.removeAttribute('data-theme'); b.setAttribute('aria-pressed', d); };
  set(root.getAttribute('data-theme') === 'dark');
  b.addEventListener('click', function(){ var d = root.getAttribute('data-theme') !== 'dark'; set(d); try { localStorage.setItem('theme', d ? 'dark' : 'light'); } catch (e) {} });
})();
