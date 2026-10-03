// Landing page: filter pills + search over findings, and the performance chart
document.addEventListener('DOMContentLoaded', function(){
  var tier = 'all', q = '';
  var items = document.querySelectorAll('#list a, #cards .proj');
  function apply(){ var n = 0;
    items.forEach(function(el){ var ok = (tier === 'all' || el.dataset.tier === tier) && (!q || el.dataset.text.indexOf(q) > -1);
      el.hidden = !ok; if (ok && el.parentNode.id === 'list') n++; });
    document.getElementById('empty').hidden = n > 0; document.getElementById('count').textContent = n + ' of 5'; }
  document.querySelectorAll('.pill[data-f]').forEach(function(p){ p.addEventListener('click', function(){
    document.querySelectorAll('.pill[data-f]').forEach(function(x){ x.setAttribute('aria-selected', x === p); }); tier = p.dataset.f; apply(); }); });
  document.getElementById('q').addEventListener('input', function(ev){ q = ev.target.value.trim().toLowerCase(); apply(); }); apply();

  var S = JSON.parse(document.getElementById('series').textContent), view = 'growth', range = 0;
  if (!window.Chart) return;
  var ctx = document.getElementById('perf').getContext('2d');
  var grad = ctx.createLinearGradient(0, 0, 0, 320); grad.addColorStop(0, 'rgba(47,91,234,.28)'); grad.addColorStop(1, 'rgba(47,91,234,0)');
  var money = function(v){ return '$' + Math.round(v).toLocaleString(); };
  var chart = new Chart(ctx, { type: 'line', data: { labels: [], datasets: [
      { label: 'Steady-Ride fund', data: [], borderColor: '#2F5BEA', backgroundColor: grad, fill: true, borderWidth: 2, pointRadius: 0, tension: .3 },
      { label: 'S&P 500', data: [], borderColor: '#F5A04A', borderWidth: 1.5, pointRadius: 0, tension: .3, fill: false } ] },
    options: { responsive: true, maintainAspectRatio: false, animation: false, interaction: { mode: 'index', intersect: false },
      plugins: { legend: { labels: { color: '#475569', usePointStyle: true, boxWidth: 8 } },
        tooltip: { backgroundColor: '#1B237A', borderColor: '#1B237A', borderWidth: 0, padding: 12, cornerRadius: 10, titleColor: '#C7D2FE', bodyColor: '#FFFFFF',
          callbacks: { label: function(c){ return c.dataset.label + ': ' + (view === 'growth' ? money(c.parsed.y) : c.parsed.y.toFixed(1) + '%'); } } } },
      scales: { x: { ticks: { color: '#64748B', maxTicksLimit: 8, callback: function(v){ var d = this.getLabelForValue(v); return d.slice(0, 7); } }, grid: { display: false } },
                y: { ticks: { color: '#64748B', callback: function(v){ return view === 'growth' ? '$' + (v / 1000) + 'k' : v + '%'; } }, grid: { color: 'rgba(100,116,139,.16)' } } } } });
  function draw(){ var n = S.dates.length, s = range ? Math.max(0, n - range) : 0;
    chart.data.labels = S.dates.slice(s);
    chart.data.datasets[0].data = (view === 'growth' ? S.rule : S.rule_dd).slice(s);
    chart.data.datasets[1].data = (view === 'growth' ? S.spy : S.spy_dd).slice(s);
    chart.data.datasets[0].fill = view === 'growth' ? true : 'origin'; chart.update(); }
  document.querySelectorAll('.chip[data-v]').forEach(function(b){ b.addEventListener('click', function(){
    document.querySelectorAll('.chip[data-v]').forEach(function(x){ x.setAttribute('aria-selected', x === b); }); view = b.dataset.v; draw(); }); });
  document.querySelectorAll('.chip[data-r]').forEach(function(b){ b.addEventListener('click', function(){
    document.querySelectorAll('.chip[data-r]').forEach(function(x){ x.setAttribute('aria-selected', x === b); }); range = +b.dataset.r; draw(); }); });
  draw();
});
