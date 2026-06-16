// ============================================
// COOKBOOK · MLX SCHEDULER PANEL
// Surfaces the MLX memory budget (used / free / total), the loaded MLX
// serves, and per-serve pin / unload controls — the GUI half of the P2
// governance work. Talks to:
//   GET  /api/mlx/scheduler          -> { budget_mb, used_mb, free_mb, models[] }
//   POST /api/mlx/scheduler/unload   { session_id }
//   POST /api/mlx/scheduler/pin      { session_id, pinned }
//
// The panel only appears when MLX serves are actually loaded, so non-MLX
// (Linux/CUDA/Windows) setups never see it. Self-contained: one mount call
// from _renderRunningTab plus a light self-refresh poll while it's on screen.
// ============================================

const PANEL_ID = 'mlx-scheduler-panel';
const POLL_MS = 4000;

let _pollTimer = null;
let _inFlight = false;

function _gb(mb) { return (Number(mb || 0) / 1024); }

function _fmtGb(mb) {
  const v = _gb(mb);
  return (v >= 10 ? v.toFixed(1) : v.toFixed(2)).replace(/\.?0+$/, '') + ' GB';
}

function _shortName(repo) {
  const s = String(repo || '').split('/').pop() || repo || 'model';
  return s.length > 38 ? s.slice(0, 37) + '…' : s;
}

function _fmtIdle(sec) {
  if (sec == null) return '';
  if (sec < 60) return `idle ${Math.round(sec)}s`;
  if (sec < 3600) return `idle ${Math.round(sec / 60)}m`;
  return `idle ${(sec / 3600).toFixed(1)}h`;
}

async function _api(path, opts) {
  const res = await fetch(path, {
    credentials: 'same-origin',
    headers: { 'Content-Type': 'application/json' },
    ...opts,
  });
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  return res.json();
}

async function _fetchSnapshot() {
  try { return await _api('/api/mlx/scheduler', { method: 'GET' }); }
  catch { return null; }
}

async function _unload(sessionId) {
  try { await _api('/api/mlx/scheduler/unload', { method: 'POST', body: JSON.stringify({ session_id: sessionId }) }); }
  catch {}
}

async function _setPinned(sessionId, pinned) {
  try { await _api('/api/mlx/scheduler/pin', { method: 'POST', body: JSON.stringify({ session_id: sessionId, pinned }) }); }
  catch {}
}

function _barColor(frac) {
  // green -> amber -> red as the budget fills.
  if (frac >= 0.9) return '#e5534b';
  if (frac >= 0.7) return '#d6a020';
  return '#2ea043';
}

function _renderPanel(panel, snap) {
  const models = (snap && snap.models) || [];
  const budget = snap ? snap.budget_mb : 0;
  const used = snap ? snap.used_mb : 0;
  const frac = budget > 0 ? Math.min(1, used / budget) : 0;

  const rows = models.map(m => {
    const idle = _fmtIdle(m.idle_seconds);
    const pinned = !!m.pinned;
    const star = pinned ? '★' : '☆';
    const pinTitle = pinned ? 'Pinned — exempt from eviction & idle-unload. Click to unpin.' : 'Pin to protect from eviction & idle-unload.';
    return (
      `<div class="mlx-serve-row" data-sid="${m.session_id}" style="display:flex;align-items:center;gap:8px;padding:5px 0;border-top:1px solid var(--border-subtle,rgba(128,128,128,0.18));">` +
        `<span style="flex:1;min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;" title="${m.repo_id || ''}">${_shortName(m.repo_id)}</span>` +
        `<span style="opacity:0.7;font-variant-numeric:tabular-nums;">${_fmtGb(m.footprint_mb)}</span>` +
        `<span style="opacity:0.55;font-size:0.85em;min-width:54px;text-align:right;">${idle}</span>` +
        `<button class="mlx-pin-btn" title="${pinTitle}" style="background:none;border:none;cursor:pointer;font-size:1.1em;line-height:1;padding:2px 4px;color:${pinned ? '#d6a020' : 'inherit'};opacity:${pinned ? 1 : 0.6};">${star}</button>` +
        `<button class="mlx-unload-btn" title="Stop & unload this model" style="background:none;border:none;cursor:pointer;line-height:1;padding:2px 4px;opacity:0.65;">✕</button>` +
      `</div>`
    );
  }).join('');

  panel.innerHTML =
    `<div style="display:flex;align-items:baseline;gap:8px;margin-bottom:4px;">` +
      `<strong style="font-size:0.92em;">MLX memory</strong>` +
      `<span style="margin-left:auto;opacity:0.7;font-size:0.85em;font-variant-numeric:tabular-nums;">${_fmtGb(used)} / ${_fmtGb(budget)} · ${_fmtGb(snap ? snap.free_mb : 0)} free</span>` +
    `</div>` +
    `<div style="height:7px;border-radius:4px;background:var(--border-subtle,rgba(128,128,128,0.2));overflow:hidden;">` +
      `<div style="height:100%;width:${(frac * 100).toFixed(1)}%;background:${_barColor(frac)};transition:width 0.4s ease;"></div>` +
    `</div>` +
    `<div class="mlx-serve-list" style="margin-top:6px;">${rows}</div>`;

  panel.querySelectorAll('.mlx-unload-btn').forEach(btn => {
    btn.addEventListener('click', async (e) => {
      const sid = e.target.closest('.mlx-serve-row')?.dataset.sid;
      if (!sid) return;
      btn.disabled = true; btn.style.opacity = '0.3';
      await _unload(sid);
      await refresh(panel);
    });
  });
  panel.querySelectorAll('.mlx-pin-btn').forEach(btn => {
    btn.addEventListener('click', async (e) => {
      const row = e.target.closest('.mlx-serve-row');
      const sid = row?.dataset.sid;
      if (!sid) return;
      const isPinned = btn.textContent.trim() === '★';
      btn.disabled = true;
      await _setPinned(sid, !isPinned);
      await refresh(panel);
    });
  });
}

async function refresh(panel) {
  if (!panel || !panel.isConnected || _inFlight) return;
  _inFlight = true;
  try {
    const snap = await _fetchSnapshot();
    if (!panel.isConnected) return;
    if (!snap || !snap.models || snap.models.length === 0) {
      // No MLX serves loaded — hide the panel entirely (keeps non-MLX setups clean).
      panel.style.display = 'none';
      panel.innerHTML = '';
      return;
    }
    panel.style.display = '';
    _renderPanel(panel, snap);
  } finally {
    _inFlight = false;
  }
}

function _startPoll(panel) {
  if (_pollTimer) return;
  _pollTimer = setInterval(() => {
    if (!panel.isConnected) { clearInterval(_pollTimer); _pollTimer = null; return; }
    refresh(panel);
  }, POLL_MS);
}

// Mount (idempotently) into the given container (the Running tab's admin-card).
// Safe to call on every _renderRunningTab; reuses the existing panel element.
export function mountMlxSchedulerPanel(container) {
  if (!container) return;
  let panel = container.querySelector('#' + PANEL_ID);
  if (!panel) {
    panel = document.createElement('div');
    panel.id = PANEL_ID;
    panel.className = 'mlx-scheduler-panel';
    panel.style.cssText = 'margin:8px 0 4px;padding:8px 10px;border-radius:8px;background:var(--surface-2,rgba(128,128,128,0.06));display:none;';
    // Sit just under the "Active downloads and serving processes." description.
    const desc = container.querySelector('.memory-desc');
    if (desc && desc.parentNode === container) desc.insertAdjacentElement('afterend', panel);
    else container.appendChild(panel);
  }
  refresh(panel);
  _startPoll(panel);
}
