// Time Machine: cross-season matchups, computed in the browser.
//
// Mirrors cfbrank.time_machine.matchup() exactly -- read that module's
// docstring for the reasoning. Same code on the live and static sites;
// the per-season numbers are embedded in #snapshots.
(function () {
  const form = document.getElementById("time-machine-form");
  if (!form) return;

  const snapshots = JSON.parse(document.getElementById("snapshots").textContent);
  const result = document.getElementById("result");

  // Own-property check: `"constructor" in {}` is true.
  const has = (obj, key) => !!obj && Object.prototype.hasOwnProperty.call(obj, key);

  function matchup(a, teamA, b, teamB) {
    const ta = a.teams[teamA], tb = b.teams[teamB];
    const margin = ta.rating - tb.rating;
    const total = a.points_mean + b.points_mean + ta.off + tb.off - ta.def - tb.def;
    const errors = a.errors.concat(b.errors);
    const winA = errors.length
      ? errors.filter((e) => e > -margin).length / errors.length : 0.5;
    const mean = errors.reduce((s, e) => s + e, 0) / (errors.length || 1);
    const spread = Math.sqrt(errors.reduce((s, e) => s + (e - mean) ** 2, 0)
                             / (errors.length || 1));
    return { margin, total, winA, spread,
             scoreA: (total + margin) / 2, scoreB: (total - margin) / 2 };
  }

  function esc(s) {
    return String(s).replace(/[&<>"']/g, (c) => ({
      "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
    })[c]);
  }
  const fmtSigned = (n) => (n >= 0 ? "+" : "") + n.toFixed(1);

  function render(A, B, m) {
    const pctA = Math.round(m.winA * 100);
    const fav = m.margin >= 0 ? A : B;
    const bars = [[A, pctA, "bg-brick"], [B, 100 - pctA, "bg-ink"]];
    const call = Math.abs(m.margin) < 0.5 ? "Pick'em"
      : `${esc(fav.name)} by ${Math.abs(m.margin).toFixed(1)}`;
    const why = [A, B].map((s) => `
      <tr class="border-b border-dashed border-muted">
        <td class="py-1.5">${esc(s.name)}
          <span class="text-muted text-xs">#${s.row.rank} of ${esc(s.snap.label)}, ${esc(s.row.record)}</span></td>
        <td class="py-1.5 text-right tabular-nums font-bold ${s.row.rating < 0 ? "text-brick" : ""}">${fmtSigned(s.row.rating)}</td>
      </tr>`).join("");

    result.innerHTML = `
      <div class="retro-card overflow-hidden">
        <div class="scoreboard px-5 py-5">
          <div class="text-xs uppercase tracking-[0.3em] text-center mb-3 opacity-80">Neutral field &middot; never played</div>
          <div class="flex items-center justify-between gap-4">
            <div class="text-right flex-1">
              <div class="text-sm uppercase tracking-wider">${esc(A.name)}</div>
              <div class="text-4xl sm:text-5xl font-bold tabular-nums">${Math.round(m.scoreA)}</div>
            </div>
            <div class="text-sm opacity-70">VS</div>
            <div class="flex-1">
              <div class="text-sm uppercase tracking-wider">${esc(B.name)}</div>
              <div class="text-4xl sm:text-5xl font-bold tabular-nums">${Math.round(m.scoreB)}</div>
            </div>
          </div>
          <div class="mt-4 text-center">${call} &middot; total ${m.total.toFixed(1)}</div>
        </div>
        <div class="p-5 grid sm:grid-cols-2 gap-6">
          <div>
            <h3 class="section-title text-sm mb-3">Win probability</h3>
            ${bars.map(([s, pct, bar]) => `
              <div class="mb-3">
                <div class="flex justify-between text-sm mb-1">
                  <span>${esc(s.name)}</span><span class="tabular-nums font-bold">${pct}%</span>
                </div>
                <div class="h-4 bg-paper border-2 border-ink">
                  <div class="h-full ${bar}" style="width: ${pct}%"></div>
                </div>
              </div>`).join("")}
          </div>
          <div>
            <h3 class="section-title text-sm mb-3">Why</h3>
            <p class="text-xs text-muted mb-2">Points better than an average FBS team that season.</p>
            <table class="w-full text-sm"><tbody>${why}
              <tr><td class="py-1.5">Edge</td>
                <td class="py-1.5 text-right tabular-nums font-bold">${esc(fav.name)} ${Math.abs(m.margin).toFixed(1)}</td></tr>
            </tbody></table>
          </div>
        </div>
        <p class="border-t-2 border-ink bg-paper px-5 py-3 text-xs">
          <strong>&plusmn;${m.spread.toFixed(1)} points</strong> typical miss, from both seasons' real misses.
          Anything inside two touchdowns is close to a coin flip.
        </p>
      </div>`;
  }

  function renderError(message) {
    result.innerHTML = `
      <div class="retro-card p-4 text-brick font-bold" role="alert">&#9888; ${esc(message)}</div>`;
  }

  function side(key) {
    const team = document.getElementById(`${key}_team`).value.trim();
    const year = document.getElementById(`${key}_season`).value;
    const snap = snapshots[year];
    return { team, year, snap, name: `${year} ${team}`,
             row: has(snap.teams, team) ? snap.teams[team] : null };
  }

  form.addEventListener("submit", (e) => {
    e.preventDefault();
    const A = side("a"), B = side("b");
    for (const s of [A, B]) {
      if (s.row) continue;
      const everFbs = Object.values(snapshots).some((snap) => has(snap.teams, s.team));
      renderError(everFbs
        ? `${s.team} wasn't an FBS team in ${s.year}. Pick another season.`
        : `Unknown team: ${s.team}. Pick one from the list.`);
      return;
    }
    if (A.name === B.name) {
      renderError(`That's ${A.name} twice. Try another season (${A.team} vs ${A.team} from a different year works).`);
      return;
    }
    render(A, B, matchup(A.snap, A.team, B.snap, B.team));
  });
})();
