// Client-side prediction engine for the static export.
//
// The live site posts to /predict and lets the FastAPI backend run
// Predictor.predict(). A static export has no backend, so this file
// reproduces that exact math from the payload embedded in #model-data
// by src/cfbrank/static_export.py -- see that module's docstring for
// why "calibrated" ratings + a sorted error list are enough to do it.
//
// Only #predict-form (the static_build branch of predict.html) wires
// this up; the live htmx form ignores it entirely.
(function () {
  const form = document.getElementById("predict-form");
  if (!form) return; // Live app: htmx handles the form instead.

  const dataEl = document.getElementById("model-data");
  const model = JSON.parse(dataEl.textContent);
  const result = document.getElementById("result");

  // Fraction of historical errors greater than -margin -- the same
  // empirical count cfbrank.predict._empirical_win_probability does.
  function winProbability(margin) {
    const errors = model.errors;
    if (!errors.length) return 0.5;
    // errors is sorted ascending; count entries > -margin via the first
    // index not satisfying (<= -margin), i.e. a manual upper_bound.
    let lo = 0, hi = errors.length;
    while (lo < hi) {
      const mid = (lo + hi) >> 1;
      if (errors[mid] <= -margin) lo = mid + 1; else hi = mid;
    }
    return (errors.length - lo) / errors.length;
  }

  function errorStd() {
    const e = model.errors;
    if (!e.length) return 0.0;
    const mean = e.reduce((a, b) => a + b, 0) / e.length;
    const variance = e.reduce((a, b) => a + (b - mean) ** 2, 0) / e.length;
    return Math.sqrt(variance);
  }

  // Own-property check: `"constructor" in {}` is true, so plain `in`
  // would wave a typed "constructor" through as a team.
  const has = (obj, key) => !!obj && Object.prototype.hasOwnProperty.call(obj, key);

  function predictTotal(home, away) {
    const hasBoth = has(model.offense, home) && has(model.offense, away)
      && has(model.defense, home) && has(model.defense, away);
    if (!hasBoth) return model.average_total;
    // Home field cancels out of a combined total; see static_export.py.
    return (2 * model.total_mean + model.offense[home] + model.offense[away]
            - model.defense[home] - model.defense[away]);
  }

  function predict(home, away, neutral) {
    const homeField = neutral ? 0.0 : model.home_field;
    const margin = (model.calibrated[home] - model.calibrated[away]) + homeField;
    const total = predictTotal(home, away);
    const homeScore = (total + margin) / 2.0;
    const awayScore = (total - margin) / 2.0;

    const factors = [
      [`${home} rating`, model.rating[home]],
      [`${away} rating`, model.rating[away]],
      ["Rating edge", model.rating[home] - model.rating[away]],
      [neutral ? "Home field (neutral)" : "Home field", homeField],
      ["Predicted margin", margin],
    ];

    return {
      home, away, neutral, margin, total, homeScore, awayScore, factors,
      winProbability: winProbability(margin),
      errorStd: errorStd(),
      favourite: margin > 0 ? home : away,
      spread: Math.abs(margin),
    };
  }

  // Jinja autoescapes on the live site; innerHTML does not. The error
  // path echoes whatever was typed, so everything interpolated goes
  // through here.
  function esc(s) {
    return String(s).replace(/[&<>"']/g, (c) => ({
      "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
    })[c]);
  }

  function fmt(n, digits) { return n.toFixed(digits); }
  function fmtSigned(n, digits) { return (n >= 0 ? "+" : "") + n.toFixed(digits); }

  function render(p) {
    const homePct = Math.round(p.winProbability * 100);
    const awayPct = 100 - homePct;
    const rows = [[p.home, homePct, "bg-crimson"], [p.away, awayPct, "bg-ink"]];

    result.innerHTML = `
      <div class="retro-card overflow-hidden">
        <div class="scoreboard px-5 py-5">
          <div class="text-xs uppercase tracking-[0.3em] text-center mb-3 opacity-80">
            ${p.neutral ? "Neutral site" : `At ${esc(p.home)}`}
          </div>
          <div class="flex items-center justify-between gap-4">
            <div class="text-right flex-1">
              <div class="text-sm uppercase tracking-wider">${esc(p.away)}</div>
              <div class="text-4xl sm:text-5xl font-bold tabular-nums">${Math.round(p.awayScore)}</div>
            </div>
            <div class="text-sm opacity-70">AT</div>
            <div class="flex-1">
              <div class="text-sm uppercase tracking-wider">${esc(p.home)}</div>
              <div class="text-4xl sm:text-5xl font-bold tabular-nums">${Math.round(p.homeScore)}</div>
            </div>
          </div>
          <div class="mt-4 text-center">
            ${esc(p.favourite)} by ${fmt(p.spread, 1)} &middot; total ${fmt(p.total, 1)}
          </div>
        </div>
        <div class="p-5 grid sm:grid-cols-2 gap-6">
          <div>
            <h3 class="font-varsity uppercase tracking-wider text-crimson mb-3">Win probability</h3>
            ${rows.map(([team, pct, bar]) => `
              <div class="mb-3">
                <div class="flex justify-between text-sm mb-1">
                  <span>${esc(team)}</span><span class="tabular-nums font-bold">${pct}%</span>
                </div>
                <div class="h-4 bg-paper border-2 border-ink">
                  <div class="h-full ${bar}" style="width: ${pct}%"></div>
                </div>
              </div>`).join("")}
          </div>
          <div>
            <h3 class="font-varsity uppercase tracking-wider text-crimson mb-3">Why</h3>
            <table class="w-full text-sm"><tbody>
              ${p.factors.map(([label, value]) => `
                <tr class="border-b border-dashed border-muted last:border-0">
                  <td class="py-1.5">${esc(label)}</td>
                  <td class="py-1.5 text-right tabular-nums font-bold ${value < 0 ? "text-crimson" : ""}">${fmtSigned(value, 2)}</td>
                </tr>`).join("")}
            </tbody></table>
          </div>
        </div>
        <p class="border-t-[3px] border-ink bg-paper px-5 py-3 text-xs">
          <strong>&plusmn;${fmt(p.errorStd, 1)} points</strong>
          typical miss. Anything inside two touchdowns is close to a coin flip.
        </p>
      </div>`;
  }

  function renderError(message) {
    result.innerHTML = `
      <div class="retro-card p-4 text-crimson font-bold" role="alert">
        &#9888; ${esc(message)}
      </div>`;
  }

  form.addEventListener("submit", (e) => {
    e.preventDefault();
    const home = document.getElementById("home_team").value.trim();
    const away = document.getElementById("away_team").value.trim();
    const neutral = document.getElementById("neutral_site").checked;

    const unknown = [away, home].filter((t) => !has(model.calibrated, t));
    if (unknown.length) {
      renderError(`Unknown team: ${unknown.join(", ")}. Pick one from the list.`);
      return;
    }
    if (home === away) {
      renderError("A team cannot play itself. Pick two different teams.");
      return;
    }
    render(predict(home, away, neutral));
  });
})();
