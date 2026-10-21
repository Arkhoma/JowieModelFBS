// Site-wide helpers, loaded (deferred) in <head> on every page, so these
// globals exist before any page script runs:
//
//   matchScore(name, conference, q)  team-search ranking (header + pickers)
//   esc(s)                           HTML-escape for innerHTML templates
//   teamIndex()                      search-index.json, fetched once, lazily
//   linkTeams(root)                  turn <span data-team data-season> into links
//
// Plus the header search box itself. search-index.json is
// {team: {c: conference, u: {season: page url}}}, built by
// app.pages.team_index -- the only place JS learns team URLs, so live
// and static sites never need a JS copy of the URL scheme.

// How well a team matches the query; lower is better, null = no match.
//   0 exact name        "tennessee" -> Tennessee
//   1 name starts with  "tenn"      -> Tennessee (not Middle Tennessee)
//   2 a word starts     "tenn"      -> Middle Tennessee
//   3 anywhere in name  "essee"
//   4 conference only   "sec"       -> every SEC team
function matchScore(name, conference, q) {
  if (!q) return 0;
  if (name === q) return 0;
  if (name.startsWith(q)) return 1;
  if (name.split(/[\s\-&().]+/).some(w => w.startsWith(q))) return 2;
  if (name.includes(q)) return 3;
  if (conference.includes(q)) return 4;
  return null;
}

// Jinja autoescapes server pages; innerHTML does not. Anything a user
// typed, or any team name, goes through here before it's interpolated.
function esc(s) {
  return String(s).replace(/[&<>"']/g, (c) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  })[c]);
}

const teamIndex = (() => {
  let pending = null;
  return () => {
    if (!pending) {
      const url = document.querySelector(".site-search").dataset.index;
      pending = fetch(url)
        .then((r) => { if (!r.ok) throw new Error(`HTTP ${r.status}`); return r.json(); })
        .catch((err) => { pending = null; throw err; });  // let a retry try again
    }
    return pending;
  };
})();

// Progressive enhancement: JS-rendered results print plain team names in
// <span data-team="LSU" data-season="2019">; they become links once the
// index arrives (instantly if the header search already loaded it).
function linkTeams(root) {
  teamIndex().then((index) => {
    root.querySelectorAll("[data-team][data-season]").forEach((el) => {
      const entry = index[el.dataset.team];
      const url = entry && entry.u[el.dataset.season];
      if (!url) return;                       // not FBS that season: stay text
      const a = document.createElement("a");
      a.href = url;
      a.className = "team-link";
      a.textContent = el.textContent;
      el.replaceWith(a);
    });
  }).catch(() => {});                         // links are a nicety, not required
}

(function () {
  const root = document.querySelector(".site-search");
  if (!root) return;
  const input = root.querySelector("input");
  const list = root.querySelector("[role=listbox]");
  const status = root.querySelector("[role=status]");
  const YEAR = /\b(?:19|20)\d{2}\b/;
  let index = null, results = [], active = -1;

  // "lsu 2019" / "2019 lsu" -> that season; plain "lsu" -> latest season.
  function search(raw) {
    const year = (raw.match(YEAR) || [])[0];
    const q = raw.replace(YEAR, "").trim().toLowerCase();
    if (!q) return [];
    return Object.entries(index)
      .map(([name, e]) => {
        const seasons = Object.keys(e.u);       // integer-like keys: ascending
        const season = year ? (e.u[year] ? year : null) : seasons[seasons.length - 1];
        return { name, conf: e.c || "", season, url: season && e.u[season],
                 score: season ? matchScore(name.toLowerCase(), (e.c || "").toLowerCase(), q) : null };
      })
      .filter((r) => r.score !== null)
      .sort((a, b) => (a.score - b.score) || a.name.localeCompare(b.name))
      .slice(0, 8);
  }

  function open(show) {
    list.hidden = !show;
    input.setAttribute("aria-expanded", String(show));
  }

  function setActive(i) {
    active = i;
    list.querySelectorAll("[role=option]").forEach((o, j) =>
      o.setAttribute("aria-selected", String(j === i)));
    if (i >= 0) input.setAttribute("aria-activedescendant", `site-search-opt-${i}`);
    else input.removeAttribute("aria-activedescendant");
  }

  function update() {
    if (!index) return;
    results = search(input.value);
    list.innerHTML = results.map((r, i) => `
      <li role="option" id="site-search-opt-${i}" aria-selected="false"
          class="px-3 py-3 cursor-pointer flex justify-between gap-3 hover:bg-paper">
        <span class="font-semibold">${esc(r.name)}</span>
        <span class="text-xs text-muted whitespace-nowrap">${esc(r.conf)}${r.conf ? " &middot; " : ""}${r.season}</span>
      </li>`).join("");
    const typed = input.value.trim() !== "";
    status.textContent = !typed ? "" : results.length
      ? `${results.length} team${results.length === 1 ? "" : "s"}` : "No matching teams";
    open(typed && results.length > 0);
    setActive(results.length ? 0 : -1);
  }

  const go = (i) => { if (results[i]) window.location.href = results[i].url; };

  input.addEventListener("focus", () => {
    teamIndex()
      .then((loaded) => { index = loaded; update(); })
      .catch(() => { status.textContent = "Team search is unavailable right now."; });
  });
  input.addEventListener("input", update);
  input.addEventListener("keydown", (e) => {
    if (e.key === "ArrowDown") {
      e.preventDefault(); setActive(Math.min(active + 1, results.length - 1));
    } else if (e.key === "ArrowUp") {
      e.preventDefault(); setActive(Math.max(active - 1, 0));
    } else if (e.key === "Enter") {
      e.preventDefault(); go(active >= 0 ? active : 0);
    } else if (e.key === "Escape") {
      open(false);
    }
  });
  list.addEventListener("mousedown", (e) => e.preventDefault());   // keep focus
  list.addEventListener("click", (e) => {
    const option = e.target.closest("[role=option]");
    if (option) go(Number(option.id.split("-").pop()));
  });
  input.addEventListener("blur", () => open(false));
})();
