// Searchable team picker (WAI-ARIA 1.2 combobox, list autocomplete).
// Type to filter; arrows move; Enter/click picks; Esc closes; the toggle
// button opens the full list like a dropdown.

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

(function () {
  function init(root) {
    const input = root.querySelector("input[role=combobox]");
    const list = root.querySelector("[role=listbox]");
    const toggle = root.querySelector("button");
    const status = root.querySelector("[role=status]");
    const options = Array.from(list.children);  // server order = model rank
    let active = -1;

    // DOM order is the visual order, so reordering nodes keeps arrow keys,
    // aria-activedescendant and what the user sees in agreement.
    const visible = () => Array.from(list.children).filter(o => !o.hidden);

    function open(show) {
      list.hidden = !show;
      input.setAttribute("aria-expanded", String(show));
      toggle.setAttribute("aria-expanded", String(show));
      if (!show) setActive(-1);
    }

    function setActive(i) {
      const shown = visible();
      options.forEach(o => o.setAttribute("aria-selected", "false"));
      active = i;
      if (i < 0 || !shown[i]) { input.removeAttribute("aria-activedescendant"); return; }
      shown[i].setAttribute("aria-selected", "true");
      input.setAttribute("aria-activedescendant", shown[i].id);
      shown[i].scrollIntoView({ block: "nearest" });
    }

    function filter() {
      const q = input.value.trim().toLowerCase();
      const ranked = options
        .map((o, rank) => ({ o, rank, score: matchScore(
          o.dataset.value.toLowerCase(), o.dataset.conference || "", q) }))
        .sort((a, b) => ((a.score ?? 99) - (b.score ?? 99)) || (a.rank - b.rank));
      ranked.forEach(({ o, score }) => { o.hidden = score === null; list.appendChild(o); });
      const n = visible().length;
      status.textContent = n ? `${n} teams` : "No matching teams";
      open(true);
      setActive(n ? 0 : -1);
    }

    function showAll() {
      options.forEach(o => { o.hidden = false; list.appendChild(o); });
    }

    function pick(option) {
      input.value = option.dataset.value;
      open(false);
      input.focus();
    }

    input.addEventListener("input", filter);
    input.addEventListener("keydown", e => {
      const n = visible().length;
      if (e.key === "ArrowDown") {
        e.preventDefault();
        if (list.hidden) { filter(); return; }
        setActive(Math.min(active + 1, n - 1));
      } else if (e.key === "ArrowUp") {
        e.preventDefault();
        setActive(Math.max(active - 1, 0));
      } else if (e.key === "Enter" && !list.hidden && active >= 0) {
        e.preventDefault();
        pick(visible()[active]);
      } else if (e.key === "Escape") {
        open(false);
      }
    });
    toggle.addEventListener("click", () => {
      if (list.hidden) {
        showAll();
        open(true);
        input.focus();
      } else {
        open(false);
      }
    });
    list.addEventListener("mousedown", e => e.preventDefault()); // keep focus
    list.addEventListener("click", e => {
      const option = e.target.closest("[role=option]");
      if (option) pick(option);
    });
    input.addEventListener("blur", () => open(false));
  }

  document.querySelectorAll("[data-combobox]").forEach(init);

  // Swap home/away without retyping.
  const swap = document.getElementById("swap-teams");
  if (swap) swap.addEventListener("click", () => {
    const home = document.getElementById("home_team");
    const away = document.getElementById("away_team");
    [home.value, away.value] = [away.value, home.value];
  });
})();
