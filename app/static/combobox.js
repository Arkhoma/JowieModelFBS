// Searchable team picker (WAI-ARIA 1.2 combobox, list autocomplete).
// Type to filter; arrows move; Enter/click picks; Esc closes; the toggle
// button opens the full list like a dropdown.
(function () {
  function init(root) {
    const input = root.querySelector("input[role=combobox]");
    const list = root.querySelector("[role=listbox]");
    const toggle = root.querySelector("button");
    const status = root.querySelector("[role=status]");
    const options = Array.from(list.children);
    let active = -1;

    const visible = () => options.filter(o => !o.hidden);

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
      options.forEach(o => { o.hidden = q && !o.dataset.search.includes(q); });
      const n = visible().length;
      status.textContent = n ? `${n} teams` : "No matching teams";
      open(true);
      setActive(n ? 0 : -1);
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
        options.forEach(o => { o.hidden = false; });
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
