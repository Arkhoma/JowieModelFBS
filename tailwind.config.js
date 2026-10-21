// Tailwind config -- compiled ahead of time by tools/build_css.py into
// app/static/site.css (was the in-browser CDN compiler, which blocked
// every page load by ~300-500 ms). JS files are scanned too: predict.js
// and time_machine.js build class names in strings ("bg-brick").
module.exports = {
  content: ["./app/templates/**/*.html", "./app/static/**/*.js"],
  theme: { extend: {
    // Navy + brick, lifted from the NCAA's own retro pennant-and-
    // football mark: navy carries the structure (header, headings,
    // buttons, table headers); brick is the pop accent (alerts,
    // upsets, the headline stat on a page). Cream stays the page
    // background -- a 90s sports-page look.
    colors: {
      paper:   "#f5efe0",   // page background
      card:    "#fffbf2",
      ink:     "#1c1a17",
      navy:    { DEFAULT: "#2c3a63", dark: "#1f2947", light: "#dde2ef" },
      brick:   { DEFAULT: "#a3431f", dark: "#7c3015", light: "#f1ddd0" },
      muted:   "#5f574b",   // 6.3:1 on paper, WCAG AA
      line:    "#d8cfbd",   // hairline borders
    },
    // One typeface sitewide; weight and case do the hierarchy.
    fontFamily: { sans: ["Archivo", "Helvetica Neue", "Arial", "sans-serif"] },
  }},
};
