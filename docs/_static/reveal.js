// Fade/slide elements in as they scroll into view. Content stays visible if
// JS is off or the user prefers reduced motion (the hidden state is opt-in).
(function () {
  if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) return;
  if (!("IntersectionObserver" in window)) return;
  var targets = document.querySelectorAll(
    ".bd-article .sd-card, .bd-article .km-reveal, .bd-article div.highlight, .bd-article .admonition"
  );
  document.documentElement.classList.add("km-js");
  var io = new IntersectionObserver(function (entries) {
    entries.forEach(function (e) {
      if (e.isIntersecting) { e.target.classList.add("km-in"); io.unobserve(e.target); }
    });
  }, { rootMargin: "0px 0px -8% 0px", threshold: 0.05 });
  targets.forEach(function (el, i) {
    el.classList.add("km-pre");
    // stagger siblings inside a grid row
    var p = el.closest(".sd-row");
    if (p) el.style.transitionDelay = (Array.prototype.indexOf.call(p.children, el.parentElement) % 4) * 80 + "ms";
    io.observe(el);
  });
})();
