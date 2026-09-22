(function () {
  "use strict";

  // ── Colour scheme toggle ───────────────────────────────────────────────────

  var STORAGE_KEY = "quarto-color-scheme";

  function setScheme(mode) {
    if (mode === "dark") {
      document.documentElement.setAttribute("data-theme", "dark");
    } else {
      document.documentElement.removeAttribute("data-theme");
    }
    try {
      localStorage.setItem(STORAGE_KEY, mode);
    } catch (e) {}
  }

  var toggle = document.querySelector(".color-scheme-toggle");
  if (toggle) {
    toggle.addEventListener("click", function (e) {
      e.preventDefault();
      var isDark = document.documentElement.getAttribute("data-theme") === "dark";
      setScheme(isDark ? "light" : "dark");
    });
  }

  // ── Mobile navigation ──────────────────────────────────────────────────────

  var toggler = document.querySelector(".navbar-toggler");
  if (toggler) {
    toggler.addEventListener("click", function () {
      var container = document.querySelector(".navbar-container");
      var open = container.classList.toggle("open");
      toggler.setAttribute("aria-expanded", open ? "true" : "false");
    });
  }

  // ── Blog listing: filter by category ───────────────────────────────────────

  var listing = document.getElementById("blog-listing");
  if (!listing) return;

  var posts = Array.prototype.slice.call(listing.querySelectorAll(".quarto-post"));
  var sections = Array.prototype.slice.call(listing.querySelectorAll(".listing-section"));
  var noMatching = listing.querySelector(".listing-no-matching");
  var sidebarCategories = Array.prototype.slice.call(
    document.querySelectorAll("#margin-sidebar .category")
  );

  function applyCategory(category) {
    var matched = 0;

    posts.forEach(function (post) {
      var cats = (post.getAttribute("data-categories") || "").split(",");
      var show = !category || cats.indexOf(category) !== -1;
      post.classList.toggle("hidden", !show);
      if (show) matched++;
    });

    // Hide a year heading whose posts are all filtered out.
    sections.forEach(function (section) {
      var visible = section.querySelectorAll(".quarto-post:not(.hidden)").length;
      section.classList.toggle("hidden", visible === 0);
    });

    if (noMatching) noMatching.classList.toggle("visible", matched === 0);

    sidebarCategories.forEach(function (el) {
      el.classList.toggle("active", (el.getAttribute("data-category") || "") === (category || ""));
    });

    var hash = category ? "#category=" + encodeURIComponent(category) : " ";
    history.replaceState(null, "", window.location.pathname + hash.trim());
  }

  sidebarCategories.forEach(function (el) {
    el.addEventListener("click", function () {
      applyCategory(el.getAttribute("data-category") || "");
    });
  });

  listing.addEventListener("click", function (e) {
    var tag = e.target.closest(".listing-category");
    if (!tag) return;
    e.preventDefault();
    applyCategory(tag.getAttribute("data-category") || "");
  });

  var initial = "";
  var m = window.location.hash.match(/^#category=(.*)$/);
  if (m) initial = decodeURIComponent(m[1]);
  applyCategory(initial);
})();
