// Open the search of Read the Docs, which searches on its server and
// records search analytics, instead of the theme's search, which searches
// in the browser. conf.py includes this script only in builds on Read the
// Docs, so local builds keep the theme's search.
function showReadTheDocsSearch(event) {
  event.preventDefault();
  event.stopImmediatePropagation();
  document.dispatchEvent(new CustomEvent("readthedocs-search-show"));
}

document.addEventListener("DOMContentLoaded", function () {
  // The search buttons of the navbar and the sidebar.
  document.querySelectorAll(".search-button__button").forEach(function (button) {
    button.addEventListener("click", showReadTheDocsSearch, true);
  });
});

// The theme's shortcut, Ctrl+K or Cmd+K. Read the Docs opens its search on
// "/" by itself.
document.addEventListener("keydown", function (event) {
  if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === "k") {
    showReadTheDocsSearch(event);
  }
}, true);
