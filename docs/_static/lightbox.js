document.addEventListener("DOMContentLoaded", function () {
  var overlay = document.createElement("div");
  overlay.className = "lightbox-overlay";
  var img = document.createElement("img");
  overlay.appendChild(img);
  document.body.appendChild(overlay);

  overlay.addEventListener("click", function () {
    overlay.classList.remove("active");
  });

  document.querySelectorAll("article img").forEach(function (el) {
    if (el.closest(".tikz-diagram")) return;

    el.style.cursor = "zoom-in";
    el.addEventListener("click", function (e) {
      e.preventDefault();
      e.stopPropagation();
      img.src = el.src;
      overlay.classList.add("active");
    });
  });
});
