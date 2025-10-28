/* ===========================================
   PsycheSpace — Main Script
   =========================================== */

// Wait until the page has fully loaded
document.addEventListener("DOMContentLoaded", () => {
  console.log("PsycheSpace site loaded 🚀");

  // ===== Smooth Scroll for Navigation =====
  const navLinks = document.querySelectorAll(".nav a");

  navLinks.forEach(link => {
    link.addEventListener("click", e => {
      e.preventDefault();
      const targetId = link.getAttribute("href");

      if (targetId.startsWith("#")) {
        document.querySelector(targetId).scrollIntoView({
          behavior: "smooth"
        });
      }
    });
  });

  // ===== Button Animation =====
  const ctaButton = document.getElementById("cta-btn");

  if (ctaButton) {
    ctaButton.addEventListener("click", () => {
      ctaButton.classList.add("clicked");
      setTimeout(() => {
        ctaButton.classList.remove("clicked");
      }, 300);
      alert("Welcome to PsycheSpace 🌱");
    });
  }

  // ===== Example Future Interaction Placeholder =====
  // You can expand this section later for:
  // - Dark mode toggle
  // - Dynamic content loading
  // - API data fetching
});
