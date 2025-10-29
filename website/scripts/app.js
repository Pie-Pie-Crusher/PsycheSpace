// ======================================================================
// PsycheSpace – UI logic (no frameworks)
// - Landing flow (Proceed / Learn More sheet)
// - Multi-step questionnaire: next/back, minimal validation
// - Live range outputs + name token
// - Countdown animation -> result screen
// - Random demo score + 3 actionable tips
// - Floating particles background (#5b49b0) on #0b042c
// - Accessible touches: ESC to close sheet, click-outside to dismiss
// ======================================================================
(() => {
  'use strict';
  console.log('App.js loaded successfully!');

  // --------------------------- Utilities -------------------------------
  const $  = (sel, root = document) => root.querySelector(sel);
  const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));
  const clamp = (n, min, max) => Math.max(min, Math.min(max, n));

  // --------------------------- Particles --------------------------------
  // Simple & light: floating, slowly drifting circles
  const canvas = $('#bgParticles');
  const ctx = canvas.getContext('2d');
  const PARTICLE_COLOR = '#5b49b0';     // requested particle color
  const PARTICLE_COUNT = 70;            // tweak if you want more/less
  const SPEED = 0.15;                   // base speed

  let particles = [];
  let dpr = Math.max(1, window.devicePixelRatio || 1);
  let width = 0, height = 0;

  function resizeCanvas() {
    width = window.innerWidth;
    height = window.innerHeight;
    canvas.width = Math.floor(width * dpr);
    canvas.height = Math.floor(height * dpr);
    canvas.style.width = width + 'px';
    canvas.style.height = height + 'px';
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  }

  function makeParticle() {
    // random position and drift direction
    const r = Math.random() * 2 + 1.6; // radius
    const angle = Math.random() * Math.PI * 2;
    return {
      x: Math.random() * width,
      y: Math.random() * height,
      r,
      vx: Math.cos(angle) * (SPEED + Math.random() * 0.25),
      vy: Math.sin(angle) * (SPEED + Math.random() * 0.25),
      alpha: 0.35 + Math.random() * 0.4
    };
  }

  function initParticles() {
    particles = Array.from({ length: PARTICLE_COUNT }, makeParticle);
  }

  function stepParticles() {
    ctx.clearRect(0, 0, width, height);
    for (const p of particles) {
      // move
      p.x += p.vx;
      p.y += p.vy;

      // wrap around edges
      if (p.x < -10) p.x = width + 10;
      if (p.x > width + 10) p.x = -10;
      if (p.y < -10) p.y = height + 10;
      if (p.y > height + 10) p.y = -10;

      // draw
      ctx.beginPath();
      ctx.globalAlpha = p.alpha;
      ctx.fillStyle = PARTICLE_COLOR;
      ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2);
      ctx.fill();
    }
    ctx.globalAlpha = 1;
    requestAnimationFrame(stepParticles);
  }

  // --------------------------- Elements --------------------------------
  const home        = $('.home');
  const quizSection = document.querySelector('[data-quiz]');
  const learnSheet  = $('#learnSheet');

  const form        = $('#quizForm');
  const steps       = $$('.step', form);
  const progressBar = $('#progressBar');
  const stepNowEl   = $('#stepNow');
  const stepTotalEl = $('#stepTotal');

  const exercise    = $('#exercise');
  const stress      = $('#stress');
  const social      = $('#social');
  const exerciseOut = $('#exerciseOut');
  const stressOut   = $('#stressOut');
  const socialOut   = $('#socialOut');

  const nameInput   = $('#name');
  const nameTokens  = $$('.token');

  const scoreOut    = $('#scoreOut');
  const tipsList    = $('#tipsList');
  const yearEl      = $('#year');

  // ----------------------- State / Constants ---------------------------
  const stepCount = steps.length;    // includes final result step
  let current = 0;

  // denominator shown excludes the very last result screen
  stepTotalEl.textContent = stepCount - 1;

  // ----------------------- Learn More Sheet ----------------------------
  const openLearn  = () => learnSheet.setAttribute('aria-hidden', 'false');
  const closeLearn = () => learnSheet.setAttribute('aria-hidden', 'true');

  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape' && learnSheet.getAttribute('aria-hidden') === 'false') {
      closeLearn();
    }
  });
  learnSheet?.addEventListener('click', (e) => {
    if (e.target === learnSheet) closeLearn();
  });

  // ----------------------- Progress / Steps ----------------------------
  function updateProgress() {
    const visibleStep = Math.min(current + 1, stepCount - 1);
    stepNowEl.textContent = String(visibleStep);
    const numerator   = Math.max(0, visibleStep - 1);
    const denominator = Math.max(1, (stepCount - 2)); // exclude result
    const pct = (numerator / denominator) * 100;
    progressBar.style.width = `${clamp(pct, 0, 100)}%`;
  }

  function showStep(index) {
    steps.forEach((s, i) => {
      if (i === index) {
        s.classList.add('is-active'); s.setAttribute('aria-current', 'step');
      } else {
        s.classList.remove('is-active'); s.removeAttribute('aria-current');
      }
    });
    current = index;
    updateProgress();
  }

  // ----------------------- Landing Actions -----------------------------
  document.addEventListener('click', (e) => {
    const t = e.target;
    if (!(t instanceof HTMLElement)) return;

    // Sheet
    if (t.matches('[data-open-learn]')) openLearn();
    if (t.matches('[data-close-learn]')) closeLearn();

    // Proceed -> show quiz
    if (t.matches('[data-start]')) {
      console.log('Proceed button clicked!');
      console.log('home element:', home);
      console.log('quizSection element:', quizSection);
      home?.classList.add('is-hidden');
      quizSection?.classList.remove('is-hidden');
      showStep(0);
    }

    // Brand click -> go home (also closes sheet)
    if (t.matches('[data-home]')) {
      e.preventDefault();
      closeLearn();
      quizSection?.classList.add('is-hidden');
      home?.classList.remove('is-hidden');
    }
  });

  // ----------------------- Form Navigation -----------------------------
  form?.addEventListener('click', (e) => {
    const target = e.target;
    if (!(target instanceof HTMLElement)) return;

    if (target.hasAttribute('data-prev')) {
      e.preventDefault();
      showStep(Math.max(0, current - 1));
      return;
    }

    if (target.hasAttribute('data-next')) {
      e.preventDefault();
      const currStep = steps[current];
      const required = $$('input[required]', currStep);
      for (const input of required) {
        if (!String(input.value || '').trim()) { input.focus(); return; }
      }

      // countdown animation before result
      if (currStep?.dataset.mode === 'countdown') {
        playCountdown(() => {
          showStep(clamp(current + 1, 0, stepCount - 1)); // go to result
          showResult();
        });
        return;
      }

      showStep(clamp(current + 1, 0, stepCount - 1));
      return;
    }

    if (target.hasAttribute('data-restart')) {
      e.preventDefault();
      resetForm();
      return;
    }
  });

  // ------------------------ Inputs / Outputs ---------------------------
  const setOut = (input, out) => { if (input && out) out.textContent = String(input.value); };
  ['input', 'change'].forEach(evt => {
    exercise?.addEventListener(evt, () => setOut(exercise, exerciseOut));
    stress?.addEventListener(evt,   () => setOut(stress,   stressOut));
    social?.addEventListener(evt,   () => setOut(social,   socialOut));
  });

  nameInput?.addEventListener('input', () => {
    const v = (nameInput.value || '').trim() || 'friend';
    nameTokens.forEach(tok => (tok.textContent = v));
  });

  // -------------------------- Countdown --------------------------------
  function playCountdown(done) {
    const bubbles = $$('.bubble', steps[current]);
    bubbles.forEach(b => b.classList.remove('is-on'));
    let i = 0;
    const tick = () => {
      if (i >= bubbles.length) { done && done(); return; }
      bubbles[i].classList.add('is-on'); i++;
      setTimeout(tick, 500);
    };
    tick();
  }

  // ---------------------------- Result ---------------------------------
  // ---------------------------- Result (uses model) ----------------------
function showResult() {
  // Gather input values
  const payload = {
    Country: $('#country')?.value || 'USA',
    Age: parseInt($('#age')?.value || '30', 10),
    Gender: $('#gender')?.value || 'Male',
    "Exercise Level": parseInt($('#exercise')?.value || '5', 10) <= 3 ? 'Low' : 'Moderate',
    "Diet Type": $('#diet')?.value || 'Balanced',
    "Sleep Hours": parseFloat($('#sleep')?.value || '7'),
    "Stress Level": parseInt($('#stress')?.value || '5') <= 3 ? 'Low' : 'High',
    "Mental Health Condition": $('#mental')?.value || 'None',
    "Work Hours per Week": parseInt($('#work')?.value || '40', 10),
    "Screen Time per Day (Hours)": parseFloat($('#screen')?.value || '3'),
    "Social Interaction Score": parseInt($('#social')?.value || '5', 10)
  };

  fetch('http://127.0.0.1:5001/predict', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload)
  })
  .then(response => response.json())
  .then(data => {
    if (data.prediction !== undefined) {
      const score = clamp(Math.round(data.prediction), 0, 100);
      scoreOut.textContent = String(score);
    } else if (data.error) {
      console.error("Prediction error:", data.error);
      scoreOut.textContent = "Error";
    }
  })
  .catch(err => {
    console.error("Request failed:", err);
    scoreOut.textContent = "Error";
  });

  // Suggestions (still random tips)
  const suggestions = [
    'Try a 20-minute walk three times this week',
    'Aim for a consistent sleep schedule (±30 minutes)',
    'Swap one sugary drink for water each day',
    'Do a 5-minute breathwork break when stress spikes',
    'Schedule one social check-in with a friend',
    'Reduce late-night screen time by 15 minutes',
    'Add a serving of fruit/veg to lunch',
  ];
  const picks = new Set();
  while (picks.size < 3) picks.add(Math.floor(Math.random() * suggestions.length));
  tipsList.innerHTML = [...picks].map(i => `<li>${suggestions[i]}</li>`).join('');
}


  // --------------------------- Reset Flow -------------------------------
  function resetForm() {
    form?.reset();
    if (exerciseOut) exerciseOut.textContent = exercise?.value || '5';
    if (stressOut)   stressOut.textContent   = stress?.value   || '5';
    if (socialOut)   socialOut.textContent   = social?.value   || '6';
    progressBar.style.width = '0%';
    showStep(0);
    // also return to landing
    quizSection?.classList.add('is-hidden');
    home?.classList.remove('is-hidden');
  }

  // --------------------------- Bootstrap --------------------------------
  // Footer year
  const y = new Date().getFullYear();
  if (yearEl) yearEl.textContent = String(y);

  // Particles init
  resizeCanvas();
  initParticles();
  requestAnimationFrame(stepParticles);
  window.addEventListener('resize', () => {
    resizeCanvas();
    // keep same count but re-position softly
    if (particles.length === 0) initParticles();
  });

  // If quiz is visible on load (hard refresh), start at step 0
  if (!quizSection?.classList.contains('is-hidden')) showStep(0);
})();
