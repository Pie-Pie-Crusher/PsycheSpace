// ======================================================================
// File: /scripts/app.js
// PsycheSpace – UI logic (no frameworks)
// - Landing flow (Proceed / Learn More sheet)
// - Multi-step questionnaire: next/back, minimal validation
// - Live range outputs + name token
// - Countdown animation -> result screen
// - Random demo score + 3 actionable tips
// - Accessible touches: ESC to close sheet, click-outside to dismiss
// ======================================================================
(() => {
  'use strict';

  // --------------------------- Utilities -------------------------------
  const $  = (sel, root = document) => root.querySelector(sel);
  const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));

  const clamp = (n, min, max) => Math.max(min, Math.min(max, n));

  // --------------------------- Elements --------------------------------
  const home        = $('.home');                 // Landing screen
  const quizSection = document.querySelector('[data-quiz]'); // Quiz wrapper (hidden initially)
  const learnSheet  = $('#learnSheet');           // Learn More dialog

  const form        = $('#quizForm');             // Questionnaire form
  const steps       = $$('.step', form);          // All step panels
  const progressBar = $('#progressBar');
  const stepNowEl   = $('#stepNow');
  const stepTotalEl = $('#stepTotal');

  // Range inputs & outputs
  const exercise    = $('#exercise');
  const stress      = $('#stress');
  const social      = $('#social');
  const exerciseOut = $('#exerciseOut');
  const stressOut   = $('#stressOut');
  const socialOut   = $('#socialOut');

  // Name token mirrors
  const nameInput   = $('#name');
  const nameTokens  = $$('.token');

  // Result elements
  const scoreOut    = $('#scoreOut');
  const tipsList    = $('#tipsList');

  // Footer year
  const yearEl      = $('#year');

  // ----------------------- State / Constants ---------------------------
  const stepCount = steps.length;    // includes the final result step
  let current = 0;                   // 0-indexed current step

  // Display total as the last *question* index (exclude final result)
  // Steps are: 1..11 questions, 12 = countdown, 13 = result → display denominator as 12
  stepTotalEl.textContent = stepCount - 1;

  // ----------------------- Learn More Sheet ----------------------------
  const openLearn  = () => learnSheet.setAttribute('aria-hidden', 'false');
  const closeLearn = () => learnSheet.setAttribute('aria-hidden', 'true');

  // Close sheet on ESC and outside click
  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape' && learnSheet.getAttribute('aria-hidden') === 'false') {
      closeLearn();
    }
  });
  learnSheet?.addEventListener('click', (e) => {
    if (e.target === learnSheet) closeLearn(); // click backdrop to close
  });

  // ----------------------- Navigation Helpers --------------------------
  function updateProgress() {
    // visible steps for the bar exclude the very last result screen
    const visibleStep = Math.min(current + 1, stepCount - 1); // clamp to (stepCount - 1)
    stepNowEl.textContent = String(visibleStep);

    // progress from step 1..(stepCount-1) mapped to 0..100
    const numerator   = Math.max(0, visibleStep - 1);
    const denominator = Math.max(1, (stepCount - 2)); // exclude result
    const pct = (numerator / denominator) * 100;
    progressBar.style.width = `${clamp(pct, 0, 100)}%`;
  }

  function showStep(index) {
    steps.forEach((s, i) => {
      if (i === index) {
        s.classList.add('is-active');
        s.setAttribute('aria-current', 'step');
      } else {
        s.classList.remove('is-active');
        s.removeAttribute('aria-current');
      }
    });
    current = index;
    updateProgress();
  }

  // ----------------------- Landing Actions -----------------------------
  // Global delegation keeps markup simple
  document.addEventListener('click', (e) => {
    const t = e.target;
    if (!(t instanceof HTMLElement)) return;

    // Open / close Learn More
    if (t.matches('[data-open-learn]')) openLearn();
    if (t.matches('[data-close-learn]')) closeLearn();

    // Start the quiz
    if (t.matches('[data-start]')) {
      home?.classList.add('is-hidden');
      quizSection?.classList.remove('is-hidden');
      showStep(0);
    }

    // Brand = Home (reset to landing)
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

    // Back
    if (target.hasAttribute('data-prev')) {
      e.preventDefault();
      showStep(Math.max(0, current - 1));
      return;
    }

    // Next / Reveal
    if (target.hasAttribute('data-next')) {
      e.preventDefault();

      // Inline validation for required inputs within the current step
      const currStep = steps[current];
      const required = $$('input[required]', currStep);
      for (const input of required) {
        if (!String(input.value || '').trim()) {
          input.focus();
          return;
        }
      }

      // Countdown step: animate, then move to result step *after* the animation
      if (currStep?.dataset.mode === 'countdown') {
        playCountdown(() => {
          showStep(clamp(current + 1, 0, stepCount - 1)); // go to result
          showResult();
        });
        return; // don't advance immediately
      }

      // Normal advance
      showStep(clamp(current + 1, 0, stepCount - 1));
      return;
    }

    // Restart
    if (target.hasAttribute('data-restart')) {
      e.preventDefault();
      resetForm();
      return;
    }
  });

  // ------------------------ Inputs / Outputs ---------------------------
  // Live outputs for range inputs
  const setOut = (input, out) => {
    if (input && out) out.textContent = String(input.value);
  };
  ['input', 'change'].forEach(evt => {
    exercise?.addEventListener(evt, () => setOut(exercise, exerciseOut));
    stress?.addEventListener(evt,   () => setOut(stress,   stressOut));
    social?.addEventListener(evt,   () => setOut(social,   socialOut));
  });

  // Live name token mirrors
  nameInput?.addEventListener('input', () => {
    const v = (nameInput.value || '').trim() || 'friend';
    nameTokens.forEach(tok => (tok.textContent = v));
  });

  // -------------------------- Countdown --------------------------------
  function playCountdown(done) {
    const bubbles = $$('.bubble', steps[current]);
    // Reset state
    bubbles.forEach(b => b.classList.remove('is-on'));

    let i = 0;
    const tick = () => {
      if (i >= bubbles.length) {
        done && done();
        return;
      }
      bubbles[i].classList.add('is-on');
      i++;
      setTimeout(tick, 500); // 0.5s per bubble
    };
    tick();
  }

  // ---------------------------- Result ---------------------------------
  function showResult() {
    // Grab inputs with sensible fallbacks
    const age = parseInt($('#age')?.value || '18', 10);
    const ex  = parseInt($('#exercise')?.value || '5', 10);
    const slp = parseFloat($('#sleep')?.value || '7', 10);
    const str = parseInt($('#stress')?.value   || '5', 10);
    const soc = parseInt($('#social')?.value   || '6', 10);

    // Lightweight demo scoring (0–100)
    const raw = Math.round(
      55 + (Math.random() * 25)      // randomness
      + ex * 1.7                     // exercise helps
      + (slp - 7) * 1.2              // around 7 hrs is neutral
      - str * 1.2                    // stress hurts
      + soc * 0.9                    // social helps
      - Math.max(0, age - 50) * 0.2  // slight age penalty after 50
    );
    const score = clamp(raw, 0, 100);
    scoreOut.textContent = String(score);

    // Suggestions pool (simple + actionable)
    const suggestions = [
      'Try a 20-minute walk three times this week',
      'Aim for a consistent sleep schedule (±30 minutes)',
      'Swap one sugary drink for water each day',
      'Do a 5-minute breathwork break when stress spikes',
      'Schedule one social check-in with a friend',
      'Reduce late-night screen time by 15 minutes',
      'Add a serving of fruit/veg to lunch',
    ];

    // Pick 3 unique suggestions
    const picks = new Set();
    while (picks.size < 3) {
      picks.add(Math.floor(Math.random() * suggestions.length));
    }
    tipsList.innerHTML = [...picks].map(i => `<li>${suggestions[i]}</li>`).join('');
  }

  // --------------------------- Reset Flow -------------------------------
  function resetForm() {
    form?.reset();
    // Reset visible outputs to defaults matching initial attributes
    if (exerciseOut) exerciseOut.textContent = exercise?.value || '5';
    if (stressOut)   stressOut.textContent   = stress?.value   || '5';
    if (socialOut)   socialOut.textContent   = social?.value   || '6';

    progressBar.style.width = '0%';
    showStep(0);
  }

  // --------------------------- Bootstrap --------------------------------
  // Footer year
  if (yearEl) yearEl.textContent = String(new Date().getFullYear());

  // If quiz is visible on load (e.g., hard refresh mid-quiz), keep UX sane
  if (!quizSection?.classList.contains('is-hidden')) {
    showStep(0);
  }
})();
