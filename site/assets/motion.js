/* Decorative signal and hero depth, independent of audio playback. */
(() => {
  'use strict';
  const root = document.documentElement;
  const hero = document.querySelector('.hero');
  const toggle = document.querySelector('#motion-toggle');
  if (!hero || !toggle) return;

  const reduced = matchMedia('(prefers-reduced-motion: reduce)');
  const preferenceKey = 'mlx-speech-motion';
  let paused = false;
  try { paused = localStorage.getItem(preferenceKey) === 'paused'; } catch { /* Optional. */ }
  let enabled = false;
  let visible = true;
  let frame = 0;
  let start = 0;
  let distance = 1;

  const render = () => {
    frame = 0;
    const progress = enabled ? Math.min(1, Math.max(0, (scrollY - start) / distance)) : 0;
    hero.style.setProperty('--hero-progress', progress.toFixed(4));
  };
  const schedule = () => {
    if (!frame && enabled && visible && !document.hidden) frame = requestAnimationFrame(render);
  };
  const measure = () => {
    start = Math.max(0, hero.offsetTop - 120);
    distance = Math.max(1, Math.min(innerHeight * .75, hero.offsetHeight * .75));
    schedule();
  };
  const updateActivity = () => {
    hero.dataset.signalActive = String(enabled && visible && !document.hidden);
    schedule();
  };
  const applyPreference = () => {
    if (frame) cancelAnimationFrame(frame);
    enabled = !paused && !reduced.matches;
    root.dataset.motion = enabled ? 'on' : 'off';
    toggle.disabled = reduced.matches;
    toggle.querySelector('[data-motion-label]').textContent = reduced.matches ? 'Animation off' : enabled ? 'Pause animation' : 'Resume animation';
    toggle.querySelector('.motion-symbol').textContent = enabled ? 'Ⅱ' : '∿';
    toggle.title = reduced.matches ? 'Your device prefers reduced motion' : enabled ? 'Pause decorative animation and parallax' : 'Resume decorative animation and parallax';
    render();
    updateActivity();
  };

  toggle.addEventListener('click', () => {
    paused = !paused;
    try { localStorage.setItem(preferenceKey, paused ? 'paused' : 'on'); } catch { /* Still works. */ }
    applyPreference();
  });
  reduced.addEventListener('change', applyPreference);
  window.addEventListener('scroll', schedule, { passive: true });
  window.addEventListener('resize', measure);
  window.addEventListener('pageshow', () => { measure(); applyPreference(); });
  document.addEventListener('visibilitychange', updateActivity);
  new ResizeObserver(measure).observe(hero);
  new IntersectionObserver(([entry]) => {
    visible = entry.isIntersecting;
    updateActivity();
  }).observe(hero);
  measure();
  applyPreference();
  toggle.hidden = false;
})();
