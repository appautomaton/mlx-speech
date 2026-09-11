/* Apply the edition and theme before the stylesheet paints. */
(() => {
  'use strict';
  const root = document.documentElement;
  const palettes = ['ember', 'electric', 'ultraviolet'];
  const paletteKey = 'mlx-speech-palette';
  const themeKey = 'mlx-speech-theme';
  const systemTheme = window.matchMedia('(prefers-color-scheme: dark)');
  let previous = null;
  let preference = null;
  try { previous = sessionStorage.getItem(paletteKey); } catch { /* Storage is optional. */ }
  try { preference = localStorage.getItem(themeKey); } catch { /* Follow the device. */ }
  if (!['light', 'dark'].includes(preference)) preference = null;

  const choices = palettes.filter(palette => palette !== previous);
  const palette = choices[Math.floor(Math.random() * choices.length)];
  root.dataset.palette = palette;
  try { sessionStorage.setItem(paletteKey, palette); } catch { /* Keep the current edition. */ }

  const applyTheme = () => {
    const theme = preference || (systemTheme.matches ? 'dark' : 'light');
    root.dataset.theme = theme;
    const toggle = document.getElementById('theme-toggle');
    if (toggle) {
      toggle.setAttribute('aria-pressed', String(theme === 'dark'));
      toggle.setAttribute('aria-label', 'Night mode');
      toggle.title = theme === 'dark' ? 'Turn night mode off' : 'Turn night mode on';
    }
    const backgrounds = {
      ember: { light: '#f4f3ef', dark: '#171619' },
      electric: { light: '#f0f3f8', dark: '#131925' },
      ultraviolet: { light: '#f4f0f7', dark: '#1b1622' },
    };
    document.querySelector('meta[name="theme-color"]').content = backgrounds[palette][theme];
  };
  applyTheme();
  systemTheme.addEventListener('change', () => { if (!preference) applyTheme(); });
  window.addEventListener('storage', event => {
    if (event.key !== themeKey && event.key !== null) return;
    preference = ['light', 'dark'].includes(event.newValue) ? event.newValue : null;
    applyTheme();
  });

  document.addEventListener('DOMContentLoaded', () => {
    document.querySelectorAll('[data-edition]').forEach(label => {
      label.textContent = `${palette.toUpperCase()} EDITION`;
    });
    const toggle = document.getElementById('theme-toggle');
    if (!toggle) return;
    toggle.addEventListener('click', () => {
      preference = root.dataset.theme === 'dark' ? 'light' : 'dark';
      try { localStorage.setItem(themeKey, preference); } catch { /* Still works for this visit. */ }
      applyTheme();
    });
    applyTheme();
    toggle.hidden = false;
  });
})();
