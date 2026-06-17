/**
 * AcneVision Theme System
 * Location: static/js/theme.js
 */

// Apply immediately to prevent flash
(function() {
  const t = localStorage.getItem('av-theme') || 'dark';
  document.body.classList.add('theme-' + t);
})();

function toggleTheme() {
  const isDark = document.body.classList.contains('theme-dark');
  const next   = isDark ? 'light' : 'dark';
  document.body.classList.remove('theme-dark', 'theme-light');
  document.body.classList.add('theme-' + next);
  localStorage.setItem('av-theme', next);
  _updateIcons(next);
}

function _updateIcons(theme) {
  document.querySelectorAll('.theme-toggle-btn').forEach(btn => {
    btn.textContent = theme === 'dark' ? '☀️' : '🌙';
  });
}

document.addEventListener('DOMContentLoaded', () => {
  const saved = localStorage.getItem('av-theme') || 'dark';
  document.body.classList.remove('theme-dark', 'theme-light');
  document.body.classList.add('theme-' + saved);
  _updateIcons(saved);
});