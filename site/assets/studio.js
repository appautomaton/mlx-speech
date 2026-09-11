(() => {
  'use strict';

  const audio = document.querySelector('#demo-audio');
  const player = document.querySelector('#custom-player');
  const playButton = document.querySelector('#play-demo');
  const seek = document.querySelector('#demo-seek');
  const restart = document.querySelector('#restart-demo');
  const label = document.querySelector('#playback-label');
  const elapsed = document.querySelector('#elapsed');
  const waveProgress = document.querySelector('#wave-progress');
  const consolePanel = document.querySelector('.console');
  const audioStatus = document.querySelector('#audio-status');
  const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)');

  const formatTime = value => {
    const seconds = Math.max(0, Math.floor(value || 0));
    return `${Math.floor(seconds / 60)}:${String(seconds % 60).padStart(2, '0')}`;
  };

  if (audio && player && playButton && seek) {
    let pendingSeek = null;

    const duration = () => Number.isFinite(audio.duration) ? audio.duration : 42.933333;
    const updateProgress = () => {
      const total = duration();
      const current = Math.min(audio.currentTime || 0, total);
      seek.max = String(total);
      seek.value = String(current);
      seek.setAttribute('aria-valuetext', `${Math.round(current)} seconds of ${Math.ceil(total)} seconds`);
      elapsed.textContent = formatTime(current);
      document.querySelector('.time > span:last-child').textContent = `/ ${formatTime(Math.ceil(total))}`;
      waveProgress.setAttribute('width', String((current / total) * 600));
    };

    const updatePlayback = () => {
      const playing = !audio.paused && !audio.ended;
      consolePanel.classList.toggle('is-playing', playing);
      playButton.setAttribute('aria-label', playing ? 'Pause VibeVoice conversation' : 'Play VibeVoice conversation');
      label.textContent = playing ? 'Playing VibeVoice' : audio.ended ? 'Listen again' : audio.currentTime > 0 ? 'Resume the demo' : 'Listen to the demo';
    };

    const seekTo = time => {
      if (audio.readyState === 0) {
        pendingSeek = time;
        audio.load();
      } else {
        audio.currentTime = Math.min(time, duration());
        updateProgress();
      }
    };

    playButton.addEventListener('click', async () => {
      audioStatus.textContent = '';
      if (!audio.paused) {
        audio.pause();
        return;
      }
      try {
        if (audio.ended) audio.currentTime = 0;
        await audio.play();
      } catch (error) {
        if (error.name !== 'AbortError') {
          audioStatus.textContent = 'Playback could not start. Try the audio controls below or download the recording.';
          audio.hidden = false;
          audio.controls = true;
        }
        updatePlayback();
      }
    });

    seek.addEventListener('input', () => seekTo(Number(seek.value)));
    restart.addEventListener('click', () => seekTo(0));
    audio.addEventListener('loadedmetadata', () => {
      if (pendingSeek !== null) {
        audio.currentTime = Math.min(pendingSeek, duration());
        pendingSeek = null;
      }
      updateProgress();
    });
    audio.addEventListener('timeupdate', updateProgress);
    audio.addEventListener('seeked', () => { updateProgress(); updatePlayback(); });
    audio.addEventListener('play', updatePlayback);
    audio.addEventListener('pause', updatePlayback);
    audio.addEventListener('ended', () => { updateProgress(); updatePlayback(); });
    audio.addEventListener('error', () => {
      audioStatus.textContent = 'The recording could not load. Check your connection and try the audio controls below.';
      audio.hidden = false;
      audio.controls = true;
      updatePlayback();
    });

    audio.hidden = true;
    audio.controls = false;
    player.hidden = false;
    updateProgress();
    updatePlayback();
  }

  const filters = document.querySelector('#model-filters');
  const rows = [...document.querySelectorAll('.model-row')];
  if (filters && rows.length) {
    filters.addEventListener('click', event => {
      const selected = event.target.closest('button[data-filter]');
      if (!selected) return;
      const task = selected.dataset.filter;
      filters.querySelectorAll('button').forEach(button => {
        const active = button === selected;
        button.classList.toggle('is-active', active);
        button.setAttribute('aria-pressed', String(active));
      });
      rows.forEach(row => { row.hidden = task !== 'all' && row.dataset.task !== task; });
      const count = rows.filter(row => !row.hidden).length;
      const suffix = task === 'asr' ? 'transcription models' : task === 'tts' ? 'speech & sound variants' : 'model variants';
      document.querySelector('#model-count').textContent = `${count} ${suffix}`;
    });
    filters.hidden = false;
  }

  const tablist = document.querySelector('#code-tabs');
  if (tablist) {
    const tabs = [...tablist.querySelectorAll('[role=tab]')];
    const activate = selected => {
      tabs.forEach(tab => {
        const active = tab === selected;
        tab.setAttribute('aria-selected', String(active));
        tab.tabIndex = active ? 0 : -1;
        const panel = document.getElementById(tab.getAttribute('aria-controls'));
        panel.hidden = !active;
        panel.setAttribute('role', 'tabpanel');
        panel.setAttribute('aria-labelledby', tab.id);
      });
    };
    tabs.forEach((tab, index) => {
      tab.addEventListener('click', () => activate(tab));
      tab.addEventListener('keydown', event => {
        const indices = { ArrowRight: (index + 1) % tabs.length, ArrowLeft: (index + tabs.length - 1) % tabs.length, Home: 0, End: tabs.length - 1 };
        if (!(event.key in indices)) return;
        event.preventDefault();
        const next = tabs[indices[event.key]];
        activate(next);
        next.focus();
      });
    });
    activate(tabs[0]);
    tablist.hidden = false;
  }

  document.querySelectorAll('[data-copy]').forEach(button => {
    const target = document.getElementById(button.dataset.copy);
    if (!target) return;
    let timer;
    button.addEventListener('click', async () => {
      const buttonLabel = button.querySelector('[data-copy-label]');
      const original = button.dataset.copy === 'install-command' ? 'Copy' : 'Copy code';
      const status = document.querySelector('#copy-status');
      clearTimeout(timer);
      try {
        if (!navigator.clipboard?.writeText) throw new Error('Clipboard unavailable');
        await navigator.clipboard.writeText(target.textContent.trim());
        buttonLabel.textContent = 'Copied';
        status.textContent = button.dataset.copy === 'install-command' ? 'Installation command copied.' : 'Python example copied.';
      } catch {
        const selection = window.getSelection();
        const range = document.createRange();
        range.selectNodeContents(target);
        selection.removeAllRanges();
        selection.addRange(range);
        status.textContent = 'Automatic copying is unavailable. The text is selected so you can copy it manually.';
      }
      timer = setTimeout(() => { buttonLabel.textContent = original; }, 1800);
    });
    button.hidden = false;
  });

  // A model link can jump to the sample without starting playback.
  document.querySelectorAll('a[href="#listen"]').forEach(link => {
    link.addEventListener('click', () => {
      if (!playButton || player.hidden) return;
      window.setTimeout(() => playButton.focus({ preventScroll: true }), reducedMotion.matches ? 0 : 300);
    });
  });
})();
