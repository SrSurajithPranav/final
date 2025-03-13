document.addEventListener('DOMContentLoaded', () => {
    // Theme Switcher
    const themeSwitch = document.getElementById('themeSwitch');
    if (themeSwitch) {
        themeSwitch.addEventListener('click', () => {
            document.body.classList.toggle('cyberpunk-theme');
            const themeStyle = document.getElementById('theme-style');
            themeStyle.href = document.body.classList.contains('cyberpunk-theme')
                ? '/static/css/cyberpunk.css'
                : '/static/css/theme.css';
        });
    }

    // Mode Switcher
    const modeSwitch = document.getElementById('modeSwitch');
    if (modeSwitch) {
        modeSwitch.addEventListener('click', () => {
            document.body.classList.toggle('light-mode');
            const modeStyle = document.getElementById('mode-style');
            modeStyle.href = document.body.classList.contains('light-mode')
                ? '/static/css/light.css'
                : '/static/css/theme.css';
        });
    }

    // Language Switcher
    const languageSwitch = document.getElementById('languageSwitch');
    if (languageSwitch) {
        languageSwitch.addEventListener('change', function() {
            document.cookie = `locale=${this.value}; path=/`;
            location.reload();
        });
    }

    // Analyze Static User
    const analyzeStaticButton = document.getElementById('analyzeStatic');
    if (analyzeStaticButton) {
        analyzeStaticButton.addEventListener('click', () => {
            const username = document.getElementById('staticUserSelect').value;
            if (username) {
                const form = document.createElement('form');
                form.method = 'POST';
                form.action = '/analyze';
                const input = document.createElement('input');
                input.type = 'hidden';
                input.name = 'username';
                input.value = username;
                form.appendChild(input);
                document.body.appendChild(form);
                form.submit();
            }
        });
    }

    // Loader Display
    const analyzeForm = document.getElementById('analyzeForm');
    if (analyzeForm) {
        analyzeForm.addEventListener('submit', () => {
            document.getElementById('loader').classList.remove('hidden');
        });
    }

    // Initialize Vanilla Tilt for cards
    if (typeof VanillaTilt !== 'undefined') {
        VanillaTilt.init(document.querySelectorAll('.tilt-card'), {
            max: 15,
            speed: 400,
            glare: true,
            'max-glare': 0.5
        });
    }

    // Dynamic Background
    const hour = new Date().getHours();
    const videoSource = document.querySelector('.background-video source');
    if (videoSource) {
        videoSource.src = hour >= 18 || hour < 6
            ? '/static/assets/rainy-night.mp4'
            : '/static/assets/neon-day.mp4';
        videoSource.parentElement.load();
    }
});