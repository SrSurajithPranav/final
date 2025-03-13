document.addEventListener('DOMContentLoaded', () => {
    const assistant = document.getElementById('assistant');
    if (assistant) {
        assistant.classList.remove('hidden');

        // Simulate assistant tips
        const tips = [
            'Tip: Use the voice search to quickly analyze a target!',
            'Tip: Provide feedback on results to improve the model!',
            'Tip: Check the live dashboard for real-time bot activity!'
        ];
        let tipIndex = 0;

        setInterval(() => {
            assistant.innerHTML = `<p>OSINT Cyber Agent: ${tips[tipIndex]}</p>`;
            tipIndex = (tipIndex + 1) % tips.length;
        }, 10000);
    }
});