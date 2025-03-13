document.addEventListener('DOMContentLoaded', () => {
    const gameArea = document.getElementById('game-area');
    const scoreDisplay = document.getElementById('score');
    let score = 0;
    let level = 1;
    let botsPerLevel = 3;

    if (!gameArea) return;

    function createBot() {
        const bot = document.createElement('div');
        bot.classList.add('bot');
        bot.style.width = '30px';
        bot.style.height = '30px';
        bot.style.background = '#ff4d4d';
        bot.style.position = 'absolute';
        bot.style.borderRadius = '50%';
        bot.style.left = `${Math.random() * (gameArea.clientWidth - 30)}px`;
        bot.style.top = `${Math.random() * (gameArea.clientHeight - 30)}px`;

        bot.addEventListener('click', () => {
            score++;
            scoreDisplay.textContent = score;
            bot.remove();
            if (score % 10 === 0) {
                level++;
                botsPerLevel++;
                alert(`Level Up! Level: ${level}`);
                for (let i = 0; i < botsPerLevel; i++) {
                    createBot();
                }
            } else {
                createBot();
            }

            // Save score
            fetch('/save_score', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ score: score })
            })
            .then(response => response.json())
            .then(data => console.log(data.message))
            .catch(error => console.error('Error saving score:', error));
        });

        gameArea.appendChild(bot);

        gsap.to(bot, {
            x: Math.random() * (gameArea.clientWidth - 30),
            y: Math.random() * (gameArea.clientHeight - 30),
            duration: 2,
            repeat: -1,
            yoyo: true,
            ease: 'power1.inOut'
        });
    }

    gameArea.style.width = '300px';
    gameArea.style.height = '200px';
    gameArea.style.background = 'rgba(0, 0, 0, 0.5)';
    gameArea.style.position = 'relative';
    gameArea.style.border = '1px solid #00ffea';

    for (let i = 0; i < botsPerLevel; i++) {
        createBot();
    }

    // Show the game
    gameArea.classList.remove('hidden');
});