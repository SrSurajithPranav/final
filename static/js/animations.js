document.addEventListener('DOMContentLoaded', () => {
    // GSAP Animations
    if (typeof gsap !== 'undefined') {
        gsap.registerPlugin(TextPlugin);

        // Glitch effect on title
        const title = document.querySelector('.title');
        if (title) {
            gsap.to(title, {
                duration: 0.1,
                x: () => Math.random() * 10 - 5,
                y: () => Math.random() * 10 - 5,
                color: '#ff00ff',
                repeat: -1,
                yoyo: true,
                ease: 'none'
            });
            gsap.to(title, {
                duration: 2,
                text: "OSINT DETECTOR <span style='color:#ff00ff'>ONLINE</span>",
                ease: "none",
                repeat: -1,
                yoyo: true,
                delay: 1
            });
        }

        gsap.from('.header', { duration: 1, y: -50, opacity: 0, ease: 'power2.out' });
        gsap.from('.glass-card', { duration: 1, y: 50, opacity: 0, stagger: 0.2, ease: 'power2.out', delay: 0.5 });
        gsap.from('.button-group', { duration: 1, scale: 0, opacity: 0, ease: 'back.out(1.7)', delay: 1 });
    }

    // ScrollMagic Controller
    if (typeof ScrollMagic !== 'undefined') {
        const controller = new ScrollMagic.Controller();

        // Fade in sections on scroll
        document.querySelectorAll('.glass-card').forEach((card) => {
            new ScrollMagic.Scene({
                triggerElement: card,
                triggerHook: 0.9,
                reverse: false
            })
                .setTween(gsap.from(card, { duration: 1, y: 50, opacity: 0, ease: 'power2.out' }))
                .addTo(controller);
        });

        // Parallax effect for background video
        const backgroundVideo = document.querySelector('.background-video');
        if (backgroundVideo) {
            new ScrollMagic.Scene({
                triggerElement: 'body',
                duration: '100%'
            })
                .setTween(gsap.to(backgroundVideo, { y: 100, ease: 'none' }))
                .addTo(controller);
        }
    }
});