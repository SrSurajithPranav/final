document.addEventListener('DOMContentLoaded', () => {
    if (typeof Chart === 'undefined') return;

    // Feature Chart
    const featureChartCtx = document.getElementById('featureChart');
    if (featureChartCtx && window.result) {
        const features = window.result.features;
        new Chart(featureChartCtx, {
            type: 'bar',
            data: {
                labels: Object.keys(features),
                datasets: [{
                    label: 'Feature Values',
                    data: Object.values(features),
                    backgroundColor: 'rgba(0, 255, 234, 0.6)',
                    borderColor: '#00ffea',
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                scales: {
                    y: { beginAtZero: true }
                }
            }
        });
    }

    // Behavior Chart
    const behaviorChartCtx = document.getElementById('behaviorChart');
    if (behaviorChartCtx && window.result && window.result.behavior) {
        const behavior = window.result.behavior;
        new Chart(behaviorChartCtx, {
            type: 'line',
            data: {
                labels: behavior.timestamps,
                datasets: [{
                    label: 'Activity Over Time',
                    data: behavior.activity,
                    borderColor: '#ff00ff',
                    backgroundColor: 'rgba(255, 0, 255, 0.2)',
                    fill: true
                }]
            },
            options: {
                responsive: true,
                scales: {
                    y: { beginAtZero: true }
                }
            }
        });
    }

    // Sentiment History Chart
    const sentimentChartCtx = document.getElementById('sentimentChart');
    if (sentimentChartCtx && window.result && window.result.sentiment_history) {
        const sentiment = window.result.sentiment_history;
        new Chart(sentimentChartCtx, {
            type: 'line',
            data: {
                labels: sentiment.timestamps,
                datasets: [{
                    label: 'Sentiment Over Time',
                    data: sentiment.values,
                    borderColor: '#00ffea',
                    backgroundColor: 'rgba(0, 255, 234, 0.2)',
                    fill: true
                }]
            },
            options: {
                responsive: true,
                scales: {
                    y: { beginAtZero: true, min: -1, max: 1 }
                }
            }
        });
    }
});