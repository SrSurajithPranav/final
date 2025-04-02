// script.js

// Function to handle form submission (example)
document.addEventListener('DOMContentLoaded', function() {
    const form = document.querySelector('form');
    if (form) {
        form.addEventListener('submit', function(event) {
            // Prevent default form submission
            event.preventDefault();

            // Example: Validate form fields
            const username = document.getElementById('username').value;
            if (!username) {
                alert('Please enter a username.');
                return;
            }

            // Optionally, submit the form via AJAX
            // You can add code here to send the form data to the server
            // For example:
            // fetch('/analyze', {
            //     method: 'POST',
            //     body: new FormData(form)
            // })
            // .then(response => response.json())
            // .then(data => {
            //     // Handle the response data
            //     console.log('Success:', data);
            // })
            // .catch(error => {
            //     console.error('Error:', error);
            // });

            // For demonstration, just log the username
            console.log('Username submitted:', username);
        });
    }
});
