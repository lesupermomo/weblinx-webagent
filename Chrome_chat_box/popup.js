document.getElementById('send-btn').addEventListener('click', function() {
    const userInput = document.getElementById('user-input');
    const chatContainer = document.getElementById('chat-container');

    // Display user's message
    const userMessage = document.createElement('p');
    userMessage.textContent = userInput.value;
    chatContainer.appendChild(userMessage);

    // Simple response logic (implement your own or integrate an API)
    const botResponse = document.createElement('p');
    botResponse.textContent = "Yes, sure";
    chatContainer.appendChild(botResponse);

    // Clear input field
    userInput.value = '';
});
