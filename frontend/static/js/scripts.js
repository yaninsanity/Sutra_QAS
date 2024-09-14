document.addEventListener('DOMContentLoaded', function () {
    const chatWindow = document.getElementById('chat-window');
    const userInput = document.getElementById('user-input');
    const sendBtn = document.getElementById('send-btn');
    const resetBtn = document.getElementById('reset-btn');
    const socket = io.connect(location.protocol + '//' + document.domain + ':' + location.port);

    // Function to add chat bubbles
    function addChatBubble(content, type = 'user') {
        const bubble = document.createElement('div');
        bubble.classList.add('chat-bubble', type);
        bubble.textContent = content;
        chatWindow.appendChild(bubble);
        chatWindow.scrollTop = chatWindow.scrollHeight; // Ensure the chat is scrolled to the bottom
    }

    // Event listener for the send button
    sendBtn.addEventListener('click', function () {
        const message = userInput.value.trim();
        if (message !== '') {
            // Add user message bubble
            addChatBubble(message, 'user');

            // Clear input field
            userInput.value = '';

            // Add a loading bubble while waiting for the assistant's response
            const loadingBubble = document.createElement('div');
            loadingBubble.classList.add('chat-bubble', 'loading');
            loadingBubble.textContent = '...';
            chatWindow.appendChild(loadingBubble);
            chatWindow.scrollTop = chatWindow.scrollHeight;

            // Emit the message to the backend
            socket.emit('send_message', { message: message });
        }
    });

    // Allow pressing Enter to send the message
    userInput.addEventListener('keypress', function (e) {
        if (e.key === 'Enter') {
            sendBtn.click();
        }
    });

    // Handle receiving a message from the backend
    socket.on('receive_message', function (data) {
        const loadingBubble = document.querySelector('.chat-bubble.loading');
        if (loadingBubble) {
            chatWindow.removeChild(loadingBubble);
        }

        if (data.source === 'error') {
            addChatBubble('An error occurred. Please try again.', 'assistant');
        } else {
            // Add assistant's response bubble
            addChatBubble(data.message, 'assistant');
        }
    });

    // Event listener for the reset button
    resetBtn.addEventListener('click', function () {
        socket.emit('reset_conversation');
    });

    // Handle reset confirmation from the backend
    socket.on('receive_message', function (data) {
        if (data.message === 'Conversation history reset.') {
            chatWindow.innerHTML = '';  // Clear chat window
        }
    });
});
