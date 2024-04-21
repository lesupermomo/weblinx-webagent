This part of the application is the selenium automation tool. It allows for the application to send the given request from the Chrome Extension chatbox to the model.
After recieving the output from the model, the application then fulfills the requested tasks
On top of sending the request to the model, the application also sends the Screenshot of the page, and Key information about the webpage to help the model in its predictions.

for this part of the code to run properly, make sure to do the following:

 `pip install selenium`

The webdriver that is installed must be the same instance of the chrome version that you have.
It can be installed here: https://googlechromelabs.github.io/chrome-for-testing/#stable
In our case we are using the win64 version 

To use the weblinx library and import models make sure to install 

`pip install weblinx`
`pip install huggingface_hub[cli]`

To test the the code in a specific chrome envirnoment, start chrome as such from the terminal

`chrome.exe --remote-debugging-port=9222`

Experimentation with different models. We found that we could not use the McGill-NLP/Sheared-LLaMA-2.7B-weblinx model due it necessiting 30GB during inference which is not a capacity that we were able to accomodate 
31.90 Gib needed at least for the McGill-NLP/Sheared-LLaMA-2.7B-weblinx model
32.39 GiB needed at least for the McGill-NLP/Sheared-LLaMA-1.3B-weblinx model