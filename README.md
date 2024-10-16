# weblinx-webagent

# abstract

This project introduces a web navigation agent on Chrome using selenium and models trained on the WebLINX benchmark, establishing a new standard in conversational web navigation. Unlike previous models such as VisualWebArena, which are non-conversational and limited in scope, or closed-source models like ACT-1 which cannot be freely tested across different environments, our tool enhances accessibility and usability in web navigation. It addresses a significant gap in current web navigation technology by providing a platform that not only tests these models against real-world data in real-time environments but also demonstrates their scalability and performance effectively. Currently, the application adeptly executes user-requested actions based on the WebLINX validation dataset. However, it faces limitations with processing HTML pages via the DMR model, which we plan to overcome in future developments. Future enhancements will focus on integrating a broader range of model options, reducing application latency, and refining the user interface to ensure seamless web navigation experiences.

# Getting started 

To start using the application start by installing the necessary libraries with the following command:

`pip intall -r requirements.txt`

# Running the application

First start by navigating to the selenium folder

`cd Selenium_automation`

To directly see how our application performs the actions of the weblinx validation dataset simply run the validateData python program

`python validateData.py`

To directly see how our application prepares the data for our action model run the main file

`python main.py`

# Demo
Link to demo: https://github.com/lesupermomo/weblinx-webagent/raw/refs/heads/main/projectDemo.mp4 

![image](https://github.com/user-attachments/assets/20d9dd6b-60b2-48ae-8231-211433fa3dda)
![image](https://github.com/user-attachments/assets/4783f865-ad57-45af-b6cb-242d41fea143)


