from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from datasets import load_dataset
from huggingface_hub import snapshot_download
from transformers import pipeline
import time
import requests
import torch
from pathlib import Path
import weblinx as wl
import os
import re
from bs4 import BeautifulSoup

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = ['cuda','cpu']
device_choice = 1
device = torch.device(device[1])

def send_prediction_request(turn_text):
    response = requests.post('http://localhost:5000/predict', json={'turn_next': turn_text})
    if response.status_code == 200:
        return response.json()['prediction']
    else:
        print("Failed to get prediction:", response.json())
        return None

# Function to clean HTML using BeautifulSoup
def clean_html_page(raw_html):
    soup = BeautifulSoup(raw_html, "html.parser")
    
    # Remove scripts and style elements
    for script in soup(["script", "style"]):
        script.decompose()
    clean_text = soup.get_text()
    return clean_text

## Initializing variables
action_history = []   
utterances = [] 

"""
user_request: contains the utterance request of the user, if None, then the model is still predicting next actions
action_history: this will contain raw action_history but with no "</s><s>[INST]" in the end of the last instruction, but </s><s>[INST] in the beginnning of instructor instructions
utterances: this will contain the processed utterances using the time for example [00:00] hi
"""
def action_model_predict(user_request, action_history, utterances):
    """
    {clean_html} - > clean html page
    {utterances} - > The user's first and last 4 utterances
    {viewport} - > View port 
    {candidates} - > the top candidates for this turn
    {action_history} -> Only the last 5 turns are provided
    """
 
    action_history_element = ""
    if user_request == None:
        # then the model should continue predicting and </s><s>[INST] should be temporarly be put at the end of the action history
           
        for action in action_history[-5:]:
            if action.startswith("<"):
                # Don't add a space if the action was a speaker instruction
                action_history_element += action

            else:
                action_history_element += " "+action
         
        action_history_element += f"</s><s>[INST]"
    
    else:
        # then you should add the user_request to the action history with </s><s>[INST] like: </s><s>[INST] say(speaker="instructor", utterance="Open the one which has the highest reviews.")
        instruction = f"</s><s>[INST] say(speaker=\"instructor\", utterance=\"{user_request}\")"
        action_history.append(instruction)
        action_history_element = " ".join(action_history[-5:])
        # Add the new user request to the action_history and action_history_element
                
    # Append the new last user request
    if user_request is not None:
        elapsed_time = time.strftime("[%M:%S]", time.gmtime(time.time() - start_time))
        utterances.append(f"{elapsed_time} {user_request}")
    
    utterances_element = ""
    if len(utterances) > 5:
        utterances_element = " ".join([utterances[0]]+utterances[-4:])
    else:
        utterances_element = " ".join(utterances)
     
    # Retrieving clean html page
    raw_html = driver.page_source
    clean_html = clean_html_page(raw_html)
    clean_html = ""

    # 746h x 1536w
    viewport = driver.execute_script("return [window.innerWidth, window.innerHeight];")
    width , height = viewport[0], viewport[1]
    viewport = f"{height}h x {width}w"

    # Use DMR model to get candidates for prediction
    candidates = ""

    turn = {"clean_html": clean_html, "utterances": utterances_element, "viewport": viewport, "candidates": candidates, "action_history": action_history_element}
    
    turn_text = template.format(**turn)
    print("turn_next is:------------------------------------------------------------------------------------------------------\n",turn_text)
    print("------------------------------------------------------------------------------------------------------")


    ## POST METHOD to get the result of the model running on a server
    if user_request !=  None and user_request.startswith("load"):
        site = user_request.split(" ")[1]
        action_str = f"load(url=\"https://{site}.com\")"
    else:
        action_str = "say(speaker=\"navigator\", utterance=\"Hi\")" if len(utterances) == 0 else "say(speaker=\"navigator\", utterance=\"Done.\")"
    # action_str = send_prediction_request(turn_text)
    
    # Append the new last action
    execute_action(driver, action_str, candidates)
    action_history.append(action_str)
    
    # for the action history, always put a temporary </s><s>[INST] in the end, unless the action is say(speaker="instructor"), then the </s><s>[INST] is in the beginning, for example: </s><s>[INST] say(speaker="instructor", utterance="Open the one which has the highest reviews.")
    ## whenever the action is a say(speaker="navigator") , then you must wait for the user for a new input instead of continuing in this loop
    if action_str[0:3] == "say":
        return 
    else:
        return action_model_predict(None, action_history, utterances)
    

def parse_action_string(action_str):
    # Extracting the main command and its arguments
    command_match = re.match(r"(\w+)\((.*)\)", action_str)
    if not command_match:
        return None
    
    command_type = command_match.group(1)
    args_str = command_match.group(2)
    
    # Parsing the arguments into a dictionary
    args = {}
    for arg in re.findall(r"(\w+)=\"(.*?)\"", args_str):
        key, value = arg
        # Attempt to convert numerical values appropriately
        if value.isdigit():
            value = int(value)
        args[key] = value
    
    # print("parsed action: ",command_type," ",args)
    return command_type, args

retrieve_element = """
    var x = arguments[0], y = arguments[1], width = arguments[2], height = arguments[3];
    var elements = document.elementsFromPoint(x + width/2, y + height/2);
    if (elements.length > 0) return elements[0];
    return null;
    """
retrieve_form = """
        var x = arguments[0], y = arguments[1], width = arguments[2], height = arguments[3];
        var elements = document.elementsFromPoint(x + width/2, y + height/2);
        for (var i = 0; i < elements.length; i++) {
            if (elements[i].tagName.toLowerCase() === 'form') {
                elements[i].submit();
                return true;
            }
        }
        return false;
        """

def retrieve_from_candidates(uid, candidates):
    # Regex to match the pattern containing the uid and extract the entire line along with bbox values
    candidates += "\n" # helps with pattern matching
    print(candidates)
    pattern = r'(\(uid = {}\) .*?\[\[bbox\]\] x=(.*?) y=(.*?) width=(.*?) height=(.*?) \[\[.*?\n)'.format(uid)
    
    # Search for the pattern in the candidates string
    match = re.search(pattern, candidates)
    
    if match:
        # Extract the entire line and the x, y, width, and height from the match
        entire_line = match.group(1)
        x = float(match.group(2))
        y = float(match.group(3))
        width = float(match.group(4))
        height = float(match.group(5))
        
        # Print the entire line for debugging or information
        print("Matched Data:", entire_line.strip())
        
        # Return x, y, width, and height
        return x, y, width, height
    else:
        # Print a message or raise an error if the uid is not found
        print("UID not found in candidates.")
        return None


def handle_change(driver, candidates, value, uid):
    bbox = retrieve_from_candidates(uid,candidates)
    if bbox is None: 
        return 
    else:
        x, y, width, height = bbox
    
    # Execute the script and get the element
    element = driver.execute_script(retrieve_element, x, y, width, height)
    element.clear()
    element.send_keys(value)

def handle_click(driver, candidates, uid):
    bbox = retrieve_from_candidates(uid,candidates)
    if bbox is None: 
        return 
    else:
        x, y, width, height = bbox
    # Execute the script and get the element
    element = driver.execute_script(retrieve_element, x, y, width, height)
    element.click()

def handle_load(driver, url):
    driver.get(url)

def handle_say(utterance):
    print(utterance)

def handle_scroll(driver, x, y):
    driver.execute_script(f"window.scrollBy({x}, {y})")

def handle_submit(driver, candidates, uid):
    bbox = retrieve_from_candidates(uid,candidates)
    if bbox is None: 
        return 
    else:
        x, y, width, height = bbox
    # Execute the script and get the element
    element = driver.execute_script(retrieve_form, x, y, width, height)
    element.submit()

def handle_text_input(driver, candidates, text, uid):
    bbox = retrieve_from_candidates(uid,candidates)
    if bbox is None: 
        return 
    else:
        x, y, width, height = bbox
    # Execute the script and get the element
    element = driver.execute_script(retrieve_element, x, y, width, height)
    element.send_keys(text)

def execute_action(driver, action_str, candidates):
    command_type, args = parse_action_string(action_str)
    
    if command_type == "change":
        handle_change(driver, **args)
    elif command_type == "click":
        handle_click(driver, candidates, **args)
    elif command_type == "load":
        handle_load(driver, **args )
    elif command_type == "say":
        handle_say(args["utterance"])
    elif command_type == "scroll":
        handle_scroll(driver, **args)
    elif command_type == "submit":
        # handle_submit(driver, candidates, **args)
        handle_click(driver, candidates, **args)
    elif command_type == "text_input":
        handle_text_input(driver, candidates, **args)

def take_screenshot(driver, path="./image.png"):
    # Returns and base64 encoded string into image
    driver.save_screenshot(path)
    return


# opening template file
with open('templates/llama.txt') as f:
        template = f.read()

#### Automation tool section
chrome_options = Options()
chrome_options.add_experimental_option("debuggerAddress", "127.0.0.1:9222")

service = Service(executable_path="chromedriver.exe") # Path to the chromedriver.exe
# driver = webdriver.Chrome(service=service, options=chrome_options) # the driver is what allows use to the chrome as an agent
driver = webdriver.Chrome(service=service) # the driver is what allows use to the chrome as an agent

# Go to the google page
driver.get("https://google.com")

start_time = time.time()
user_request = None

while True: 
    # action_str = str(input("Write an action to perform in the correct format:  \nchange(value=[str], uid=[str]) ; click(uid=[str]) ; load(url=[str]) ; say(speaker=\"navigator\", utterance=[str]) ; scroll(x=[int], y=[int]) ; submit(uid=[str]) ; text_input(text=[str], uid=[str])):\n"))
    # start the program with an empty action_history and utterances (the model will start by saying hello, then expect an input)
    action_model_predict(user_request, action_history, utterances)
    user_request = str(input("Input: "))
    print()
