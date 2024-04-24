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

models = ["McGill-NLP/Sheared-LLaMA-1.3B-weblinx","McGill-NLP/Sheared-LLaMA-2.7B-weblinx"]
model_choice = 0
 
# Downloading validation dataset
# Load the validation split
valid = load_dataset("McGill-NLP/weblinx", split="validation")

# Download the input templates and use the LLaMA one
snapshot_download(
    "McGill-NLP/WebLINX", repo_type="dataset", allow_patterns="templates/*", local_dir="."
)

# To get the input text, simply pass a turn from the valid split to the template
turns = valid


def action_model_predict_from_data(turn):
    """
    {clean_html} - > clean html page
    {utterances} - > The user's first and last 4 utterances
    {viewport} - > View port 
    {candidates} - > the top candidates for this turn
    {action_history} -> Only the last 5 turns are provided
    """
 
    turn_text = template.format(**turn)

    ## POST METHOD to get the result of the model running on a server    
    action_str = "load(url=\"https://wikipedia.com\")"
    action_str = "say(speaker=\"navigator\", utterance=\"Hi\")"
    action_str = turn['action']
    print("action to execute: ", action_str)
    execute_action(driver, action_str, turn['candidates'])
    time.sleep(4)
        
    return   

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

driver.get("https://www.encyclopedia.com/") # Go Encyclopedia
driver.maximize_window() # Maximize the window to full screen
time.sleep(2)


# main program
for i in range(len(turns)):
    action_model_predict_from_data(turns[i])

