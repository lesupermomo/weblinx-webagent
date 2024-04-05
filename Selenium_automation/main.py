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
import torch
from pathlib import Path
import weblinx as wl
import os
import re

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print(device)


downloadValidation = False
downloadDataset = False
startService = True


# Downloading validation dataset
if downloadValidation:

    # Load the validation split
    valid = load_dataset("McGill-NLP/weblinx", split="validation")

    # Download the input templates and use the LLaMA one
    snapshot_download(
        "McGill-NLP/WebLINX", repo_type="dataset", allow_patterns="templates/*", local_dir="."
    )
    with open('templates/llama.txt') as f:
        template = f.read()

    # To get the input text, simply pass a turn from the valid split to the template
    turn = valid[0]
    turn_text = template.format(**turn)

    action_model = pipeline(
        model="McGill-NLP/Sheared-LLaMA-2.7B-weblinx", device=device, torch_dtype='auto'
    )

    out = action_model(turn_text, return_full_text=False, max_new_tokens=64, truncation=True)
    pred = out[0]['generated_text'].strip()

    print("Ref:", turn["action"])
    print("Pred:", pred)
        
    


if downloadDataset:
    # Downloading a couple of demos
    demo_names = ['ajjgtji']  # random demo from valid
    patterns = [f"demonstrations/{name}/*" for name in demo_names]
    snapshot_download(
        repo_id="McGill-NLP/WebLINX-full", repo_type="dataset", local_dir="./wl_data", allow_patterns=patterns
    )

    wl_dir = Path("./wl_data")
    base_dir = wl_dir / "demonstrations"
    split_path = wl_dir / "splits.json"

    demo_names = ['ajjgtji']  # random demo from valid

    # Load the demonstrations
    demos = [wl.Demonstration(name, base_dir=base_dir) for name in demo_names]

    # Select a demo to work with
    demo = demos[0]

    # Load the Replay object, which contains the turns of the demonstration
    replay = wl.Replay.from_demonstration(demo)

    # Filter the turns to keep only the ones that are relevant for the task
    turns = replay.filter_by_intents(
        "click", "textInput", "load", "say", "submit"
    )
    # print("all turns are:",turns)

    turn = turns[0]
    # print("turn 0 is:",turn)
    # print("HTML sneak peak:", turn.html[:75])
    # print("Random Bounding Box:", turn.bboxes['bc7dcf18-542d-48e6'])

print()

def action_model_predict(action_str):
    # place holder for now
    print("Yes, sure")
    time.sleep(3)
    return "load(url=\"https://wikipedia.com\")"

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

def handle_change(driver, value, uid):
    element = driver.find_element(By.ID, uid)
    element.clear()
    element.send_keys(value)

def handle_click(driver, uid):
    element = driver.find_element(By.ID, uid)
    element.click()

def handle_load(driver, url):
    driver.get(url)

def handle_say(utterance):
    print(utterance)

def handle_scroll(driver, x, y):
    driver.execute_script(f"window.scrollBy({x}, {y})")

def handle_submit(driver, uid):
    element = driver.find_element(By.ID, uid)
    element.submit()

def handle_text_input(driver, text, uid):
    element = driver.find_element(By.ID, uid)
    element.send_keys(text)

def execute_action(driver, action_str):
    command_type, args = parse_action_string(action_str)
    
    if command_type == "change":
        handle_change(driver, **args)
    elif command_type == "click":
        handle_click(driver, **args)
    elif command_type == "load":
        handle_load(driver, **args)
    elif command_type == "say":
        handle_say(**args)
    elif command_type == "scroll":
        handle_scroll(driver, **args)
    elif command_type == "submit":
        handle_submit(driver, **args)
    elif command_type == "text_input":
        handle_text_input(driver, **args)

if startService:
    #### Automation tool section

    chrome_options = Options()
    chrome_options.add_experimental_option("debuggerAddress", "127.0.0.1:9222")

    service = Service(executable_path="chromedriver.exe") # Path to the chromedriver.exe
    # driver = webdriver.Chrome(service=service, options=chrome_options) # the driver is what allows use to the chrome as an agent
    driver = webdriver.Chrome(service=service) # the driver is what allows use to the chrome as an agent

    # Go to the google page
    driver.get("https://google.com")

    while True: 
        # action_str = str(input("Write an action to perform in the correct format:  \nchange(value=[str], uid=[str]) ; click(uid=[str]) ; load(url=[str]) ; say(speaker=\"navigator\", utterance=[str]) ; scroll(x=[int], y=[int]) ; submit(uid=[str]) ; text_input(text=[str], uid=[str])):\n"))
        action_str = str(input("Input: "))
        action_str= action_model_predict(action_str)
        print("model prediction: ",action_str) # example action_str: load(url="https://wikipedia.com")
        execute_action(driver, action_str)
        print()
    
    # ################## Input to the model
    # {clean_html} - > clean html page
    # {utterances} - > The user's first and last 4 utterances
    # {viewport} - > View port 
    # {candidates} - > the top candidates for this turn
    # {action_history} -> Only the last 5 turns are provided

    # ################## Output of the model
    # # change(value=[str], uid=[str]) ; click(uid=[str]) ; load(url=[str]) ; say(speaker="navigator", utterance=[str]) ; scroll(x=[int], y=[int]) ; submit(uid=[str]) ; text_input(text=[str], uid=[str])

    # # if it exists then get the element, clear the input in it and type something and click enter

    # # try to find the google search bar by its ID
    # WebDriverWait(driver, 5).until(
    #     EC.presence_of_element_located((By.CLASS_NAME, "gLFyf"))
    # )

    # input_element = driver.find_element(By.CLASS_NAME, "gLFyf")
    # input_element.clear()

    # websiteToGo = str(input("Which website would you like to navigate to?"))
    # input_element.send_keys(websiteToGo + Keys.ENTER)

    # # Look if there is a precense of a partial_link_text containing the word reddit
    # WebDriverWait(driver, 5).until(
    #     EC.presence_of_element_located((By.PARTIAL_LINK_TEXT, websiteToGo))
    # )

    # # Click on the first link containing the word reddit
    # link = driver.find_element(By.PARTIAL_LINK_TEXT, websiteToGo)
    # link.click()

    # time.sleep(10)

    # driver.quit()