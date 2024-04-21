import os
from flask import Flask, request, jsonify
from transformers import pipeline
import torch

# Set CUDA device
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = ['cuda', 'cpu']
device_choice = 1
device = torch.device(device[device_choice])

# Model selection
models = ["McGill-NLP/Sheared-LLaMA-1.3B-weblinx", "McGill-NLP/Sheared-LLaMA-2.7B-weblinx"]
model_choice = 0

# Load model
action_model = pipeline(
    model=models[model_choice], device=device, torch_dtype='auto'
)

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    turn_next = data.get('turn_next')
    if not turn_next:
        return jsonify({'error': 'No input provided'}), 400
    try:
        prediction = action_model(turn_next, return_full_text=False, max_new_tokens=64, truncation=True)[0]['generated_text'].strip()
        return jsonify({'prediction': prediction})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
