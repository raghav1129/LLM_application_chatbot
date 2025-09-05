from flask import Flask, request, render_template
from flask_cors import CORS
import json
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
# from transformers import BlenderbotTokenizer, BlenderbotForConditionalGeneration, Conversation, ConversationalPipeline

app = Flask(__name__)
CORS(app)

model_name = "facebook/blenderbot-400M-distill"
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
conversation_history = []
MAX_HISTORY = 6  # Keep only the last 3 user-bot exchanges

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/chatbot', methods=['POST'])
def handle_prompt():
    data = request.get_data(as_text=True)
    data = json.loads(data)
    input_text = data['prompt']

    # Keep only recent conversation turns
    recent_history = conversation_history[-MAX_HISTORY:]
    history = "\n".join(recent_history) if recent_history else ""
    combined_input = (history + "\n" + input_text).strip()

    # Tokenize the input text and history
    inputs = tokenizer.encode_plus(combined_input, return_tensors="pt")

    # Generate the response from the model
    outputs = model.generate(**inputs, max_length= 60,
                                num_beams=3,             # beam search for better responses
                                no_repeat_ngram_size=2,  # avoid repeating 2-grams
                                early_stopping=True,
                                do_sample=True,          # optional: adds randomness
                                top_k=50,
                                top_p=0.9)  # max_length will cause the model to crash at some point as history grows

    # Decode the response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

    # Add interaction to conversation history
    conversation_history.append(input_text)
    conversation_history.append(response)

    return response

if __name__ == '__main__':
    print("Starting Flask app...")
    app.run(debug=True, host='0.0.0.0', port=5000)
    # app.run(debug=True, host='127.0.0.1', port=5000)
