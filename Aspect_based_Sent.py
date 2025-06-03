from flask import Flask, render_template, request, redirect, url_for, session
from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification
from googletrans import Translator
import requests

# Initialize Flask App
app = Flask(__name__)

# Initialize session secret key
app.secret_key = "your_secret_key"  # Replace this with a more secure key in production

# Initialize Translator
translator = Translator()

# Temporary in-memory user storage
users_db = {}

# Load ABSA Model
absa_model_name = "yangheng/deberta-v3-base-absa-v1.1"
tokenizer_absa = AutoTokenizer.from_pretrained(absa_model_name, use_fast=False)
model_absa = AutoModelForTokenClassification.from_pretrained(absa_model_name)
absa_pipeline = pipeline(
    "token-classification",
    model=model_absa,
    tokenizer=tokenizer_absa,
    aggregation_strategy="simple"
)

# Load Sentiment Model
sentiment_pipeline = pipeline(
    "sentiment-analysis",
    model="nlptown/bert-base-multilingual-uncased-sentiment"
)

# Call Ollama API with Prompt Size Limitation
def query_ollama(prompt, model="tinyllama", max_chars=1000):
    try:
        # Limit prompt size
        if len(prompt) > max_chars:
            prompt = prompt[:max_chars] + "\n\n[Note: Truncated for processing]"

        response = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": model, "prompt": prompt, "stream": False},
            timeout=60
        )

        if response.status_code != 200:
            return f"[Ollama Error] HTTP {response.status_code} - {response.text}"

        result = response.json()
        return result.get("response", "No response from Ollama.")

    except requests.exceptions.RequestException as e:
        return f"[Ollama Error] {str(e)}"
    except Exception as e:
        return f"[Unexpected Ollama Error] {str(e)}"

# Main Analysis Function
def analyze_aspects(input_text):
    if not input_text:
        return {"error": "Please enter some text."}

    try:
        # Translate input synchronously
        translated = translator.translate(input_text, dest='en')  # Synchronous
        translated_text = translated.text
        detected_lang = translated.src

        # ABSA
        absa_results = absa_pipeline(translated_text)
        absa_output = f"[Language: {detected_lang}]\n[Translated: {translated_text}]\n\n"
        if not absa_results:
            absa_output += "No aspects found.\n"
        else:
            for res in absa_results:
                absa_output += (
                    f"Aspect: {res['word']}\n"
                    f"Sentiment: {res['entity_group']}\n"
                    f"Confidence: {round(res['score'] * 100, 2)}%\n\n"
                )

        # Document Sentiment (NBA)
        sentiment_result = sentiment_pipeline(translated_text)[0]
        sentiment_output = (
            f"[Language: {detected_lang}]\n[Translated: {translated_text}]\n\n"
            f"Overall Sentiment: {sentiment_result['label']}\n"
            f"Confidence: {round(sentiment_result['score'] * 100, 2)}%"
        )

        # Ollama (tinyllama model)
        if len(translated_text.strip()) < 10:
            ollama_result = "[Ollama Skipped] Input too short for meaningful analysis."
        else:
            ollama_prompt = f"Please analyze the following text for sentiment and insights:\n\n{translated_text}"
            ollama_result = query_ollama(ollama_prompt, model="tinyllama")

        ollama_output = f"[Ollama LLM Response]\n\n{ollama_result}"

        return {
            "absa_output": absa_output,
            "sentiment_output": sentiment_output,
            "ollama_output": ollama_output
        }

    except Exception as e:
        return {"error": f"An error occurred: {str(e)}"}

# Flask Route for Login Page
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")

        # Check if the user exists and password matches
        if username in users_db and users_db[username] == password:
            session['user'] = username  # Store user in session
            return redirect(url_for('index'))
        else:
            return render_template("login.html", error="Invalid credentials. Please try again.")
    
    return render_template("login.html")

# Flask Route for Registration Page
@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")
        confirm_password = request.form.get("confirm_password")

        if password == confirm_password:
            # Store the new user credentials in the in-memory database
            if username in users_db:
                return render_template("register.html", error="Username already exists.")
            users_db[username] = password
            return redirect(url_for('login'))
        else:
            return render_template("register.html", error="Passwords do not match. Please try again.")
    
    return render_template("register.html")

# Flask Route for Home Page
@app.route("/", methods=["GET", "POST"])
def index():
    if 'user' not in session:
        return redirect(url_for('login'))

    if request.method == "POST":
        input_text = request.form.get("input_text")
        results = analyze_aspects(input_text)
        if "error" in results:
            return render_template("index.html", error=results["error"])

        return render_template("index.html", 
                               absa_output=results["absa_output"],
                               sentiment_output=results["sentiment_output"],
                               ollama_output=results["ollama_output"])

    return render_template("index.html")

# Run the Flask App
if __name__ == "__main__":
    app.run(debug=True)
