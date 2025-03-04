# 🔮 Next Word Prediction Using LSTM

## 📖 Project Overview
This project develops an **AI-powered Next Word Prediction model** using **Long Short-Term Memory (LSTM)** networks. The model is trained on Shakespeare's **Hamlet** and predicts the next word based on a given sequence. The application is deployed using **Streamlit**, allowing users to interact with the model in real-time.

---

## 🎯 Objectives
✔️ Train an **LSTM model** for next-word prediction.  
✔️ Implement **text preprocessing** (tokenization & padding).  
✔️ Utilize **early stopping** to prevent overfitting.  
✔️ Save the trained model (`.h5` file) and tokenizer (`.pickle` file).  
✔️ Deploy an **interactive Streamlit web app** for real-time predictions.  

---

## 🛠️ Technologies Used
- **Python** – Programming language  
- **TensorFlow/Keras** – Deep learning framework  
- **Streamlit** – Web application framework  
- **NumPy** – Numerical computing  
- **Pickle** – Model and tokenizer serialization  
- **Natural Language Processing (NLP)** – Text processing  

---

## 📂 Dataset Used
The dataset used is **Shakespeare's "Hamlet"**, which provides **rich and complex text** for training the model. The text data is **tokenized, sequenced, and padded** to maintain uniform input sizes.

---

## 🔎 Project Breakdown

### **1️⃣ Data Preprocessing**
- **Tokenization** – Convert text into numerical sequences.  
- **Padding** – Ensures all sequences are of the same length.  
- **Splitting** – Training and testing sets are created.  

### **2️⃣ LSTM Model Architecture**
The model consists of:
- **Embedding Layer** – Converts words into dense vectors.
- **Two LSTM Layers** – Handles long-term dependencies.
- **Dense Output Layer (Softmax Activation)** – Predicts the next word.

### **3️⃣ Model Training**
- **Early Stopping** prevents overfitting by monitoring validation loss.  
- Model is trained using **Categorical Cross-Entropy Loss** and **Adam Optimizer**.

### **4️⃣ Model Evaluation**
- Tested on sample text sequences.
- Predicts the next word **with high accuracy**.

### **5️⃣ Deployment with Streamlit**
- Users enter a **sequence of words**, and the model predicts the **next word**.
- The UI is **simple, interactive, and real-time**.

---

## 🛠️ Installation Guide

### **1️⃣ Clone the Repository**
```bash
git clone https://github.com/your-username/next-word-prediction.git
cd next-word-prediction
2️⃣ Install Dependencies
bash
Copy
Edit
pip install -r requirements.txt
3️⃣ Ensure Model & Tokenizer Files Are Available
Make sure the following files are present in the project directory:

next_word_lstm.h5 → Pre-trained LSTM model
tokenizer.pickle → Saved tokenizer file
If not, train the model again before proceeding.

🚀 How to Run the Streamlit App
bash
Copy
Edit
streamlit run app.py
This will launch a web interface where users can input a phrase and receive AI-generated next-word predictions.

📦 Files & Directories
File/Directory	Description
app.py	Streamlit web app script
next_word_lstm.h5	Trained LSTM model
tokenizer.pickle	Saved tokenizer for word sequences
requirements.txt	Required Python dependencies
📚 Example Usage
User Input:
plaintext
Copy
Edit
To be or not to
Model Prediction:
plaintext
Copy
Edit
Next word: "be"
🧩 Customization & Improvements
🔹 Train on a larger dataset to improve predictions.
🔹 Implement Bidirectional LSTM for better text generation.
🔹 Use attention mechanisms to enhance accuracy.
🔹 Fine-tune hyperparameters for optimized performance.

🛡️ Best Practices & Notes
✔️ Ensure TensorFlow/Keras is installed for model inference.
✔️ Avoid excessive training—use early stopping to prevent overfitting.
✔️ Make sure .h5 and .pickle files exist before running the app.
✔️ Optimize tokenization & preprocessing for better predictions.

