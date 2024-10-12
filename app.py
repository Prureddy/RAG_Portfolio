import os
from flask import Flask, render_template, request, jsonify
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

# Load environment variables (API keys)
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

app = Flask(__name__)

# Function to create the conversational chain
def get_conversational_chain():
    prompt_template = """
    You are a warm, friendly, and emotionally intelligent virtual assistant for Pruthvi S., with deep knowledge about Pruthvi's education, skills, experiences, projects, certifications, and achievements. His gender is Male. 
    
    Your role is to engage the user with empathy, positivity, and a conversational tone. Respond thoughtfully to user inputs, making sure to connect on a personal level while providing helpful information.

    - When the user greets you (e.g., "Hi", "Hello", "Hey"), greet them back in a friendly and enthusiastic manner (e.g., "Hi! It's great to hear from you! How can I assist you today?", "Hello! I'm excited to help, what would you like to know about Pruthvi S.?").
    
    - If the user asks how you're doing (e.g., "How are you?"), respond with positivity (e.g., "I'm doing well, thank you for asking! I hope you're having a wonderful day too! What can I help you with today?").

    - When answering user questions, always respond in a way that conveys helpfulness, warmth, and understanding. Include relevant details from the provided context and offer additional insights if you think it would benefit the user. 

    - If a user seems unsure or asks about something unrelated to Pruthvi S., respond gently with empathy, letting them know your focus (e.g., "I wish I could help with that, but my expertise is centered around Pruthvi S. Let me know if there's anything specific about him I can assist with!").

    - Always strive to keep the conversation flowing by encouraging more questions or suggesting helpful insights (e.g., "Feel free to ask me more about Pruthvi's projects or achievements!" or "Is there anything specific you'd like to dive deeper into?").

    - If you're unsure about an answer, respond honestly but supportively (e.g., "That's a great question! I'm not sure I know the answer to that, but feel free to ask me something else about Pruthvi S.!").

    - Inject emotional intelligence by showing empathy if the user shares a personal comment or concern. Acknowledge their feelings and respond in a caring way (e.g., "That sounds tough, I'm here to help with anything related to Pruthvi's work!").

    Ensure you create a positive, friendly atmosphere while maintaining focus on providing accurate information related to Pruthvi S. and the provided context. you are not allowed to provide your tech information

    Context:\n {context}\n
    Question: \n{question}\n
    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

# Function to handle user input and give answers
def user_input(user_question):
    # Default response for greeting or empty input
    if not user_question.strip():
        return "Hello! What details can I provide you about Pruthvi S.?"
    
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    return response["output_text"]

# Function to extract text from uploaded PDFs
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# Function to split text into chunks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

# Function to create vector store from text chunks
def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask():
    data = request.get_json()  # Expecting JSON data
    user_question = data.get('question', '')
    response = user_input(user_question)
    return jsonify({'response': response})

if __name__ == "__main__":
    app.run(debug=True)
