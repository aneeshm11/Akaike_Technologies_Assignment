import streamlit as st
import json
import os
from utils import get_output, save_vector_store, text_to_speech, play_mp3, chat_questions
import datetime
import pytz
import base64

# folder to temporarily store the files obtained from RAG (vectorstores) , mp3 file and the json of the company report 

OUTPUT_DIR = "data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# session states for navigating between the pages
if "page" not in st.session_state:
    st.session_state.page = "input"
if "output" not in st.session_state:
    st.session_state.output = None
if "corpus" not in st.session_state:
    st.session_state.corpus = ""
if "audio_path" not in st.session_state:
    st.session_state.audio_path = ""
if "json_path" not in st.session_state:
    st.session_state.json_path = os.path.join(OUTPUT_DIR, "output.json")

def reset_app():
    st.session_state.page = "input"
    st.session_state.output = None
    st.session_state.corpus = ""
    st.session_state.audio_path = ""



def get_time_based_content():
    # I made a small feature of putting greeting message and background based on it. 
    # I hardcoded the function to work on Indian time as the servers on huggingface spaces are hosted elsewhere. I put this to avoid confusion 
    ist = pytz.timezone('Asia/Kolkata')
    now = datetime.datetime.now(ist)
    hour = now.hour
    
    if 4 <= hour < 12:
        return "Good Morning!", "morning.jpeg"
    elif 12 <= hour < 16:
        return "Good Afternoon", "afternoon.jpeg"
    else:
        return "Good Evening", "evening.jpeg"

def set_background(image_file):
    try:
        with open(image_file, "rb") as f:
            data = f.read()
        
        st.markdown(
            f"""
            <style>
            .stApp {{
                background-image: url("data:image/jpeg;base64,{base64.b64encode(data).decode()}");
                background-size: cover;
                background-position: center;
                background-repeat: no-repeat;
                background-attachment: fixed;
            }}
            </style>
            """,
            unsafe_allow_html=True
        )
    except Exception as e:
        st.warning(f"Could not load background image: {e}")

greeting, bg_image = get_time_based_content()

try:
    set_background(bg_image)
except Exception:
    pass

# using css to syle the buttons, div and other containers

st.markdown("""
<style>

    /* main containers */

    .container {
        padding: 20px;
        margin-bottom: 20px;
    }
    
    /* for greeting */


    .greeting {
        font-size: 28px;
        font-weight: bold;
        color: white;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.6);
        padding: 10px;
        /* Removed background color and border to remove the box */
        margin-bottom: 20px;
        text-align: center;
    }
    
    /* header components */


    .main-header {
        color: #1E3A8A;
        font-size: 32px;
        font-weight: bold;
        margin-bottom: 20px;
        border-bottom: 2px solid #1E3A8A;
        padding-bottom: 10px;
    }
    
    .section-header {
        color: #2563EB;
        font-size: 22px;
        font-weight: bold;
        margin: 15px 0;
    }
    
    /* buttons  */


    .stButton>button {
        background-color: #3B82F6;
        color: white;
        font-weight: bold;
        padding: 10px 24px;
        border: none;
        border-radius: 6px;
        cursor: pointer;
        transition: all 0.3s;
        width: 100%;
    }
    
    .stButton>button:hover {
        background-color: #2563EB;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
    }
    
    /* inputs for taking num_articles for getting number of articles needed */


    .stTextInput>div>div>input, .stNumberInput>div>div>input {
        border-radius: 6px;
        border: 1px solid #CBD5E1;
        padding: 10px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
    }
    
    /* container showing chat answers upon user questions */

    .answer-container {
        background-color: #F8FAFC;
        border-left: 4px solid #3B82F6;
        padding: 15px;
        border-radius: 6px;
        margin-top: 10px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
    }
    
    /* divider */

    .divider {
        height: 1px;
        background-color: #E2E8F0;
        margin: 15px 0;
    }
</style>
""", unsafe_allow_html=True)

st.markdown(f'<div class="greeting">{greeting}!</div>', unsafe_allow_html=True)

if st.session_state.page == "input":
    st.markdown('<h1 class="main-header">Company News & Analysis</h1>', unsafe_allow_html=True)
    
    st.markdown('<div class="container">', unsafe_allow_html=True)
    st.markdown('<h3 class="section-header">Enter Company Details</h3>', unsafe_allow_html=True)
    
    company = st.text_input("Company Name", placeholder="e.g. Google, Microsoft, Tesla")
    num_articles = st.number_input("Number of Articles", min_value=1, max_value=20, value=4, step=1, 
                                   help="Select how many articles to analyze (1-20)")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        process = st.button("Process Company Data")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    if process:
        if company:
            with st.spinner("Processing articles..."):
                # company output report in "json" 
                output = get_output(company, num_articles)
                st.session_state.output = output
                st.session_state.corpus = output.get("Article corpus", "")
                
                # save json to temp folder I made at the starting 

                with open(st.session_state.json_path, "w") as f:
                    json.dump(output, f, indent=4)
                
                # saving vectorstore for RAG 

                save_vector_store(st.session_state.corpus, save_path=os.path.join(OUTPUT_DIR, "vectorstore"))
                
                # saving mp3 audio file for hindi audio of the summary
                st.session_state.audio_path = text_to_speech(output.get("Hindi summary", ""), os.path.join(OUTPUT_DIR, "output.mp3"))
            
            st.success("Processing completed successfully!")
            st.session_state.page = "results"
            st.rerun()
        else:
            st.error("Please enter a company name to proceed.")

elif st.session_state.page == "results":
    st.markdown('<h1 class="main-header">Company Report and Interactive Chat</h1>', unsafe_allow_html=True)
    
    # audio and it's download 
    st.markdown('<div class="container">', unsafe_allow_html=True)
    st.markdown('<h3 class="section-header">Report Outputs</h3>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        # audio playing button
        if st.button("🔊 Play Hindi Summary"):
            if st.session_state.audio_path and os.path.exists(st.session_state.audio_path):
                play_mp3(st.session_state.audio_path)
            else:
                st.error("Audio file not found.")
    
    with col2:
        # audio downloading button
        with open(st.session_state.json_path, "rb") as f:
            st.download_button(
                "📄 Download Company Report", 
                f, 
                file_name="company_report.json", 
                mime="application/json"
            )
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # chat container 
    st.markdown('<div class="container">', unsafe_allow_html=True)
    st.markdown('<h3 class="section-header">Chat with an LLM</h3>', unsafe_allow_html=True)
    
    mode = st.radio(
        "Select Chat Mode:", 
        options=["simple", "advanced"], 
        index=0, 
        horizontal=True,
        help="Simple mode gives quick answers || Advanced provides more detailed answers"
    )
    
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    
    user_question = st.text_input(
        "Ask a question about the company:", 
        placeholder="e.g. What are the main products or recent developments?"
    )
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        submit_question = st.button("Submit Question")
    
    if submit_question:
        if user_question:
            with st.spinner("Analyzing your question..."):
                answer = chat_questions(user_question, st.session_state.corpus, mode=mode)
            
            st.markdown("<div class='answer-container'>", unsafe_allow_html=True)
            st.markdown(f"**Answer:** {answer}")
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.error("Please enter a question to proceed.")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # resets the application and takes back to the home page (where company name and num_articles are entered)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("Take me to Homepage"):
            reset_app()
            st.rerun()
