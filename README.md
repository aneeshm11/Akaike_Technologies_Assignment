# Company News Articles Summarization & Analysis

My assignment is a full-stack application that gathers, processes, and analyzes company news articles. It provides both a web interface and RESTful API endpoints to deliver a detailed company report that includes article summaries, sentiment analysis, topic extraction, and even audio summaries in Hindi. The application uses various natural language processing (NLP) techniques, web scraping methods, and large language model (LLM) integrations to produce its outputs.

## Tech Stack

The following tech stacks are mainly used and power my application:

1. **Langchain, FAISS, Transformers, Gemini and Groq API** for NLP and text processing and summarization 
2. **Beautiful/Bs4** for web scrapping 
3. **Streamlit, FastAPI** for deployement
4. **gTTs, playsound, VLC** for audio related operations

## Workflow

1. **News Article Retrieval** from multiple news sources (using DuckDuckGo, Bing, and direct access) to gather a specified number of articles about a company.

2. **Using NLP techniques and LLM interactions** to extract key components from each article:
   - **Title Extraction**: Retrieves or generates a title.
   - **Summary Generation**: Produces a detailed summary.
   - **Sentiment Analysis**: Classifies articles as "Positive" or "Negative" or "Neutral".
   - **Topic Extraction**: Identifies and lists topics covered in each article.

3. **Aggregates individual article analyses** to:
   - Generate a comparative sentiment score.
   - Identify common and unique topics across articles.
   - Produce a final review with supportive evidence from the articles.
   - Convert the final review to a Hindi audio summary using text-to-speech.

4. **Chat Interface**: Provides two chat modes (simple and advanced) to ask follow-up questions about the company report:
   - **Simple Mode**: Answers are generated using the full corpus of articles.
   - **Advanced Mode**: Uses a vector store (via FAISS and LangChain) for retrieval-augmented generation (RAG) based on the most relevant text chunks.

5. **Web Interface**: 
   - **Frontend**: Using Streamlit apk on huggingface spaces 
   - **Backend**: Using fastAPI

## User Interaction

1. The user inputs a company name and the desired number of articles.
2. The articles are scrapped and processed in the backend
3. An output report in form of json is produced having all key information extracted
4. Hindi audio is generated summarizing final review of summary.
5. User can ask questions about the company and LLM answers from the article corpus or using a RAG system

In the repo I have included a jupyter notebook which works faster without all the backend and frontend code.
