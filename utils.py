import re
import os
import sys
import time
import json
import random
import requests
from groq import Groq
from gtts import gTTS
import streamlit as st
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from transformers import pipeline
import google.generativeai as genai
from langchain_groq import ChatGroq
from langchain.vectorstores import FAISS
from urllib.parse import quote_plus, urlparse
from langchain.docstore.document import Document
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

os.environ['GROQ_API_KEY'] = st.secrets["GROQ_KEY"]
os.environ['LANGCHAIN_API_KEY'] = st.secrets["LANGCHAIN_API_KEY"]

print(st.secrets["GROQ_KEY"] , st.secrets["LANGCHAIN_API_KEY"] , st.secrets["GOOGLE_API_KEY"] )


genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
client = Groq()
model_gemini = genai.GenerativeModel("gemini-1.5-flash")

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# to convert english text to hindi for playing hindi audio of the summary
translator = pipeline("translation_en_to_hi", model="Helsinki-NLP/opus-mt-en-hi") 

# WEB SCRAPER and its helper functions
def is_valid_url(url):
    if not url:
        return False

    excluded_domains = [
        'google.com', 'youtube.com', 'facebook.com', 'twitter.com', 
        'instagram.com', 'tiktok.com', 'linkedin.com', 'reddit.com'
    ]

    try:
        domain = urlparse(url).netloc.lower()
        if domain.startswith('www.'):
            domain = domain[4:]
        return not any(domain.endswith(excluded) for excluded in excluded_domains)
    except:
        return False

def get_random_headers():
    # random headers to avoid detection.

    user_agents = [
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.131 Safari/537.36',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.2 Safari/605.1.15',
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:90.0) Gecko/20100101 Firefox/90.0'
    ]
    
    return {
        'User-Agent': random.choice(user_agents),
        'Accept-Language': 'en-US,en;q=0.9',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8'
    }

def is_english_content(content):
     # for now I will only consider english webpages

    if not content:
        return False

    english_words = ['the', 'and', 'for', 'that', 'with', 'this']
    text_lower = content.lower()
    
    word_count = sum(1 for word in english_words if f" {word} " in text_lower)
    return word_count >= 3

def direct_news_search(keyword):
    urls = []
    encoded_keyword = quote_plus(keyword)

    # directly accessing news sources as google and bing gave errors when number of requests were too many
    news_sites = [
        f"https://www.reuters.com/search/news?blob={encoded_keyword}",
        f"https://www.bloomberg.com/search?query={encoded_keyword}",
        f"https://www.cnbc.com/search/?query={encoded_keyword}",
        f"https://www.bbc.com/news/search?q={encoded_keyword}",
        f"https://www.nytimes.com/search?query={encoded_keyword}",
        f"https://www.theguardian.com/search?q={encoded_keyword}&type=article"
    ]
    urls.extend(news_sites)

    company_domain = keyword.lower().replace(' ', '')
    company_urls = [
        f"https://www.{company_domain}.com",
        f"https://en.wikipedia.org/wiki/{encoded_keyword}"
    ]
    urls.extend(company_urls)

    return urls

def search_duckduckgo(keyword):
    #using DuckDuckGo for getting keyword and  URLs
    urls = []
    search_query = f"{keyword} news"
    try:
        ddg_url = f"https://html.duckduckgo.com/html/?q={quote_plus(search_query)}"
        headers = get_random_headers()
        response = requests.get(ddg_url, headers=headers, timeout=10)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            for result in soup.select('.result__a'):
                href = result.get('href')
                if href and '/uddg=' in href:
                    url = href.split('/uddg=')[1].split('&')[0]
                    url = requests.utils.unquote(url)
                    if is_valid_url(url):
                        urls.append(url)
    except Exception as e:
        print(f"DuckDuckGo search error: {e}")
    return urls

def search_bing(keyword):
    urls = []
    search_query = f"{keyword} news"
    try:
        bing_url = f"https://www.bing.com/search?q={quote_plus(search_query)}&setlang=en"
        headers = get_random_headers()
        response = requests.get(bing_url, headers=headers, timeout=10)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            for link in soup.select('a[href^="http"]'):
                url = link.get('href')
                if url and is_valid_url(url):
                    urls.append(url)
    except Exception as e:
        print(f"Bing search error: {e}")
    return urls

# extracts html components from the web page
def extract_full_content(url):  
    try:
        headers = get_random_headers()
        response = requests.get(url, headers=headers, timeout=15)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            content = " "
            
             #  title
            if soup.title:
                content += f"TITLE: {soup.title.string.strip()}\n\n"
           
            # article meta data
            meta_tags = {}
            for meta in soup.find_all('meta'):
                name = meta.get('name', meta.get('property', '')).lower()
                if name and 'content' in meta.attrs:
                    meta_tags[name] = meta['content']
            important_meta = ['description', 'keywords', 'author', 'date', 
                              'article:published_time', 'og:title', 'og:description']
            for meta_name in important_meta:
                if meta_name in meta_tags:
                    content += f"METADATA {meta_name.upper()}: {meta_tags[meta_name]}\n"
            content += "\n"
            
            # headings / header tags in the html page
            for heading in soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6']):
                text = heading.get_text(strip=True)
                if text:
                    content += f"{heading.name.upper()}: {text}\n"
            content += "\nMAIN CONTENT:\n"
            
            text_blocks = set()  # repeated texts are removed as an edgecase
            content_containers = soup.find_all(['article', 'main']) or []
            
            # all kinds of html elements are checked here 
            
            content_patterns = ['content', 'article', 'story', 'news', 'body', 'text']
            for pattern in content_patterns:
                elements = soup.find_all(class_=re.compile(f".*{pattern}.*", re.I))
                elements.extend(soup.find_all(id=re.compile(f".*{pattern}.*", re.I)))
                content_containers.extend(elements)
            
            if content_containers:
                for container in content_containers:
                    for p in container.find_all('p'):
                        text = p.get_text(strip=True)
                        if text and len(text) > 30 and text not in text_blocks:
                            content += f"{text}\n\n"
                            text_blocks.add(text)
                    for div in container.find_all('div'):
                        if any(c in str(div.get('class', [])) for c in ['nav', 'menu', 'header', 'footer']):
                            continue
                        text = div.get_text(strip=True)
                        if text and len(text) > 100 and text not in text_blocks:
                            content += f"{text}\n\n"
                            text_blocks.add(text)
            
            if len(text_blocks) < 5:
                for p in soup.find_all('p'):
                    text = p.get_text(strip=True)
                    if text and len(text) > 30 and text not in text_blocks:
                        content += f"{text}\n\n"
                        text_blocks.add(text)
            return content
    except Exception as e:
        print(f"Error extracting content from {url}: {e}")
    return None

def get_articles(company_keyword, num_articles=10):

    # use all above helper functions, uses all functions to handle cases when number of articles obtained is lesser than required

    list_articles = []
    urls = []

    # check for URLs using multiple methods
    urls.extend(search_duckduckgo(company_keyword))
    if len(urls) < num_articles * 2:
        new_urls = search_bing(company_keyword)
        urls.extend([url for url in new_urls if url not in urls])

    # if not enough, try news sites directly
    if len(urls) < num_articles * 2:
        new_urls = direct_news_search(company_keyword)
        urls.extend([url for url in new_urls if url not in urls])
    
    
    # go to each URL and extract content
    for url in urls:
        if len(list_articles) >= num_articles:
            break
        try:
            time.sleep(1 + random.random())
            content = extract_full_content(url)
            if content and is_english_content(content):
                list_articles.append(content)
        except Exception as e:
            print(f"Error processing {url}: {e}")
            continue
    # if still number of articles needed is not achieved,  this will just put some random placeholder text

    while len(list_articles) < num_articles:
        list_articles.append(f"Unable to retrieve content #{len(list_articles)+1} for {company_keyword}.")
    return list_articles[:num_articles]

# Text to Speech and saving audio file

# below "hi" -> hindi
def text_to_speech(text, output_path, language='hi'): 
    try:
        if os.path.isdir(output_path) or not os.path.splitext(output_path)[1]:
            output_path = os.path.join(output_path, "output.mp3")
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        tts = gTTS(text=text, lang=language, slow=False)
        tts.save(output_path)
        
        print(f"Audio saved to {output_path}")
        return output_path
    
    except Exception as e:
        print(f"Error in text to speech conversion: {e}", file=sys.stderr)
        return None

def play_mp3(filepath):
    try:
        # tries to use streamlit audio playback which opens file in binary mode.
        with open(filepath, "rb") as f:
            audio_bytes = f.read()
        st.audio(audio_bytes, format="audio/mp3")
        return
    except Exception as e:
        print("Streamlit audio playback failed, trying alternative methods:", e)
    
    # if above method fails , uses  playsound method
    try:
        from playsound import playsound
        playsound(filepath)
    except Exception as e:
        #  if playsound fails uses vlc
        try:
            import vlc
            player = vlc.MediaPlayer(filepath)
            player.play()
            # wait till playback finishes or an error occurs usually
            while True:
                state = player.get_state()
                if state in (vlc.State.Ended, vlc.State.Error):
                    break
                time.sleep(0.5)
        except Exception as e2:
            print("Audio playback error using VLC:", e2)


# CHAT FUNCTIONS for LLMs (To process the article). 
# I have put them in the same order they are called in the get_output functions that makes a complete json report of the company


trim = 1000
# this to limit the amount of text sent to LLM api as groq has Tokens sent per minute limit, around 6000 tokens
# And it will give errors while using free tier as the articles scrapped are very long

def chat(question, model="llama"):
    
# this is the main function that is used multiple times in upcoming functions. meant for interating with llm
    if model=="llama":
        chat_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": question}],
            model="llama3-70b-8192",
        )
        answer = chat_completion.choices[0].message.content
        return answer
    
    elif model=="gemini":
        answer = model_gemini.generate_content(
            question, 
            generation_config = genai.GenerationConfig(
                max_output_tokens=8192,
                temperature=0.8,
            )
        )
        return answer.text

# gets the title, summary, sentiment and list of topics present in a single article 
def process_article(article):

    instruction = f"""
    
    Given below is an article content about a company. Read the content thoroughly and perform the following:
    1) The title of the article should be extratced if present. If it is not present give me a suitable title for the article given.
    2) A very informative summary of the article should be made with good detials and length, do not miss any important point present while writing the summary.
    3) A sentiment which : either "Positive" or "Negative" should be assigned to the article I have sent you.
    4) The list of topics that the article covers should be put in a list.
    
    So after processing and producing these 4. You put them in the following format, check the example I have given below. This is very important 
    boxed_title{{"Tesla's New Model Breaks Sales Records"}}
    boxed_summary{{"Tesla's latest EV sees record sales in the year..."}}
    boxed_sentiment{{"Negative"}}
    LIST_TOPICS = ["Electric Vehicles", "Stock Market", "Innovation"]

    The topics you extracted should be placed in a list and assigned = to the LIST_TOPICS variable, do not change name here. 
    This is very important as I will be using these particular regex pattern to extract the 4 things I listed, so make sure your response will work with the below regex patterns. 
    

    The LIST_TOPICS equal to should be present like I asked.

    Below is the article on which you to perform the tasks I have asked you
    
    Article:
    {article}

    Follow all instructions carefully and produce the 4 components I have asked. This will be the end of your response. 
    """

    response = chat(instruction[:trim])
    
    boxed_title = re.search(r'boxed_title\s*{\s*"(.+?)"\s*}', response)
    boxed_summary = re.search(r'boxed_summary\s*{\s*"(.+?)"\s*}', response)
    boxed_sentiment = re.search(r'boxed_sentiment\s*{\s*"(.+?)"\s*}', response)
    
    list_topics_match = re.search(r'LIST_TOPICS\s*=\s*\[(.*?)\]', response, re.DOTALL)
    list_topics = re.findall(r'"(.*?)"', list_topics_match.group(1)) if list_topics_match else ["No topics found"]
    
    title = boxed_title.group(1) if boxed_title else "No title"
    summary = boxed_summary.group(1) if boxed_summary else "No summary"
    sentiment = boxed_sentiment.group(1) if boxed_sentiment else "No sentiment"

    print(f"Article Title: {title} Article Summary:  {summary[:50]} Article sentiment: {sentiment}")
    
    return title, summary, sentiment, list_topics


# generates a final review of the company using all articles 
def review_company(all_articles):

    instruction = f"""
    
    Given below are a number of articles about a company and a positive vs negative feedback assigned to each article of the company found on internet.
    The number of positive and negative articles I will give you may or may not be equal.
    I want you to compare all the articles and write a detailed final review about the company, stating your reasons for if it is really good (positive) or not good (negative).
    If you don't find a strong review based on the content I have given, write a plausible and relevant review for the company
    
    After that enclose the final review within the following: 
    The example text I gave should be replaced with the actual review 
    boxed_final_review{{"Tesla’s latest news coverage is mostly positive beacuse"}}
    
    I will be using regex pattern so you should ensure your response follows the format I asked     

    Here are all the articles I have for you to go through:
    {all_articles}
    
    Follow all instructions carefully and produce the final review I have asked. This will be the end of your response. 
    """
    
    response = chat(instruction[:trim])
    match = re.search(r'boxed_final_review\s*{\s*"(.*?)"\s*}', response, re.DOTALL)
    
    
    try:
        print("Extracted Review:", final_review[:40])
        print("response" , response[:100])
    except Exception as e:
        print("Error occured while reviewing company")
        pass

    final_review = match.group(1).strip() if match else "No concrete review found"
    return final_review

# takes in large text of all articles and gives of impacts and comparisons between the articles
def coverage_diff(all_articles):

    instruction = f"""
    
    Given below are a number of articles about a company found on the internet. For each article I want you to:
    1) Determine it's impact in terms of portraying details about the company
    2) Go each article and compare with any other article on which we can draw some comparisons

    So for each article get the impact of the article and comparison with other articles 
    You can atmost compare it with 2 articles so make sure other articles picked are well suited for comparison with the article being compared.
    So all the impacts and comparions should be added to 2 separate lists like this :
    For example: 
    
    IMPACT_LIST = ["Article 1 highlights Tesla's strong sales, while Article 2 discusses regulatory issues.",  "Article 3 is focused on financial success and innovation,whereas Article 4 is about legal challenges and risks."]
    COMPARISON_LIST = ["The first article boosts confidence in Tesla's market growth, while the second raises concerns about future regulatory hurdles." , "Investors may react positively to growth news but stay cautious due to regulatory scrutiny."]
    
    So make sure this way of assigning is done and don't change the names of lists I gave because i will be using these regex patterns to extract these 2 lists

    Make sure the length of the 2 lists are same and also equal to the number of articles I have given. If in any extreme case you fail to assign the impact and comparison of any article, use any relevant knowledge you know to assign something for the impact and comparison, don't leave it empty.    

    These are the articles you need to go through:
    {all_articles}
    
    Follow all instructions carefully and produce the 2 lists I have asked. This will be the end of your response. 

    """

    response = chat(instruction[:trim])

    impact_list_match = re.search(r'IMPACT_LIST\s*=\s*\[(.*?)\]', response, re.DOTALL)
    impact_list = re.findall(r'"(.*?)"', impact_list_match.group(1)) if impact_list_match else ["No impacts found"]
    
    comparison_list_match = re.search(r'COMPARISON_LIST\s*=\s*\[(.*?)\]', response, re.DOTALL)
    comparison_list = re.findall(r'"(.*?)"', comparison_list_match.group(1)) if comparison_list_match else ["No comparisons found"]
    
    return impact_list, comparison_list

# english to hindi conversion using Helsinki models
def hindi(text: str) -> str:
    try:
        translation = translator(text, max_length=20480)
        hindi_text = translation[0]['translation_text']
        return hindi_text
    except Exception as e:
        print(f"Translation error: {e}")
        return ""

# uses all functions above to generate company report 
def get_output(company, num_articles=5):
    
    all_articles_list = []
    sentiment_distribution = {}
    articles_with_score = ""
    articles_without_score = ""
    list_of_topics = []

    # getting articles from web scrapper
    articles = get_articles(company, num_articles=num_articles)
    print("ALL ARTICLES GATHERED")

    for i, article in enumerate(articles):
        title, summary, sentiment, list_topics = process_article(article)  # to process article
        article_dict = {
            "Title": title,
            "Summary": summary,
            "Sentiment": sentiment,
            "Topics": list_topics
        }
        all_articles_list.append(article_dict)
        sentiment_distribution[sentiment] = sentiment_distribution.get(sentiment, 0) + 1
        list_of_topics.append(list_topics)
        articles_with_score += f"Article review: {sentiment} Article content: {article}\n"
        articles_without_score += f"Article {i}: {article}\n"
    
    print("ARTICLES summary, title, sentiment processed")
    
    final_review = review_company(articles_with_score)                    # to get review of the company
    impact_list, comparison_list = coverage_diff(articles_without_score)  # to get impact and comparison of each article
    print("ARTICLES review and impact processed")
    
    # to get common topics between all the topics of all articles
    common_topics = set(list_of_topics[0]) 
    for topics in list_of_topics[1:]:
        common_topics.intersection_update(topics)
    common_list_of_topics = sorted(common_topics) if common_topics else ["no common topics"]
    
    topic_overlap = {"Common Topics": common_list_of_topics}
    for index, topics in enumerate(list_of_topics, start=1):
        unique_topics = sorted(set(topics) - common_topics)
        topic_overlap[f"Unique Topics in Article {index}"] = unique_topics # to get unique topics of articles
    
    coverage_dicts_list = []
    for i, (impact, comparison) in enumerate(zip(impact_list, comparison_list)):
        dic = {f"Comparison Article {i}": comparison, f"Impact of Article {i}": impact}
        coverage_dicts_list.append(dic)
    
    # hindi review so that the hindi text would be used by the play mp3 function
    hindi_review = hindi(final_review) 
    print("ARTICLES review processed")    
    
    output = {
        "Company": company,
        "Articles": all_articles_list,
        "Comparative Sentiment Score": {
            "Sentiment Distribution": sentiment_distribution,
            "Coverage differences": coverage_dicts_list,
            "Topic overlap": topic_overlap
        },
        "Final sentiment analysis": final_review,
        "Hindi summary": hindi_review,
        "Article corpus": articles_without_score,
    }
    return output

# Functions for RAG System

def save_vector_store(corpus: str, chunk_size: int = 500, chunk_overlap: int = 50, save_path="data/vectorstore"):
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = text_splitter.split_text(corpus)
    
    documents = [Document(page_content=chunk) for chunk in chunks]
    vector_store = FAISS.from_documents(documents, embedding_model)
    
    # saving vectorstore for chatting later on
    os.makedirs(save_path, exist_ok=True)
    vector_store.save_local(save_path)
    
    print("Saved vectorstore")
    return None

def retrieve_relevant_text(question: str, top_k: int = 5, save_path="data/vectorstore") -> str:
    
    vector_store = FAISS.load_local(save_path, embedding_model, allow_dangerous_deserialization=True)
    retrieved_docs = vector_store.similarity_search(question, k=top_k)
    
    retrieved_text = " ".join([doc.page_content for doc in retrieved_docs])
    
    return retrieved_text

# uses the chat function for user q and a 
def chat_questions(user_question, corpus, model="gemini", mode="simple"):
    
    if mode == "simple":
        content = corpus
    elif mode == "advanced":
        content = retrieve_relevant_text(user_question)
    
    question = f"""
    Question:
    {user_question}

    Use the following content below to search your answer. If no answer is found from the content, give the most suitable answer from your knowledge
    Also give a small description of the company along with it. 
    Content:
    {content}
    
    """

    answer = chat(question[:trim], model)

    return answer
    