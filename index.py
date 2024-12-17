from bs4 import BeautifulSoup
import requests
import os
import pickle
import time
from langchain_groq import ChatGroq
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA

# Initialize LLM
llm = ChatGroq(
    temperature=0,
    groq_api_key="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",
    model_name="llama-3.1-70b-versatile"
)

# File path for FAISS index
file_path = "faiss_store_openai.pkl"

# Function to scrape content from a website
def scrape_website(url):
    """
    Scrapes text content from the given URL.

    Args:
        url (str): The URL to scrape.

    Returns:
        str: The scraped text content.
    """
    try:
        response = requests.get(url, timeout=10)

        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')
            paragraphs = soup.find_all('p')
            text = ' '.join(para.get_text(strip=True) for para in paragraphs)
            return text
        else:
            print(f"Failed to retrieve {url} (Status code: {response.status_code})")
            return ""
    except requests.exceptions.RequestException as e:
        print(f"Error fetching {url}: {e}")
        return ""

# Function to process a list of websites
def process_websites(urls):
    """
    Processes the list of URLs, extracts text, and builds a FAISS index.

    Args:
        urls (list): List of URLs to scrape and process.

    Returns:
        None
    """
    all_text = ""

    for url in urls:
        print(f"Processing website: {url}")
        extracted_text = scrape_website(url)
        if extracted_text:
            all_text += extracted_text + "\n"
        else:
            print(f"No content extracted from: {url}")

    if not all_text.strip():
        print("No valid content found. Exiting.")
        return

    print("Splitting text into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    text_chunks = text_splitter.split_text(all_text)

    print("Creating embeddings and building FAISS index...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_texts(text_chunks, embeddings)

    # Save FAISS index to file
    with open(file_path, "wb") as f:
        pickle.dump(vectorstore, f)

    print("FAISS index saved successfully.")

# Function to load FAISS index and handle queries
def query_faiss():
    """
    Loads the FAISS index and allows querying.

    Returns:
        None
    """
    if not os.path.exists(file_path):
        print("FAISS index not found. Please process websites first.")
        return

    with open(file_path, "rb") as f:
        vectorstore = pickle.load(f)

    chain = RetrievalQA.from_llm(llm=llm, retriever=vectorstore.as_retriever())

    while True:
        query = input("\nAsk a Question (or type 'exit' to quit): ").strip()
        if query.lower() == "exit":
            print("Goodbye!")
            break

        if query:
            print("Processing query...")
            try:
                result = chain.run(query)
                print("Answer:")
                print(result)
            except Exception as e:
                print(f"Error processing query: {e}")

# Main workflow
def main():
    """
    Main function to execute the workflow: scrape, process, and query.

    Returns:
        None
    """
    urls_input = input("Enter the website URLs (comma-separated): ").strip()
    if not urls_input:
        print("No URLs provided. Exiting.")
        return

    urls = [url.strip() for url in urls_input.split(',') if url.strip()]
    if not urls:
        print("Invalid URLs. Exiting.")
        return

    # Process the URLs
    process_websites(urls)

    # Allow querying the FAISS index
    query_faiss()

# Execute main workflow
if __name__ == "__main__":
    main()
