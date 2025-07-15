from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
import google.generativeai as genai
from apify_client import ApifyClient
import time
import os

os.environ["PINECONE_API_KEY"] = "cf9ca11f-36c3-45ff-bf59-9243760ff06e"

directory = "./content"


def load_docs(directory):
    loader = DirectoryLoader(directory)
    documents = loader.load()
    return documents


def split_docs(documents, chunk_size=500, chunk_overlap=20):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    docs = text_splitter.split_documents(documents)
    return docs


def crawl_website(url):
    client = ApifyClient("apify_api_ynf7ciQqcvo3sHs8VYYuXjT7AVpFu122QjQ3")

    run_input = {
        "startUrls": [{"url": url}],
        "useSitemaps": False,
        "crawlerType": "playwright:adaptive",
        "includeUrlGlobs": [],
        "excludeUrlGlobs": [],
        "ignoreCanonicalUrl": False,
        "maxCrawlDepth": 20,
        "maxCrawlPages": 5,
        "initialConcurrency": 0,
        "maxConcurrency": 200,
        "initialCookies": [],
        "proxyConfiguration": {"useApifyProxy": True},
        "maxSessionRotations": 10,
        "maxRequestRetries": 5,
        "requestTimeoutSecs": 60,
        "minFileDownloadSpeedKBps": 128,
        "dynamicContentWaitSecs": 10,
        "maxScrollHeightPixels": 5000,
        "removeElementsCssSelector": """nav, footer, script, style, noscript, svg,
        [role=\"alert\"],
        [role=\"banner\"],
        [role=\"dialog\"],
        [role=\"alertdialog\"],
        [role=\"region\"][aria-label*=\"skip\" i],
        [aria-modal=\"true\"]""",
        "removeCookieWarnings": True,
        "clickElementsCssSelector": '[aria-expanded="false"]',
        "htmlTransformer": "readableText",
        "readableTextCharThreshold": 100,
        "aggressivePrune": False,
        "debugMode": False,
        "debugLog": False,
        "saveHtml": False,
        "saveHtmlAsFile": False,
        "saveMarkdown": True,
        "saveFiles": False,
        "saveScreenshots": False,
        "maxResults": 300,
        "clientSideMinChangePercentage": 15,
        "renderingTypeDetectionPercentage": 10,
    }

    run = client.actor("aYG0l9s7dbB7j3gbS").call(run_input=run_input)

    with open("./content/data.txt", "w", encoding="utf-8") as file:
        for item in client.dataset(run["defaultDatasetId"]).iterate_items():
            text = item["text"]
            metadata = item["metadata"]
            title = metadata.get("title", "")
            description = metadata.get("description", "")
            keywords = metadata.get("keywords", [])
            url = item["url"]

            file.write(f"Title: {title}\n")
            file.write(f"URL: {url}\n")
            file.write(f"Keywords: {keywords}\n")
            file.write(f"Description: {description}\n")
            file.write(f"Text: {text}\n\n")
    documents = load_docs(directory)
    docs = split_docs(documents)
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    query_result = embeddings.embed_query("Snacks")

    pinecone_api_key =os.environ["PINECONE_API_KEY"]
    pc = Pinecone(api_key=pinecone_api_key)

    index_name = "langchain-index1"

    existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]

    if index_name not in existing_indexes:
        pc.create_index(
            name=index_name,
            dimension=384,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        while not pc.describe_index(index_name).status["ready"]:
            time.sleep(1)

    index = pc.Index(index_name)
    docsearch = PineconeVectorStore.from_documents(
        docs, embeddings, index_name=index_name
    )
    return docsearch


def generate_response(docsearch, query):
    GOOGLE_API_KEY = "AIzaSyCqdkrGKNUQ6kjkocJ8vsZQqAF3ag3x8cE"
    genai.configure(api_key=GOOGLE_API_KEY)

    docs = docsearch.similarity_search(query)

    model = genai.GenerativeModel("gemini-1.5-flash")
    prompt = f"User's Query: {query}\n\nContext: {docs}\nGenerate a response for the query using the context which is provided. If the answer is not in context then respond with I don't know."

    response = model.generate_content(prompt)

    return response


print("Welcome to the website crawler and query answer system!")

url = input("Please enter the URL of the website to crawl: ")

try:
    docsearch = crawl_website(url)
    print(docsearch, type(docsearch))
    print("Website content successfully retrieved.")

    while True:
        query = input("Please enter your query (or type 'exit' to quit): ")
        if query.lower() == "exit":
            print("Exiting the system. Goodbye!")
            break

        response = generate_response(docsearch, query)
        print(f"Answer: {response.text}")

except Exception as e:
    print(f"An error occurred: {e}")