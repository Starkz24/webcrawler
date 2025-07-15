
# ChatBot (Web-Crawler)


This project involves building a system that crawls a user-specified website, scrapes its content, and then uses this content to answer user queries

## Tech Stack

- **Apify**: Used for crawling and scraping the website data. It extracts the website's textual content, including titles, descriptions, and keywords, and saves it for further processing.
- **Pinecone**: Utilized to create a vector database. The scraped data is split into manageable chunks and embedded using SentenceTransformerEmbeddings. These embeddings are then stored in a Pinecone index to enable efficient similarity searches.
- **Gemini API**: Employed for generating responses. When a user submits a query, the system searches the vector database for relevant documents and uses the Gemini Generative Model to generate a response based on the context provided by the retrieved documents.


## Work-Flow

- Crawling the website to retrieve content using Apify.
- Splitting and embedding the documents.
- Storing the embeddings in a Pinecone vector database.
- Querying the database and generating a context-based response using the Gemini API.
# webcrawler
