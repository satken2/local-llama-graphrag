import os
from dotenv import load_dotenv
from neo4j import GraphDatabase
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_community.graphs import Neo4jGraph
from langchain_community.vectorstores import Neo4jVector
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_experimental.graph_transformers import LLMGraphTransformer

load_dotenv()

# Initialize the Neo4j graph and LLM model
graph = Neo4jGraph()
llm = ChatOllama(model="llama3.2", temperature=0, format="json")

def make_graph():
    # Load documents from a text file
    loader = TextLoader(file_path="harrypotter.txt", encoding = 'UTF-8')
    docs = loader.load()

    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=250, chunk_overlap=24)
    documents = text_splitter.split_documents(documents=docs)
    
    print("Split documents")

    # Convert documents to graph format
    llm_transformer = LLMGraphTransformer(llm=llm, ignore_tool_usage=False)
    graph_documents = llm_transformer.convert_to_graph_documents(documents)

    # Add documents to the graph
    graph.add_graph_documents(
        graph_documents,
        baseEntityLabel=True,
        include_source=True
    )

    print("Convert documents to vectors")

    # Initialize embeddings
    embeddings = OllamaEmbeddings(
        model="mxbai-embed-large",
    )

    # Create a vector index from the existing graph
    Neo4jVector.from_existing_graph(
        embeddings,
        search_type="hybrid",
        node_label="Document",
        text_node_properties=["text"],
        embedding_node_property="embedding"
    )

    # Connect to the Neo4j database and create a full-text index
    driver = GraphDatabase.driver(
        uri=os.environ["NEO4J_URI"],
        auth=(os.environ["NEO4J_USERNAME"], os.environ["NEO4J_PASSWORD"])
    )

    def create_fulltext_index(tx):
        query = '''
        CREATE FULLTEXT INDEX `fulltext_entity_id` 
        FOR (n:__Entity__) 
        ON EACH [n.id];
        '''
        tx.run(query)

    try:
        with driver.session() as session:
            session.execute_write(create_fulltext_index)
            print("Fulltext index created successfully.")
    except Exception as e:
        print(f"Failed to create index: {e}")

    driver.close()

if __name__ == "__main__":
    make_graph()
    print("Graph generation complete.")
