import os
import numpy as np
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from neo4j import GraphDatabase
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.utils.function_calling import convert_to_openai_function
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_community.graphs import Neo4jGraph
from langchain_community.vectorstores import Neo4jVector
from langchain_community.vectorstores.neo4j_vector import remove_lucene_chars
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_experimental.graph_transformers import LLMGraphTransformer

# Read environment variables from .env
load_dotenv()

# Initialize the Neo4j graph and LLM model
graph = Neo4jGraph()
llm = ChatOllama(model="llama3.2", temperature=0, format="")

# Retrieve the vector retriever
embeddings = OllamaEmbeddings(model="mxbai-embed-large")
vector_index = Neo4jVector.from_existing_index(
    embeddings,
    search_type="hybrid",
    index_name="Document_embedding",  # 既存のインデックス名を指定
    keyword_index_name="Document_keyword",  # 既存のキーワードインデックス名を指定
    node_label="Document",
    text_node_property="text",
    embedding_node_property="embedding"
)
vector_retriever = vector_index.as_retriever()

# Define the Pydantic model for entities
class Entities(BaseModel):
    """Identifying information about entities."""
    names: list[str] = Field(
        ...,
        description="All the person, organization, or business entities that "
        "appear in the text",
    )
    
def graph_retriever(question: str, top_n=5) -> str:
    # Create a prompt for extracting entities
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are extracting organization and person entities from the text."),
            ("human", "Use the given format to extract information from the following input: {question}"),
        ]
    )

    dict_schema = convert_to_openai_function(Entities)
    entity_chain = prompt | llm.with_structured_output(dict_schema)
    entities = entity_chain.invoke({"question": question})
    entity_names = eval(entities['names'])

    # Retrival of related nodes
    related_nodes = []
    for entity in entity_names:
        response = graph.query(
            """
            MATCH (n)-[:MENTIONS]->(m)
            WHERE m.id = $entity
            RETURN n.text AS text, n.embedding AS embedding
            """,
            {"entity": entity},
        )
        for record in response:
            text = record["text"]
            embedding = np.array(record["embedding"])
            related_nodes.append((entity, text, embedding))
    
    # Create embeddings from question
    question_embedding = embeddings.embed_query(question)

    # Calculation and sorting of cosine similarity
    def cosine_similarity(vec1, vec2):
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

    related_nodes.sort(
        key=lambda x: cosine_similarity(question_embedding, x[2]),
        reverse=True
    )
    
    top_related_texts = [node[0] for node in related_nodes[:top_n]]
    
    graph_context = ""
    for node in related_nodes[:top_n]:
        graph_context += f"#Document\nExplanation about '{node[0]}' - '{node[1]}'\n\n"
    return graph_context

def vector_retriver(question: str, top_n=15) -> str:
    vector_data = [el.page_content for el in vector_retriever.invoke(question)]
    vector_context = ""
    for text in vector_data[:top_n]:
        vector_context += f"#Document\nAdditional information - '{text}'\n\n"
    return vector_context

def context_builder(question: str) -> str:
    graph_data = graph_retriever(question)
    vector_data = vector_retriver(question)
    return f"""
    Context from graph data:
    {graph_data}

    Context from vector data:
    {vector_data}
    """

# Create a prompt template for the final answer
template = """Answer the question based only on the following context:
{context}

Question: {question}

Use natural language and be concise.

Answer:"""
prompt = ChatPromptTemplate.from_template(template)

# Set up the final chain
chain = (
    {
        "context": context_builder,
        "question": lambda x: x,
    }
    | prompt
    | llm
    | StrOutputParser()
)

if __name__ == "__main__":
    question = input("Question: ")
    result = chain.invoke(question)
    print("Answer:", result)
