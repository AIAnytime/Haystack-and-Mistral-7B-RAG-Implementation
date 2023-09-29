from haystack.nodes import EmbeddingRetriever, MarkdownConverter, PreProcessor, AnswerParser, PromptModel, PromptNode, PromptTemplate
from haystack.document_stores import WeaviateDocumentStore
from haystack import Pipeline
from haystack.preview.components.file_converters.pypdf import PyPDFToDocument
from haystack.preview.dataclasses import Document
from model_add import LlamaCPPInvocationLayer

from fastapi import FastAPI, Depends, HTTPException, status, Request, Form, Response
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.encoders import jsonable_encoder
import uvicorn
import json
import re
import os
from dotenv import load_dotenv
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

print("Import Successfully")

app = FastAPI()

# Configure templates
templates = Jinja2Templates(directory="templates")

app.mount("/static", StaticFiles(directory="static"), name="static")

def get_result(query):
    # query = "Why was Manual of Air Traffrc Services prepared?"
    document_store = WeaviateDocumentStore(
        host='http://localhost',
        port=8080,
        embedding_dim=384
    )

    prompt_template = PromptTemplate(prompt = """"Given the provided Documents, answer the Query. Make your answer detailed and long\n
                                                Query: {query}\n
                                                Documents: {join(documents)}
                                                Answer: 
                                            """,
                                            output_parser=AnswerParser())
    print("Prompt Template: ", prompt_template)
    def initialize_model():
        return PromptModel(
            model_name_or_path="model/mistral-7b-instruct-v0.1.Q4_K_S.gguf",
            invocation_layer_class=LlamaCPPInvocationLayer,
            use_gpu=False,
            max_length=512
        )
    model = initialize_model()
    prompt_node = PromptNode(model_name_or_path = model, # "gpt-3.5-turbo",
                            # api_key = OPENAI_API_KEY,
                            max_length=1000,
                            default_prompt_template = prompt_template)
    print("Prompt Node: ", prompt_node)

    retriever = EmbeddingRetriever(
        document_store=document_store,
        embedding_model="sentence-transformers/all-MiniLM-L6-v2"
    )

    print("Retriever: ", retriever)

    query_pipeline = Pipeline()
    query_pipeline.add_node(component=retriever, name="Retriever", inputs=["Query"])
    query_pipeline.add_node(component=prompt_node, name="PromptNode", inputs=["Retriever"])
    print("Query Pipeline: ", query_pipeline)

    json_response = query_pipeline.run(query = query , params={"Retriever" : {"top_k": 5}})
    print("Answer: ", json_response)
    answers = json_response['answers']
    for ans in answers:
        answer = ans.answer
        break

    # Extract relevant documents and their content
    documents = json_response['documents']
    document_info = []

    for document in documents:
        content = document.content
        document_info.append(content)

    # Print the extracted information
    print("Answer:")
    print(answer)
    # Split the text into sentences using regular expressions
    sentences = re.split(r'(?<=[.!?])\s', answer)

    # Filter out incomplete sentences
    complete_sentences = [sentence for sentence in sentences if re.search(r'[.!?]$', sentence)]

    # Rejoin the complete sentences into a single string
    updated_answer = ' '.join(complete_sentences)

    relevant_documents = ""
    for i, doc_content in enumerate(document_info):
        relevant_documents+= f"Document {i + 1} Content:"
        relevant_documents+=doc_content
        relevant_documents+="\n"

    print("Relevant Documents:", relevant_documents)

    return updated_answer, relevant_documents

@app.get("/")
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/get_answer")
async def get_answer(request: Request, question: str = Form(...)):
    print(question)
    answer, relevant_documents = get_result(question)
    response_data = jsonable_encoder(json.dumps({"answer": answer, "relevant_documents": relevant_documents}))
    res = Response(response_data)
    return res

# if __name__ == "__main__":
#     uvicorn.run("app:app", host='0.0.0.0', port=8001, reload=True)