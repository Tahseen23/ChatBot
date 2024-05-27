from flask import Flask,render_template,jsonify,request
from langchain_pinecone import PineconeVectorStore
from langchain_core.prompts import PromptTemplate
from langchain import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEndpoint
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain_community.vectorstores import Pinecone



from dotenv import load_dotenv
from src.prompt import *
import os

app=Flask(__name__)

load_dotenv()

PINECONE_API_KEY=os.environ.get("PINECONE_API_KEY")
key=os.environ.get('key')

repo_id="mistralai/Mistral-7B-Instruct-v0.3"
model=HuggingFaceEndpoint(repo_id=repo_id,huggingfacehub_api_token=key)


repo_id='sentence-transformers/all-MiniLM-L6-v2'
embeddings=HuggingFaceInferenceAPIEmbeddings(repo_id=repo_id , api_key=key)

index_name="datascience"
docsearch=Pinecone.from_existing_index(index_name,embeddings)

prompt=PromptTemplate(template=Prompt_tempelate,input_variables=['context','question'])
chain_type_kwargs={'prompt':prompt}

qa=RetrievalQA.from_chain_type(
    llm=model,
    chain_type="stuff",
    retriever=docsearch.as_retriever(search_kwargs={'k':2}),
    return_source_documents=True,
    chain_type_kwargs=chain_type_kwargs
)


@app.route("/")
def index():
    return render_template("chat.html")

@app.route("/get",methods=['POST'])
def chat():
    msg=request.form['msg']
    input=msg
    print(input)
    result=qa({'query':input})
    print("Response:",result['result'])
    return str(result['result'])


if __name__=="__main__":
     app.run(debug=True, use_reloader=False, port=8000)


