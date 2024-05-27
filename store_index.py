from src.helper import load_pdf,text_split
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv
import os

load_dotenv()

PINECONE_API_KEY=os.environ.get("PINECONE_API_KEY")
key=os.environ.get('key')


# print(PINECONE_API_KEY)
# print(key)

extracted_data=load_pdf("data/")
text_chunks=text_split(extracted_data)
repo_id='sentence-transformers/all-MiniLM-L6-v2'
embeddings=HuggingFaceInferenceAPIEmbeddings(repo_id=repo_id , api_key=key,add_to_git_credential=True)



index_name="datascience"
docsearch=PineconeVectorStore.from_texts([t.page_content for t in text_chunks],embeddings,index_name=index_name)