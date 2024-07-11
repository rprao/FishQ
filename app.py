from llama_index.core import ServiceContext, StorageContext
from llama_index.core import VectorStoreIndex
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.llms.huggingface_api import HuggingFaceInferenceAPI
from llama_index.core import load_index_from_storage
from llama_index.core import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import VectorStoreIndex, get_response_synthesizer
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import SimilarityPostprocessor
import gradio as gr
import os
from huggingface_hub import login

#Define the LLM to be used
Settings.llm = HuggingFaceInferenceAPI(model_name = "mistralai/Mixtral-8x7B-Instruct-v0.1")

#Define the embedding model
embed_model = HuggingFaceEmbedding(model_name="thenlper/gte-large")
Settings.embed_model = embed_model

#
PERSIST_DIR = "./storage"
storage_context = StorageContext.from_defaults(persist_dir = PERSIST_DIR)
index = load_index_from_storage(storage_context,index_id = "vector_index")

def gradio_interface(prompt):
    try:
        # Replace 'storage_context' with the actual storage context if needed
        index = load_index_from_storage(storage_context, index_id="vector_index")
        
        retriever = VectorIndexRetriever(
            index=index,
            similarity_top_k=2,
        )
        
        response_synthesizer = get_response_synthesizer()
        
        query_engine = RetrieverQueryEngine(
            retriever=retriever,
            response_synthesizer=response_synthesizer,
            node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=0.7)],
        )
        
        response = query_engine.query(prompt)
        return response
    
    except Exception as e:
        return f"An error occurred: {str(e)}"

# Define the Gradio interface

interface = gr.Interface(
    fn=gradio_interface,
    theme = gr.themes.Soft(),
    inputs=gr.Textbox(lines=2, placeholder="Enter your query here...", label="Query"),
    outputs=gr.Textbox(label="Response"),
    title="FishQ-RAG",
    description="RAG Agent for 2021 CPCSEA Fish Experimentation Guidelines",
    article="The guidelines are intended to provide information on the humane procedures for holding, handling, and sampling fish for experimental, research, or teaching purposes.",
    examples=[
        ["What is CPCSEA?"],
        ["Give me guidlines for Zebrafish."],
        ["Write an compact email about the guidelines with their background information.."]
    ],
    live=True,
    allow_flagging="never",
    css=".output {background-color: lightyellow;}"
)

# Launch the Gradio interface
interface.launch(share=True)
