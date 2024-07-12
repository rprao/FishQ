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
Settings.llm = HuggingFaceInferenceAPI(model_name = "meta-llama/Meta-Llama-3-8B-Instruct")

#Define the embedding model
embed_model = HuggingFaceEmbedding(model_name="thenlper/gte-large")
Settings.embed_model = embed_model

#
PERSIST_DIR = "./storage"
storage_context = StorageContext.from_defaults(persist_dir = PERSIST_DIR)
index = load_index_from_storage(storage_context,index_id = "vector_index")


def gradio_interface(prompt):
    try:
        index = load_index_from_storage(storage_context, index_id="vector_index")
        retriever = VectorIndexRetriever(
            index=index,
            similarity_top_k=2,
        )
        response_synthesizer = get_response_synthesizer()
        query_engine = RetrieverQueryEngine(
            retriever=retriever,
            response_synthesizer=response_synthesizer,
            node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=0.7)]            
        )
        response = query_engine.query(prompt)
        return response

    except Exception as e:
        return f"An error occurred: {str(e)}"

with gr.Blocks(theme=gr.themes.Soft()) as interface:
    gr.Markdown("# FishQ-RAG\nRAG Agent for 2021 CPCSEA Fish Experimentation Guidelines\n\nThe guidelines are intended to provide information on the humane procedures for holding, handling, and sampling fish for experimental, research, or teaching purposes.")
    response = gr.Textbox(lines=10,label="Response",show_copy_button=True)
    query = gr.Textbox(lines=2, placeholder="Ask me about guidelines here... Click submit or Shift + Enter.", label="Query")

    gr.Examples(
        examples=[
            ["What is CPCSEA?"],
            ["Give me guidelines for Zebrafish."],
            ["Write a compact email about the guidelines with their background information."]
        ],
        inputs=query
    )

    submit_btn = gr.Button("Submit")
    query.submit(gradio_interface, query, response)
    submit_btn.click(gradio_interface, inputs=query, outputs=response)

interface.launch(share=True)

