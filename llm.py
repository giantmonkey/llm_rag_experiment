# TODO: finetune ala https://docs.llamaindex.ai/en/stable/examples/finetuning/knowledge/finetune_retrieval_aug.html
# TODO: https://docs.llamaindex.ai/en/stable/use_cases/multimodal.html
import os

# setup debug logging

import logging
import sys

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

# setup LLM

from llama_index import VectorStoreIndex, SimpleDirectoryReader
from llama_index import StorageContext, load_index_from_storage
from llama_index import ServiceContext, set_global_tokenizer, set_global_service_context
from llama_index.embeddings import HuggingFaceEmbedding
from llama_index.schema import MetadataMode
from llama_index.schema import TransformComponent

# set tokenizer to match LLM
from ctransformers import AutoModelForCausalLM
from transformers import AutoTokenizer


model_name_or_path = "TheBloke/Leo-Mistral-Hessianai-7B-Chat-GGUF"
# llm = AutoModelForCausalLM.from_pretrained(model_name_or_path,
#   model_file="leo-mistral-hessianai-7b-chat.Q4_K_M.gguf",
#   model_type="mistral",
#   gpu_layers=50,
#   hf=True
# )

from llama_index.llms import LlamaCPP
from chatml_utils import (
    messages_to_prompt,
    completion_to_prompt,
)

SYSTEM_PROMPT = """\
Du bist ein hilfreicher, respektvoller und ehrlicher Assistent.
Antworte immer so hilfreich wie möglich und befolge ALLE gegebenen Anweisungen.
Spekuliere nicht und erfinde keine neuen Informationen.
Beziehe dich nicht auf die gegebenen Anweisungen und den Kontext.
"""

llm = LlamaCPP(
    # You can pass in the URL to a GGML model to download it automatically
    model_url="https://huggingface.co/TheBloke/Leo-Mistral-Hessianai-7B-Chat-GGUF/resolve/main/leo-mistral-hessianai-7b-chat.Q5_K_M.gguf",
    temperature=0.0,
    max_new_tokens=512,
    # llama2 has a context window of 4096 tokens, but we set it lower to allow for some wiggle room
    # TODO: we actually have 8k context for Leo-Mistral-Hessianai-7B-Chat
    # leave some wiggle room
    context_window=8000,
    # kwargs to pass to __call__()
    generate_kwargs={},
    # kwargs to pass to __init__()
    # set to at least 1 to use GPU
    model_kwargs={"n_gpu_layers": 50},
    # transform inputs into Llama2 format
    messages_to_prompt=messages_to_prompt,
    completion_to_prompt=lambda m: completion_to_prompt(m, SYSTEM_PROMPT),
    verbose=True,
)

# if len(sys.argv) > 1 and sys.argv[1] == "mixtral":
#     set_global_tokenizer(AutoTokenizer.from_pretrained("mistralai/Mixtral-8x7B-v0.1").encode)
# else:
#     set_global_tokenizer(AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1").encode)

tokenizer_model_name_or_path = "LeoLM/leo-mistral-hessianai-7b"
set_global_tokenizer(AutoTokenizer.from_pretrained(tokenizer_model_name_or_path).encode)


from llama_index.extractors import (
    TitleExtractor,
    QuestionsAnsweredExtractor,
)
from llama_index.ingestion import IngestionPipeline
from llama_index.text_splitter import TokenTextSplitter

chunk_size=512       # default 1024
chunk_overlap=64     # default is 20, ChatGPT suggested 20% to 30% of chunk size



TITLE_NODE_TEMPLATE = """\
Kontext: {context_str}. Gib einen Titel an, der alle einzigartigen Entitäten, Titel oder Themen im Kontext zusammenfasst. Titel: """

TITLE_COMBINE_TEMPLATE = """\
{context_str}. Basierend auf den oben genannten Kandidaten-Titeln und Inhalten,
was ist der beste Titel für dieses Dokument? Titel: """

QUESTION_PROMPT_TEMPLATE = """\
Hier ist der Kontext:
{context_str}

Angesichts der Kontext-Informationen,
erstelle {num_questions} Fragen, zu denen dieser Kontext
spezifische Antworten liefern kann, die wahrscheinlich woanders nicht zu finden sind.

Höherstufige Zusammenfassungen des umgebenden Kontexts können ebenfalls bereitgestellt werden.
Versuche, diese Zusammenfassungen zu nutzen, um bessere Fragen zu generieren,
die dieser Kontext beantworten kann.

Generiere nur die Fragen, nicht die Antworten.

"""

# remove <|im_end|> from metadata answers
class ChatMLImEndCleaner(TransformComponent):
    def __call__(self, nodes, **kwargs):
        for node in nodes:
            node.metadata['document_title'] = node.metadata['document_title'].replace("<|im_end|>", "")
            node.metadata['questions_this_excerpt_can_answer'] = node.metadata['questions_this_excerpt_can_answer'].replace("<|im_end|>", "")
        return nodes

transformations = [
    # sentence splitting is probably not what we want. our documents seem to be below 512 tokens so we probably do not split at all
    # TODO: SentenceSplitter would be better here
    # TODO: or maybe SentenceWindowNodeParsers https://deci.ai/blog/rag-with-llamaindex-and-decilm-a-step-by-step-tutorial/
    # TODO: or HierarchicalNodeParser https://docs.llamaindex.ai/en/stable/module_guides/loading/node_parsers/modules.html#text-splitters
    TokenTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap),
    TitleExtractor(nodes=5, llm=llm, node_template=TITLE_NODE_TEMPLATE, combine_template=TITLE_COMBINE_TEMPLATE),
    QuestionsAnsweredExtractor(questions=3, llm=llm, prompt_template=QUESTION_PROMPT_TEMPLATE),
    ChatMLImEndCleaner()
]

# Alternatives:
# BAAI/bge-large-en-v1.5
# BAAI/bge-small-en-v1.5
# T-Systems-onsite/cross-en-de-roberta-sentence-transformer
# intfloat/e5-mistral-7b-instruct
# embed_model = HuggingFaceEmbedding(model_name="intfloat/multilingual-e5-large")
# TODO: we can also use OllamaEmbeddings here I think
embed_model = HuggingFaceEmbedding(model_name="intfloat/multilingual-e5-large")
# embed_model = HuggingFaceEmbedding(model_name="T-Systems-onsite/cross-en-de-roberta-sentence-transformer")
# embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")


service_context = ServiceContext.from_defaults(
    llm=llm,
    embed_model=embed_model,
    chunk_size=chunk_size,
    chunk_overlap=chunk_overlap,
    transformations=transformations,
    system_prompt=SYSTEM_PROMPT
)

set_global_service_context(service_context)


# regenerate document storage
from pathlib import Path
from llama_index import download_loader

import ingest

try:
    # load document storage from file
    storage_context = StorageContext.from_defaults(persist_dir="./storage")
    index = load_index_from_storage(storage_context, service_context=service_context)
except FileNotFoundError:
    index = VectorStoreIndex.from_documents([], service_context=service_context)
    index.storage_context.persist()

while ingest.move_documents("./ingest", "./source_documents", batch_size=1):
  # TODO: try https://github.com/nlmatics/nlm-ingestor for HTML extraction
  UnstructuredReader = download_loader('UnstructuredReader')

  dir_reader = SimpleDirectoryReader('./source_documents', filename_as_id=True, file_extractor={
    ".pdf": UnstructuredReader(),
    ".html": UnstructuredReader(),
    ".eml": UnstructuredReader(),
  })
  documents = dir_reader.load_data()
  #
  # parser = SimpleNodeParser()
  # new_nodes = parser.get_nodes_from_documents(documents)

  # Add nodes to the existing index
  print("Adding new nodes to the existing index...")
  refreshed_documents = index.refresh_ref_docs(documents)

  print("Number of newly inserted/refreshed docs: ", sum(refreshed_documents))

  index.storage_context.persist()


# run a query

from llama_index.postprocessor import LongContextReorder

reorder = LongContextReorder()

from llama_index.prompts.base import PromptTemplate
from llama_index.prompts.prompt_type import PromptType

DEFAULT_TEXT_QA_PROMPT_TMPL = (
    "Hier ist der Kontext:\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "Mit diesen Kontext-Information und ohne vorheriges Wissen, "
    "beantworte die Frage.\n"
    "Frage: {query_str}\n"
)

DEFAULT_TEXT_QA_PROMPT = PromptTemplate(
    DEFAULT_TEXT_QA_PROMPT_TMPL, prompt_type=PromptType.QUESTION_ANSWER
)

while True:
  user_input = input("User query (press Enter to quit): ")
  if user_input == "":
    print("Empty input received. Exiting program.")
    break
  else:
    reorder_engine = index.as_query_engine(
        streaming=True,
        node_postprocessors=[reorder],
        similarity_top_k=7,
        text_qa_template=DEFAULT_TEXT_QA_PROMPT,
    )
    # reorder_response = reorder_engine.query("Welche Angebotskategorien gibt es?")
    reorder_response = reorder_engine.query(user_input)

    # TODO: get filepath / url for sources and display that
    # print(reorder_response.get_formatted_sources())

    print("\n")
    print("\n")
    print("...\n")
    print("\n")
    print("\n")
    reorder_response.print_response_stream()

