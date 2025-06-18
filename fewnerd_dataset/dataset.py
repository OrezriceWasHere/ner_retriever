# dataset.py
import json
import uuid

import aiohttp
import aiofiles
import torch

DATASET_URL = "https://huggingface.co/datasets/Rosenberg/fewnerd/resolve/main/test-supervised.txt"

async def download_fewnerd_dataset():
    """
    Asynchronously downloads the fewnerd dataset.
    If the download fails, a fallback local file is used.
    """
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(DATASET_URL) as response:
                response.raise_for_status()
                data = await response.text()
                print("Dataset downloaded from remote URL.")
    except Exception as error:
        print("Error downloading dataset:", error)
        async with aiofiles.open("fewnerd_dataset.json", "r") as f:
            content = await f.read()
            data = json.loads(content)
        print("Dataset loaded from local file.")
    processed_data = process_dataset(data)
    return processed_data

def split_into_document(file_content):
    """
    Splits the dataset file content into documents.
    Each document is a list of non-empty lines.
    Documents are separated by empty lines.
    """
    documents = []
    current_doc = []
    for line in file_content.splitlines():
        if line.strip() == "":
            if current_doc:
                documents.append(current_doc)
                current_doc = []
        else:
            current_doc.append(line.strip())
    if current_doc:
        documents.append(current_doc)
    return documents

def space_when_necessary(prev_word, current_word):
    space = " "
    no_space = ""
    list_words_without_space = ["(", ")", "[", "]", "{", "}", ":", ";", ",", ".", "!", "?", "'", "\"", "`", "'s", "''",
                                "%"]

    if any((not prev_word,
            not current_word,
            current_word in list_words_without_space)):
        return no_space
    return space


def decide_word_tagging(tagging):
    if tagging == "O":
        return "O", "O"
    return tagging.split("-")

def create_embedding(text, indices):
    # Stub for embedding logic. Replace with actual LLM embedding code.
    return {}

def process_document(document):
    """
    Processes a document (a list of lines) into a list of entity records.
    Each line should contain a word and its tagging separated by a tab.
    """
    prev_word = None
    prev_tagging = None
    full_text = ""
    tagging_array = []
    # Generate a unique id for the document (could use uuid4)
    text_id = str(uuid.uuid4())
    for line in document:
        if "\t" not in line:
            continue
        word, tagging = line.split("\t")
        coarse_tag, fine_tag = decide_word_tagging(tagging)
        addition = space_when_necessary(prev_word, word) + word

        not_yet_entity = (prev_tagging in [None, "O"]) and tagging == "O"
        start_entity = (prev_tagging != tagging and tagging != "O") or (prev_tagging == "O" and tagging != "O")
        in_entity = prev_tagging == tagging and tagging != "O"
        end_entity = prev_tagging != "O" and tagging == "O"

        if not_yet_entity or end_entity:
            pass
        elif start_entity:
            tagging_array.append({
                "phrase": word,
                "coarse_type": coarse_tag,
                "fine_type": fine_tag,
                "index_start": len(full_text) + (1 if prev_word and len(word) != len(addition) else 0),
                "index_end": len(full_text) + len(addition),
            })
        elif in_entity and tagging_array:
            tagging_array[-1]["phrase"] += addition
            tagging_array[-1]["index_end"] += len(addition)

        full_text += addition
        prev_tagging = tagging
        prev_word = word

    for tagging in tagging_array:
        tagging["all_text"] = full_text
        tagging["text_id"] = text_id
        # Ensure the substring matches the phrase
        assert full_text[tagging["index_start"]:tagging["index_end"]] == tagging["phrase"]
        embedding = create_embedding(full_text, (tagging["index_start"], tagging["index_end"]))
        tagging["embedding"] = embedding

    return tagging_array

def process_dataset(file_content):
    """
    Processes the entire dataset file content.
    Splits it into documents and then processes each document.
    Returns a list of processed document records.
    """
    documents = split_into_document(file_content)
    processed_documents = []
    for document in documents:
        processed = process_document(document)
        processed_documents.extend(processed)
    return processed_documents

async def get_fewnerd_dataset(es):
    """
    Retrieve the fewnerd dataset from Elasticsearch.
    Each document should contain 'sentence', 'label', and 'fine_type' fields.
    Computes the sentence embedding using compute_llm_output.
    Returns a list of tuples: (embedding, label, fine_type)
    """
    result = await es.search(index="fewnerd", body={"query": {"match_all": {}}})
    dataset = []
    for hit in result["hits"]["hits"]:
        source = hit["_source"]
        label = source["label"]
        fine_type = source["fine_type"]
        embedding = source["embedding"]
        embedding = embedding.squeeze(0)[-1]  # shape: [hidden_dim]
        dataset.append((embedding, torch.tensor(label, dtype=torch.float), fine_type))
    return dataset

