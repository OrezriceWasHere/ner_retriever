# main.py
import asyncio
from dataset import download_fewnerd_dataset
from llm import compute_llm_output, token_indices_given_text_indices
from database import initialize_es, index_record
from tqdm import tqdm
from elasticsearch.helpers import async_bulk

BATCH_SIZE = 64

async def download_dataset():
    print("Downloading dataset...")
    return await download_fewnerd_dataset()

def process_batch(batch):
    texts = [record.get("all_text", "") for record in batch]
    outputs = compute_llm_output(texts)
    processed_records = []
    for record, output, text in zip(batch, outputs, texts):
        _, last_token_idx = token_indices_given_text_indices(text, (record["index_start"], record["index_end"]))
        record["llm_output"] = output[last_token_idx].tolist()  # Convert tensor to list
        processed_records.append(record)
    return processed_records

async def index_records(records, es):
    actions = [
        {
            "_index": "fewnerd_index",
            "_source": record
        }
        for record in records
    ]
    await async_bulk(es, actions)


async def load_dataset_into_database():
    dataset = await download_dataset()
    es, index_name = await initialize_es()

    for i in tqdm(range(0, len(dataset), BATCH_SIZE), desc="Processing batches"):
        batch = dataset[i:i+BATCH_SIZE]
        processed_batch = process_batch(batch)
        await index_records(processed_batch, es)
        del batch, processed_batch  # Free memory

    print("Dataset processed and indexed in Elasticsearch.")

if __name__ == "__main__":
    asyncio.run(load_dataset_into_database())