# database.py
from os import environ as env
from elasticsearch import AsyncElasticsearch
from elasticsearch.helpers import async_scan


ES_HOST = env.get("ELASTICSEARCH_HOST", "dsigpu06")
INDEX = env.get("ELASTICSEARCH_INDEX", "fewnerd_index")

async def initialize_es(es_host=ES_HOST, es_port=9200):
    """
    Initializes the asynchronous Elasticsearch client and creates the index if it does not exist.
    """
    es = AsyncElasticsearch([{"host": es_host, "port": es_port, "scheme": "http"}])
    exists = await es.indices.exists(index=INDEX)
    if not exists:
        await es.indices.create(index=INDEX,settings={"number_of_replicas": 0})
        print(f"Index `{INDEX}` created.")
    return es, INDEX

async def index_record(es, record):
    """
    Asynchronously indexes a record into Elasticsearch.
    """
    await es.index(index=INDEX, document=record)


# A helper to execute an Elasticsearch query
async def search_es(es, query, size=100):
    result = await es.search(index=INDEX, body={"query": query, "size": size})
    return result.get("hits", {}).get("hits", [])


async def get_filtered_dataset(es, fine_types, index=INDEX):
    """
    Retrieve the test dataset from Elasticsearch.
    Returns a list of data records filtered by fine types.
    """

    query = {
        "query": {
            "bool": {
                "must": [{"terms": {"fine_type": fine_types}}]
            }
        }
    }

    result = []
    async for doc in async_scan(es, index=index, query=query):
        result.append(doc)
    return result
# def get_fewnerd_dataset(es):
#     """
#     Retrieve the fewnerd dataset from Elasticsearch.
#     Each document should contain 'sentence', 'label', and 'fine_type' fields.
#     Computes the sentence embedding using compute_llm_output.
#     Returns a list of tuples: (embedding, label, fine_type)
#     """
#     result = es.search(index=INDEX, body={"query": {"match_all": {}}})
#     dataset = []
#     for hit in result["hits"]["hits"]:
#         source = hit["_source"]
#         sentence = source["sentence"]
#         label = source["label"]
#         fine_type = source["fine_type"]
#         embedding =  source["fine_type"]
#         embedding = embedding.squeeze(0)[-1]  # shape: [hidden_dim]
#         dataset.append((embedding, torch.tensor(label, dtype=torch.float), fine_type))
#     return dataset