import weaviate
import platform
import pandas as pd
import json

print(platform.python_version())

auth_config = weaviate.AuthApiKey(
    api_key="XVJDQkobAb0jPRoOacg89LftUnVOJCbczDtk"
)  # Replace w/ your Weaviate instance API key

# Instantiate the client
client = weaviate.Client(
    url="https://my-sandbox-cluster-4411ujj0.weaviate.network",  # Replace w/ your Weaviate cluster URL
    auth_client_secret=auth_config,
    additional_headers={
        "X-HuggingFace-Api-Key": "????",  # Replace with your HF key
        "X-OpenAI-Api-Key": "????",  # Replace with your openAI key
    },
)

if_ready = client.is_ready()  # check if everything works
# print(if_ready)

df = pd.read_csv("jeopardy_questions.csv", nrows=100)

###################################Data Class Object########################################
# the name of collection of data in the vector space : class_obj
# the properties of an object including property name and data type
# vectorize: the model that generates the embeddings for text obj : text2vec Weaviet
# moduleConfig: details of the used modules
class_obj = {
    "class": "JeopardyQuestion",
    "properties": [
        {
            "name": "category",
            "dataType": ["text"],
        },
        {
            "name": "question",
            "dataType": ["text"],
        },
        {
            "name": "answer",
            "dataType": ["text"],
        },
    ],
    "vectorizer": "text2vec-huggingface",
    "moduleConfig": {
        "text2vec-huggingface": {
            "vectorizeClassName": False,
            "model": "sentence-transformers/all-MiniLM-L6-v2",
        },
    },
}
##########################################################################################
# rune the line below only once to creat a class
# client.schema.create_class(class_obj)
# to view the created class print the line below
client.schema.get("JeopardyQuestion")

#########################################################################################

# Importing data into Weaviet -> populate the empyty Weaviet with a dataset (upserting)
# batchify

from weaviate.util import generate_uuid5

with client.batch(
    batch_size=200,  # specify batch size
    num_workers=2,  # parallelize the process
) as batch:
    for _, row in df.iterrows():
        question_object = {
            "category": row.category,
            "question": row.question,
            "answer": row.answer,
        }

        batch.add_data_object(
            question_object,
            class_name="JeopardyQuestion",
            uuid=generate_uuid5(question_object),
        )


# sanity check: print the line below
client.query.aggregate("JeopardyQuestion").with_meta_count().do()


# Query the Weaviate Vector Database : to retrieve objects , you need to query the Weaviate vector database with "get()"
# client.query.get(<Class>, [<properties>]).<arguments>.do()
# Class : the name of the class of objects to be retrieved. e.g.: "JeopardyQuestion"
# properties : specify the properties of an object to be retrieved. e.g.: "category", "question", "answer"
# arguments : specifies the search criteria to retrieve the objects, such as limits and aggregations.

res = (
    client.query.get("JeopardyQuestion", ["category", "question", "answer"])
    .with_additional(["id", "vector"])
    .with_limit(2)
    .do()
)
# sanity check for len vector database:  e.g. using Huggingface model (sentence-transformer ) -> 384

print(
    len(
        json.loads(json.dumps(res, indent=4))["data"]["Get"]["JeopardyQuestion"][0][
            "_additional"
        ]["vector"]
    )
)

# Vector search
# retriving Jeopardy questions related to a specific concept

res = (
    client.query.get("JeopardyQuestion", ["category", "question", "answer"])
    .with_near_text({"concepts": "animals"})
    .with_limit(2)
    .do()
)

# sanity check: print the line below
json.dumps(res, indent=4)

# Question answering
# the module config shoulld chang to LLMs e.g. Llama2 or OpenAI
# Module settings
# "moduleConfig": {
#     "text2vec-openai": {
#       ...
#     },
#     "qna-openai": {
#       "model": "text-davinci-002"
#     }
# },

ask = {
    "question": "Which animal was mentioned in the title of the Aesop fable?",
    "properties": ["answer"],
}

res = (
    client.query.get(
        "JeopardyQuestion",
        ["question", "_additional {answer {hasAnswer property result} }"],
    )
    .with_ask(ask)
    .with_limit(1)
    .do()
)

# sanity check: print the line below
json.dumps(res, indent=4)

# Generative search -> using LLMs -> transforming the data before returning the search result
# Module settings
# "moduleConfig": {
#     "text2vec-openai": {
#       ...
#     },
#     "generative-openai": {
#       "model": "gpt-3.5-turbo"
#     }
# },


res = (
    client.query.get("JeopardyQuestion", ["question", "answer"])
    .with_near_text({"concepts": ["animals"]})
    .with_limit(1)
    .with_generate(single_prompt="Generate a question to which the answer is {answer}")
    .do()
)
