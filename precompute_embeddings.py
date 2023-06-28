import json
import os

import openai
from openai.embeddings_utils import get_embedding

from src.experiment_iterator import ExperimentState

# from src.models.gpt_solver import *
from src.task_loaders import TEST, TRAIN

openai.api_key = os.environ["OPENAI_API_KEY"]


def precompute_embeddings(experiment_state):

    all_language = []
    all_task_id = []
    for split in (TRAIN, TEST):
        all_language += experiment_state.get_language_for_ids(
            task_split=split, task_ids=ExperimentState.ALL
        )
        all_task_id += experiment_state.get_tasks_for_ids(
            task_split=split, task_ids=ExperimentState.ALL
        )
    all_language = [x[0] for x in all_language]
    all_task_id = [x.name for x in all_task_id]
    # embedding_dict = {language: get_embedding(language, engine = "text-embedding-ada-002") for language in all_language}
    embedding_dict = {
        task_id: get_embedding(language, engine="text-embedding-ada-002")
        for task_id, language in zip(all_task_id, all_language)
    }
    json_embedding = json.dumps(embedding_dict)

    domain = experiment_state.config["metadata"]["tasks_loader"]
    embedding_directory = "data/embeddings/" + domain + "_embeddings.json"
    with open(embedding_directory, "w") as file:
        file.write(json_embedding)
