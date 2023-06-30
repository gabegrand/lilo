"""
precompute_embeddings.py | Author: Maxine Liu.


Computes the embedding from domain and saves them to a json file in data/embedding.


"""

import json
import os

import openai
from openai.embeddings_utils import get_embedding

from src.experiment_iterator import ExperimentState
from src.task_loaders import TEST, TRAIN

openai.api_key = os.environ["OPENAI_API_KEY"]

from data.clevr.encoder import *
from data.clevr.grammar import *
from data.clevr.make_tasks import *
from data.compositional_graphics.encoder import *
from data.compositional_graphics.grammar import *
from data.compositional_graphics.make_tasks import *
from data.re2.encoder import *
from data.re2.grammar import *
from data.re2.make_tasks import *
from src.config_builder import build_config
from src.experiment_iterator import ExperimentState
from src.models.model_loaders import *
from src.models.seq2seq import *
from src.task_loaders import TEST, TRAIN
from src.utils import *

ENGINE_GPT_EMBEDDING = "text-embedding-ada-002"


def precompute_embeddings(domain):
    config_embedding = build_config(
        experiment_name="embedding",
        experiment_type="embedding",
        domain=domain,
    )

    experiment_state = ExperimentState(config_embedding)

    all_language = []
    all_task_id = []
    for split in (TRAIN, TEST):
        all_language += experiment_state.get_language_for_ids(
            task_split=split, task_ids=ALL
        )
        all_task_id += experiment_state.get_tasks_for_ids(
            task_split=split, task_ids=ALL
        )
    all_language = [x[0] for x in all_language]
    all_task_id = [x.name for x in all_task_id]
    embedding_dict = {
        task_id: get_embedding(language, engine=ENGINE_GPT_EMBEDDING)
        for task_id, language in zip(all_task_id, all_language)
    }

    domain = experiment_state.config["metadata"]["tasks_loader"]
    embedding_filepath = os.path.join("data", "embeddings", f"{domain}_embeddings.json")
    with open(embedding_filepath, "w") as file:
        json.dump(embedding_dict, file)
