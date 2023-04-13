"""
gpt_solver.py | Author: Gabe Grand.

Queries GPT to solve tasks.

"""
from collections import defaultdict

from openai.error import InvalidRequestError
from openai.openai_object import OpenAIObject

import src.models.model_loaders as model_loaders
from src.experiment_iterator import RANDOM_GENERATOR
from src.models.gpt_base import DEFAULT_LINE_SEPARATOR, GPTBase, Prompt
from src.models.laps_grammar import LAPSGrammar
from src.models.sample_generator import GPTSampleGenerator
from src.task_loaders import LANGUAGE, PROGRAMS, TRAIN

ModelRegistry = model_loaders.ModelLoaderRegistries[model_loaders.GPT_SOLVER]


@ModelRegistry.register
class GPTSolver(GPTSampleGenerator, model_loaders.ModelLoader):
    name = "gpt_solver"

    @staticmethod
    def load_model(experiment_state, **kwargs):
        return GPTSolver(experiment_state=experiment_state, **kwargs)

    def __init__(self, experiment_state=None):
        super().__init__()

    def infer_programs_for_tasks(
        self,
        experiment_state,
        task_split: str,
        task_batch_ids: list,
        # Sampling
        n_samples_per_query: int = None,
        n_queries_per_task: int = None,
        max_retries: int = None,
        # Prompt construction
        body_task_types: list = [LANGUAGE, PROGRAMS],
        final_task_types: list = [LANGUAGE],
        function_name_classes: list = [LAPSGrammar.DEFAULT_FUNCTION_NAMES],
        prepend_dsl_description: bool = False,
        line_separator: str = DEFAULT_LINE_SEPARATOR,
        # GPT parameters
        temperature: float = 0.40,
        max_tokens_completion_beta: float = 2.0,
        engine: str = GPTBase.DEFAULT_ENGINE,
    ):
        results_by_task = defaultdict(list)

        for task_id in task_batch_ids:
            for query_i in range(n_queries_per_task):

                prompt = self.construct_initial_prompt(
                    experiment_state=experiment_state,
                    task_split=task_split,
                    task_id=task_id,
                    body_task_types=body_task_types,
                    final_task_types=final_task_types,
                    function_name_classes=function_name_classes,
                    prepend_dsl_description=prepend_dsl_description,
                    line_separator=line_separator,
                    max_tokens_completion_beta=max_tokens_completion_beta,
                    verbose=False,
                )

                max_retries = len(prompt.body_task_data)
                for retry_i in range(max_retries):
                    if retry_i > 0:
                        prompt.remove_last_body_task()
                        print(
                            f"Retry ({retry_i} / {max_retries}): Prompt reduced to {len(prompt)} tasks."
                        )

                    token_stats = self.get_token_stats(
                        prompt=prompt,
                        max_tokens_completion_beta=max_tokens_completion_beta,
                    )

                    completion = self.query_completion(
                        prompt.serialize(),
                        n_samples=n_samples_per_query,
                        temperature=temperature,
                        max_tokens=token_stats["max_tokens_completion"],
                        engine=engine,
                        line_separator=line_separator,
                    )

                    if not isinstance(completion, OpenAIObject):
                        if isinstance(completion, InvalidRequestError):
                            if retry_i >= max_retries - 1:
                                raise ValueError(f"Max retries {max_retries} exceeded.")
                            continue
                        else:
                            raise ValueError(
                                f"Unexpected completion type: {type(completion)}"
                            )

                    valid_request_types = self.get_valid_request_types(
                        experiment_state, task_split, task_batch_ids
                    )
                    parse_results = self.parse_completion(
                        completion,
                        experiment_state=experiment_state,
                        task_split=task_split,
                        task_ids=task_batch_ids,
                        valid_request_types=valid_request_types,
                        function_name_classes=function_name_classes,
                        evaluate_samples=True,
                        compute_likelihoods=True,
                        verbose=True,
                    )

                    results_by_task[task_id].append(parse_results)

    def construct_initial_prompt(
        self,
        experiment_state,
        task_split,
        task_id,
        body_task_types,
        final_task_types,
        function_name_classes,
        prepend_dsl_description,
        line_separator,
        max_tokens_completion_beta,
        verbose,
    ):
        rng = experiment_state.metadata[RANDOM_GENERATOR]

        non_empty_task_ids = [
            f.task.name
            for f in experiment_state.get_non_empty_frontiers_for_split(TRAIN)
        ]

        # Random ordering of the body tasks
        body_task_ids = list(rng.permutation(non_empty_task_ids))
        if len(body_task_ids) < 2:
            raise ValueError(
                "At least 2 tasks must have non-empty frontiers to construct a prompt."
            )

        prompt = None
        for body_task_i in range(len(body_task_ids)):
            body_task_ids_for_prompt = body_task_ids[: body_task_i + 1]
            prompt_i = Prompt(
                experiment_state=experiment_state,
                body_task_ids=body_task_ids_for_prompt,
                final_task_id=task_id,
                body_task_types=body_task_types,
                final_task_types=final_task_types,
                final_task_split=task_split,
                function_name_classes=function_name_classes,
                prepend_dsl_description=prepend_dsl_description,
                line_separator=line_separator,
            )

            # Estimate token budgets
            token_stats = self.get_token_stats(
                prompt=prompt_i, max_tokens_completion_beta=max_tokens_completion_beta
            )

            if token_stats["token_count_prompt"] <= token_stats["max_tokens_prompt"]:
                prompt = prompt_i
                if verbose:
                    print(
                        f"Prompt construction ({body_task_i+1} / {len(body_task_ids)}): {token_stats['token_count_prompt']} (prompt; max {token_stats['max_tokens_prompt']}) + {token_stats['max_tokens_completion']} (completion allocation) = {token_stats['token_count_prompt'] + token_stats['max_tokens_completion']} tokens"
                    )
            else:
                break

        if prompt is None:
            raise ValueError(f"Failed to construct prompt.")
        assert body_task_i > 0

        return prompt
