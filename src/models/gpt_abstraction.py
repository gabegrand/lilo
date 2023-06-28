"""
gpt_abstraction.py | Author : Maxine Liu.

Queries Codex to generate abstraction for functions.
"""
from typing import Dict, List

import src.models.model_loaders as model_loaders
from dreamcoder.type import *
from src.experiment_iterator import SKIPPED_MODEL_FN
from src.models.gpt_base import *
from src.models.laps_grammar import LAPSGrammar
from src.models.library_namer import GPTLibraryNamer, LibraryNamerPrompt
from src.task_loaders import TRAIN

ModelRegistry = model_loaders.ModelLoaderRegistries[model_loaders.LIBRARY_ABSTRACTION]


class LibraryAbstractionPrompt(LibraryNamerPrompt):

    TEXT_ABSTRACTION_HEADER = "You are about to undertake a task focused on abstraction learning. The objective is to develop reusable, compressed functions derived from existing ones. These new functions will be utilized to solve specific tasks."
    TEXT_FUNCTION_HEADER = (
        "To get started, consider the functions provided in the library:"
    )
    TEXT_TASKS_HEADER = "Now, here are the tasks that need to be tackled:"

    def __init__(
        self,
        abstraction_definitions: Dict,
        task_examples: List[Dict],
        line_separator: str = DEFAULT_LINE_SEPARATOR,
    ):
        super().__init__(
            abstraction_definitions=abstraction_definitions,
            abstraction_target=None,
            usage_examples=task_examples,
            line_separator=line_separator,
        )
        self.task_examples = task_examples

    def _build_abstraction_header(self):
        text_list = [
            self.TEXT_ABSTRACTION_HEADER
            + self.TEXT_FUNCTION_HEADER
            + self.line_separator
        ]
        for abstraction in self.abstraction_definitions.keys():
            text_list += [self._fn_docstring(abstraction) + self.line_separator]
        return self.line_separator.join(text_list)

    def _build_task_prompt(self):
        text_list = [self.TEXT_TASKS_HEADER + self.line_separator]
        for task in self.task_examples:
            text_list += [
                self.DEFAULT_PREFIX_LANGUAGE + task["language"] + self.line_separator
            ]
        text_list += [self.make_abstraction_footer()]
        return self.line_separator.join(text_list)

    def make_abstraction_footer(self):
        return (
            f"Your challenge is to author a compact, reusable function based on the functions in the library. This function should be encoded in the following JSON format."
            + "\n"
            f"It should be unique (not existing in the function library above)." + "\n"
            f"If you cannot come up with a good function, return nothing." + "\n\n"
            "{" + "\n"
            f'    "function_name": TODO,' + "\n"
            f'    "function_expression": TODO,' + "\n"
            f'    "function_description": TODO' + "\n"
            "}"
        )

    def to_message_list(self):
        message_list = [self.chat_message(self._build_abstraction_header())]
        message_list += [self.chat_message(self._build_task_prompt())]
        return message_list


###TODO


@ModelRegistry.register
class GPTLibraryAbstraction(GPTLibraryNamer):
    name = "gpt_library_abstraction"

    results_file = "gpt_library_abstraction_results.json"

    @staticmethod
    def load_model(experiment_state, **kwargs):
        return GPTLibraryAbstraction(experiment_state=experiment_state, **kwargs)

    def __init__(self, experiment_state=None, engine=None):
        super().__init__(experiment_state=experiment_state, engine=engine)

    def generate_abstraction(
        self,
        experiment_state,
        task_split: str,
        task_batch_ids: list,
        # Querying
        best_of: int = 1,
        n_samples_per_abstraction: int = 5,
        top_p: float = 0.1,
        max_tokens: int = 256,
        n_function_generated: int = 10,
        # Prompt construction
        n_task_examples: int = 10,
        # Resume from prior runs
        resume_strategy: str = None,
        # Utilities
        verbose: bool = True,
    ):
        assert task_split == TRAIN
        grammar = experiment_state.models[model_loaders.GRAMMAR]

        # Optional load from results JSON
        if self._maybe_load_from_checkpoint(
            experiment_state, task_split, resume_strategy
        ):
            return {
                SKIPPED_MODEL_FN: True,
            }

        abstraction_definitions = self._get_abstraction_definitions(experiment_state)
        gpt_abstraction_library = {}

        for i in range(n_function_generated):
            # Update to have latest names
            abstraction_definitions = self._get_abstraction_definitions(
                experiment_state
            )
            task_examples = self._get_task_examples(experiment_state, n_task_examples)
            prompt = LibraryAbstractionPrompt(
                abstraction_definitions=abstraction_definitions,
                task_examples=task_examples,
            )

            if verbose:
                print(prompt)

            # Query
            completion = self.query_completion(
                prompt,
                n_samples=n_samples_per_abstraction,
                top_p=top_p,
                max_tokens=max_tokens,
                stop=None,
            )

            # Parse response
            parse_results = self._parse_completion(completion)
            selected_result = self._select_result(parse_results)

            if verbose:
                print(f"GPT ({self.ENGINE}) completion:")
                print(json.dumps(parse_results, indent=4))

            # Update function name
            if selected_result is not None:
                readable_name = selected_result["data"]["function_name"]
                function_expression = selected_result["data"]["function_expression"]
                grammar._add_base_primitive(function_expression)
                grammar.set_function_name(
                    str(function_expression),
                    name_class=LAPSGrammar.HUMAN_READABLE,
                    name=readable_name,
                )
                grammar.set_function_description(
                    name=str(function_expression),
                    description=selected_result["data"]["function_description"],
                )

                gpt_abstraction_library[str(function_expression)] = selected_result[
                    "data"
                ]
                gpt_abstraction_library[str(function_expression)][
                    "task_examples"
                ] = task_examples

                print(f"✅ Successfully created {readable_name}:{function_expression}")
                print(json.dumps(selected_result, indent=4))

            else:
                gpt_abstraction_library[str(function_expression)] = None
                print(f"❌ Failed to create a function")

        n_abstractions_generated = len(
            [x for x in gpt_abstraction_library.values() if x is not None]
        )

        # TODO: Log/save outputs
        print("-" * 12)
        print(
            f"Completed library abstraction learning: {n_abstractions_generated} / {len(gpt_abstraction_library)} abstractions successfully generated."
        )
        print(json.dumps(gpt_abstraction_library, indent=4))

        results = {
            "params": {
                "best_of": best_of,
                "n_samples_per_abstraction": n_samples_per_abstraction,
                "n_task_examples": n_task_examples,
                "top_p": top_p,
                "max_tokens": max_tokens,
                "n_function_generated": n_function_generated,
            },
            "summary": {
                "n_abstractions_generated": n_abstractions_generated,
                "n_abstractions_total": len(gpt_abstraction_library),
            },
            "abstractions": gpt_abstraction_library,
        }

        # Save results to file
        results_filepath = os.path.join(
            os.getcwd(),
            experiment_state.get_checkpoint_directory(),
            task_split,
            self.results_file,
        )
        os.makedirs(os.path.dirname(results_filepath), exist_ok=True)
        with open(results_filepath, "w") as f:
            json.dump(results, f, indent=4)
        if verbose:
            print(f"Wrote results: {results_filepath}")

    # Question1: not sure if need to override this method
    def _maybe_load_from_checkpoint(
        self, experiment_state, task_split, resume_strategy
    ):
        if (resume_strategy == "first" and experiment_state.is_first_iteration()) or (
            resume_strategy == "every"
        ):
            # If RESUME_CHECKPOINT_DIRECTORY not defined, default to self checkpoint directory
            results_filepath_ext = os.path.join(
                os.getcwd(),
                experiment_state.get_checkpoint_directory_maybe_resume(),
                task_split,
                self.results_file,
            )
            if os.path.exists(results_filepath_ext):
                with open(results_filepath_ext, "r") as f:
                    results_json = json.load(f)

                # Update experiment state from file
                grammar = experiment_state.models[model_loaders.GRAMMAR]

                for abstraction, data in results_json["abstractions"].items():
                    if data is not None:
                        grammar.set_function_name(
                            str(abstraction),
                            name_class=LAPSGrammar.HUMAN_READABLE,
                            name=data["readable_name"],
                        )
                        grammar.set_function_description(
                            name=str(abstraction),
                            description=data["description"],
                        )

                # Copy external results file to checkpoint directory
                results_filepath = os.path.join(
                    os.getcwd(),
                    experiment_state.get_checkpoint_directory(),
                    task_split,
                    self.results_file,
                )
                os.makedirs(os.path.dirname(results_filepath), exist_ok=True)
                with open(results_filepath, "w") as f:
                    json.dump(results_json, f, indent=4)

                print(f"{self.name}: Loaded results from {results_filepath_ext}")
                return True
            else:
                print(f"{self.name}: Results not found at {results_filepath_ext}")
                # if experiment_state.is_first_iteration():
                #     raise ValueError("Unable to resume first iteration.")
                return False

    # Question2: how to make sure all new functions go to abstraction?
    def _get_abstraction_definitions(self, experiment_state):
        grammar = experiment_state.models[model_loaders.GRAMMAR]
        abstractions = [p for p in grammar.primitives if not p.isInvented]

        abstraction_definitions = {}
        # for abstraction in sorted(abstractions, key=lambda p: str(p)):
        for abstraction in abstractions:
            abstraction_definitions[abstraction] = self._get_abstraction_definition(
                experiment_state, abstraction
            )

        return abstraction_definitions

    # Question3: I want to delete this part because I think there is no need to override
    # but if I change is to not isInvented, will it affect any methods inside this function
    def _get_abstraction_definition(self, experiment_state, abstraction):
        grammar = experiment_state.models[model_loaders.GRAMMAR]
        fn_name = grammar.get_name(
            str(abstraction),
            name_classes=[
                LAPSGrammar.HUMAN_READABLE,
                LAPSGrammar.NUMERIC_FUNCTION_NAMES,
            ],
        )
        fn_body = str(
            grammar.show_program(
                str(abstraction)[
                    1:
                ],  # Remove leading `#` so that any inlined abstractions are replaced with their fn_name
                name_classes=[
                    LAPSGrammar.HUMAN_READABLE,
                    LAPSGrammar.NUMERIC_FUNCTION_NAMES,
                ],
            )
        )
        fn_description = grammar.get_function_description(abstraction)
        return {
            "fn_name": fn_name,
            "fn_body": fn_body,
            "fn_type": abstraction.infer(),
            "fn_description": fn_description,
        }

    def _get_task_examples(self, experiment_state, n_task_examples):
        rng = experiment_state.metadata[RANDOM_GENERATOR]
        experiment_state.models[model_loaders.GRAMMAR]

        tasks = list(experiment_state.task_frontiers[TRAIN].keys())
        rng.shuffle(tasks)

        task_examples = []

        for task in tasks:
            experiment_state.task_frontiers[TRAIN][task]
            task_language = rng.choice(
                experiment_state.get_language_for_ids(TRAIN, [task.name])[0]
            )
            task_examples.append(task_language)
            # Question4: What is this for? Can I delete it?
            # for e in frontier.entries:
            #     usage_examples += [
            #         {
            #             "task_name": task.name,
            #             "program": grammar.show_program(
            #                 e.program,
            #                 name_classes=[
            #                     LAPSGrammar.HUMAN_READABLE,
            #                     LAPSGrammar.NUMERIC_FUNCTION_NAMES,
            #                 ],
            #                 debug=True,
            #             ),
            #             "language": task_language,
            #         }
            #     ]

        if len(task_examples) == n_task_examples:
            return task_examples

    def _parse_completion(self, completion):
        parse_results = []
        for choice in completion["choices"]:
            try:
                data = json.loads(choice["text"])
            except:
                parse_results.append(
                    {
                        "index": choice["index"],
                        "text": choice["text"],
                        "valid": False,
                        "error": GPTLibraryAbstraction.ERROR_JSON,
                    }
                )
                continue

            if not (
                ("function_name" in data)
                and ("function_expression" in data)
                and ("function_description" in data)
            ):
                parse_results.append(
                    {
                        "index": choice["index"],
                        "text": choice["text"],
                        "valid": False,
                        "error": GPTLibraryAbstraction.ERROR_MISSING_FIELD,
                    }
                )
                continue

            if not data["function_name"]:
                parse_results.append(
                    {
                        "index": choice["index"],
                        "text": choice["text"],
                        "valid": False,
                        "error": GPTLibraryAbstraction.ERROR_NULL_NAME,
                    }
                )
                continue

            parse_results.append(
                {
                    "index": choice["index"],
                    "text": choice["text"],
                    "valid": True,
                    "data": data,
                }
            )
        return parse_results