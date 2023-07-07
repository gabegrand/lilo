"""
gpt_abstraction.py | Author : Maxine Liu.

Queries Codex to generate abstraction for functions.
"""
from typing import Dict, List

import src.models.model_loaders as model_loaders
from dreamcoder.program import Program
from dreamcoder.type import *
from src.experiment_iterator import SKIPPED_MODEL_FN
from src.models.gpt_base import *
from src.models.laps_grammar import LAPSGrammar
from src.models.library_namer import GPTLibraryNamer, LibraryNamerPrompt
from src.task_loaders import TRAIN

ModelRegistry = model_loaders.ModelLoaderRegistries[model_loaders.LIBRARY_LEARNER]


class LibraryAbstractionPrompt(LibraryNamerPrompt):

    TEXT_ABSTRACTION_HEADER = "You are about to undertake a task focused on abstraction learning. The objective is to develop reusable, compressed functions derived from existing ones. These new functions will be utilized to solve specific tasks."
    TEXT_FUNCTION_HEADER = (
        "To get started, consider the functions provided in the library:"
    )

    TEXT_PROGRAM_HEADER = (
        "Here are some examples of how to use the functions to solve tasks"
    )
    TEXT_TASKS_HEADER = "Now, here are the tasks that need to be tackled:"

    # provide some program examples
    # before providing tasks, provide some solved programs

    def __init__(
        self,
        abstraction_definitions: Dict,
        task_examples: List[Dict],
        program_examples: List[Dict],
        line_separator: str = DEFAULT_LINE_SEPARATOR,
    ):
        super().__init__(
            abstraction_definitions=abstraction_definitions,
            abstraction_target=None,
            usage_examples=task_examples,
            line_separator=line_separator,
        )
        self.task_examples = task_examples
        self.program_examples = program_examples

    def _build_abstraction_header(self):
        text_list = [
            self.TEXT_ABSTRACTION_HEADER
            + self.TEXT_FUNCTION_HEADER
            + self.line_separator
        ]
        for abstraction in self.abstraction_definitions.keys():
            text_list += [self._fn_docstring(abstraction) + self.line_separator]

        for program_example in self.program_examples:
            text_list += [
                self.DEFAULT_PREFIX_LANGUAGE
                + program_example["language"]
                + self.line_separator
                + self.DEFAULT_PREFIX_PROGRAM
                + program_example["program"]
                + self.line_separator
            ]

        return self.line_separator.join(text_list)

    def _build_task_prompt(self):
        text_list = [self.TEXT_TASKS_HEADER + self.line_separator]

        for task in self.task_examples:
            text_list += [self.DEFAULT_PREFIX_LANGUAGE + task + self.line_separator]
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


@ModelRegistry.register
class GPTLibraryLearner(GPTLibraryNamer):
    name = "gpt_library_learner"

    results_file = "gpt_library_abstraction_results.json"
    prompt_file = "gpt_library_learner_prompts.json"

    ERROR_PARSE = "parse"
    ERROR_INFER = "infer"

    @staticmethod
    def load_model(experiment_state, **kwargs):
        return GPTLibraryLearner(experiment_state=experiment_state, **kwargs)

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
        n_program_examples: int = 10,
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

        grammar = LAPSGrammar(
            logVariable=grammar.logVariable,  # TODO: Renormalize logVariable
            productions=grammar.productions,
            continuationType=grammar.continuationType,
            initialize_parameters_from_grammar=grammar,
        )

        prompt_filepath = os.path.join(
            os.getcwd(),
            experiment_state.get_checkpoint_directory(),
            task_split,
            self.prompt_file,
        )

        for i in range(n_function_generated):
            # Update to have latest names
            abstraction_definitions = self._get_abstraction_definitions(
                experiment_state
            )
            task_examples = self._get_task_examples(experiment_state, n_task_examples)

            program_examples = self._get_program_examples(
                experiment_state, n_program_examples
            )

            prompt = LibraryAbstractionPrompt(
                abstraction_definitions=abstraction_definitions,
                task_examples=task_examples,
                program_examples=program_examples,
            )

            if verbose:
                with open(prompt_filepath, "w") as f:
                    json.dump(str(prompt), f)
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
            parse_results = self._parse_completion(
                completion, experiment_state, verbose
            )
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
                print(f"❌ Failed to create a function")

        n_abstractions_generated = len(
            [x for x in gpt_abstraction_library.values() if x is not None]
        )

        # TODO: Log/save outputs
        print("-" * 12)
        print(
            f"Completed library abstraction learning: {n_abstractions_generated} / {n_function_generated} abstractions successfully generated."
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
                        grammar._add_base_primitive(abstraction)
                        grammar.set_function_name(
                            str(abstraction),
                            name_class=LAPSGrammar.HUMAN_READABLE,
                            name=data["function_name"],
                        )
                        grammar.set_function_description(
                            name=str(abstraction),
                            description=data["function_description"],
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

    def _get_abstraction_definitions(self, experiment_state):
        grammar = experiment_state.models[model_loaders.GRAMMAR]
        abstractions = [p for p in grammar.primitives]

        abstraction_definitions = {}
        # for abstraction in sorted(abstractions, key=lambda p: str(p)):
        for abstraction in abstractions:
            abstraction_definitions[abstraction] = self._get_abstraction_definition(
                experiment_state, abstraction
            )

        return abstraction_definitions

    def _get_abstraction_definition(self, experiment_state, abstraction):
        grammar = experiment_state.models[model_loaders.GRAMMAR]
        fn_name = grammar.get_name(
            str(abstraction),
            name_classes=[
                LAPSGrammar.HUMAN_READABLE,
                LAPSGrammar.NUMERIC_FUNCTION_NAMES,
            ],
        )

        abstraction_str = str(abstraction)
        if abstraction.isInvented:
            # Remove leading `#` so that any inlined abstractions are replaced with their fn_name
            abstraction_str = abstraction_str[1:]

            fn_body = str(
                grammar.show_program(
                    abstraction_str,
                    name_classes=[
                        LAPSGrammar.HUMAN_READABLE,
                        LAPSGrammar.NUMERIC_FUNCTION_NAMES,
                    ],
                )
            )
        else:
            fn_body = str(abstraction)
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
            # now its getting all tasks. I want only unsolved tasks
            frontier = experiment_state.task_frontiers[TRAIN][task]
            if frontier.empty:
                task_language = rng.choice(
                    experiment_state.get_language_for_ids(TRAIN, [task.name])[0]
                )
                task_examples.append(task_language)
                if len(task_examples) == n_task_examples:
                    return task_examples

    def _get_program_examples(self, experiment_state, n_program_examples):
        rng = experiment_state.metadata[RANDOM_GENERATOR]
        grammar = experiment_state.models[model_loaders.GRAMMAR]

        tasks = list(experiment_state.task_frontiers[TRAIN].keys())
        rng.shuffle(tasks)

        program_examples = []

        for task in tasks:
            frontier = experiment_state.task_frontiers[TRAIN][task]
            task_language = rng.choice(
                experiment_state.get_language_for_ids(TRAIN, [task.name])[0]
            )
            for e in frontier.entries:
                program_examples += [
                    {
                        "task_name": task.name,
                        "program": grammar.show_program(
                            e.program,
                            name_classes=[
                                LAPSGrammar.HUMAN_READABLE,
                                LAPSGrammar.NUMERIC_FUNCTION_NAMES,
                            ],
                            debug=True,
                        ),
                        "language": task_language,
                    }
                ]
                if len(program_examples) == n_program_examples:
                    return program_examples

                    # Go to next task
                    break

        return program_examples

    def _parse_completion(
        self,
        completion,
        experiment_state,
        verbose,
        function_name_classes: list = [LAPSGrammar.DEFAULT_FUNCTION_NAMES],
    ):
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
                        "error": GPTLibraryLearner.ERROR_JSON,
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
                        "error": GPTLibraryLearner.ERROR_MISSING_FIELD,
                    }
                )
                continue

            if not data["function_name"]:
                parse_results.append(
                    {
                        "index": choice["index"],
                        "text": choice["text"],
                        "valid": False,
                        "error": GPTLibraryLearner.ERROR_NULL_NAME,
                    }
                )
                continue

            # add filter
            program_str_gpt = data["function_expression"]
            grammar = experiment_state.models[model_loaders.GRAMMAR]
            # CHECK 1: Does the program parse?
            try:
                # Write the program back into the DreamCoder form from whatever it was initially in.
                program_str = grammar.show_program(
                    program_str_gpt, input_name_class=function_name_classes
                )
                p = Program.parse(program_str)
            except Exception as e:
                if verbose:
                    print(f"Failed to parse ({type(e)}): {program_str_gpt}")
                parse_results.append(
                    {
                        "index": choice["index"],
                        "text": choice["text"],
                        "valid": False,
                        "error": GPTLibraryLearner.ERROR_PARSE,
                    }
                )
                continue
            # CHECK 2: Does the program typecheck?
            try:
                p.infer()
            except Exception:
                if verbose:
                    print(f"Type inference failure for: {str(p)}")
                parse_results.append(
                    {
                        "index": choice["index"],
                        "text": choice["text"],
                        "valid": False,
                        "error": GPTLibraryLearner.ERROR_INFER,
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
