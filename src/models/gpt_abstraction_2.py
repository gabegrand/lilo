"""
gpt_abstraction.py | Author : Maxine Liu.

Queries GPT to generate new abstraction.
"""
import json
import os
from typing import Dict, List

import stitch_core as stitch

import src.models.model_loaders as model_loaders
from dreamcoder.frontier import Frontier, FrontierEntry
from dreamcoder.program import Invented, Program
from src.experiment_iterator import RANDOM_GENERATOR
from src.models.gpt_base import DEFAULT_LINE_SEPARATOR
from src.models.laps_grammar import LAPSGrammar
from src.models.library_namer import GPTLibraryNamer, LibraryNamerPrompt
from src.models.stitch_base import StitchBase
from src.task_loaders import ALL, TRAIN

ModelRegistry = model_loaders.ModelLoaderRegistries[model_loaders.LIBRARY_LEARNER]


class LibraryAbstractionPrompt2(LibraryNamerPrompt):

    TEXT_ABSTRACTION_HEADER = "You are about to undertake a task focused on abstraction learning. The objective is to develop reusable, compressed functions derived from existing programs. "
    TEXT_PROGRAM_HEADER = "Consider some programs of solved tasks."
    TEXT_FUNCTION_HEADER = (
        "To get started, consider the functions provided in the library:"
    )

    def __init__(
        self,
        abstraction_definitions: Dict,
        task_examples: List[Dict],
        line_separator: str = DEFAULT_LINE_SEPARATOR,
        abstraction_target: Dict = None,
    ):
        super().__init__(
            abstraction_definitions=abstraction_definitions,
            abstraction_target=abstraction_target,
            usage_examples=task_examples,
            line_separator=line_separator,
        )
        self.task_examples = task_examples
        self.message_list = []

    def _build_instruction(self):
        text = (
            "Here is an example for coming up with a reusable function for a simple math domain.\n\n"
            "Input:\n"
            "(lambda (+ 3 (* (+ 2 4) 2)))\n"
            "(lambda (map (lambda (+ 3 (* 4 (+ 3 $0)))) $1))\n"
            "(lambda (* 2 (+ 3 (* $0 (+ 2 1)))))\n\n"
            "Observation:\n"
            "All of the input programs add 3 to the result of a multiplication.\n\n"
            "Function expression:\n"
            "(lambda (lambda (+ 3 (* $0 $1))))\n\n"
            "Now, let's rewrite the input to use this function expression:\n"
            "(lambda (lambda (lambda (+ 3 (* $0 $1)))) (+ 2 4) 2)\n"
            "...\n\n"
            "---\n"
            "{\n"
            '    "readable_name": "multiply_and_add_three",\n'
            '    "function_expression": "(lambda (lambda (+ 3 (* $0 $1))))",\n'
            '    "description": "Multiplies the two arguments and adds three to the result.",\n'
            '    "explanation": "..."\n'
            "}"
        )

        return text

    # all the dsl primitives and program examples
    def _build_abstraction_header(self):
        text_list = [
            self.TEXT_ABSTRACTION_HEADER
            + self.line_separator
            + self.TEXT_FUNCTION_HEADER
            + self.line_separator
        ]
        for abstraction in self.abstraction_definitions.keys():
            text_list += [self._fn_docstring(abstraction) + self.line_separator]

        text_list += [self.line_separator]
        return self.line_separator.join(text_list)

    # task examples by cluster
    def _build_task_prompt(self):
        text_list = [self.TEXT_PROGRAM_HEADER]
        for task in self.task_examples:
            # text_list += [
            #     self.DEFAULT_PREFIX_LANGUAGE + str(task) + self.line_separator
            # ]
            text_list += [
                # self.DEFAULT_PREFIX_LANGUAGE
                # + "task description: "+task["language"]
                # + self.line_separator
                # + "program to solve task: "
                self.DEFAULT_PREFIX_PROGRAM
                + task["program"]
                + self.line_separator
            ]
        return self.line_separator.join(text_list)

    def _build_abstraction_footer(self):

        return (
            f"Your challenge is to write a compact, reusable function based on the programs. Find patterns in the programs that can be reused across multiple tasks."
            + "\n"
            f"- Only propose one function." + "\n"
            f"- Only use the functions available to you (they are in the library provided)."
            f"- Make sure the abstraction has balanced parentheses." + "\n"
            f"- Variables are named $0, $1, ..." + "\n"
            f"- Make sure to have the same number of (lambda ... ) expressions as the number of variables."
            + "\n"
            f"- The function should be encoded in the following JSON format. Description should be short (less than 15 words). Don't return anything other than this JSON."
            + "\n"
            "{" + "\n"
            f'    "readable_name": TODO,' + "\n"
            f'    "function_expression": TODO,' + "\n"
            f'    "description": TODO' + "\n" + "}"
        )

    def to_message_list(self):
        message_list = [self.chat_message(self._build_instruction())]
        message_list += [self.chat_message(self._build_abstraction_header())]
        message_list += [self.chat_message(self._build_task_prompt())]
        message_list += [self.chat_message(self._build_abstraction_footer())]
        self.message_list = message_list


@ModelRegistry.register
class GPTLibraryLearner2(GPTLibraryNamer, StitchBase):
    name = "gpt_library_learner_2"

    results_file = "gpt_library_abstraction_results.json"
    prompt_file = "gpt_library_learner_prompts.json"
    error_file = "gpt_library_learner_error.json"

    ERROR_PARSE = "parse"
    ERROR_INFER = "infer"
    ERROR_SOLVE = "solve"
    ERROR_FREE_VARIABLE = "free variable"
    error_dict = {"json": 0, "infer": 0, "parse": 0, "freevariable": 0, "rewrite": 0}

    @staticmethod
    def load_model(experiment_state, **kwargs):
        return GPTLibraryLearner2(experiment_state=experiment_state, **kwargs)

    def __init__(self, experiment_state=None, engine=None):
        super().__init__(experiment_state=experiment_state, engine=engine)

    def generate_abstraction(
        self,
        experiment_state,
        task_splits: list,
        task_ids_in_splits: dict,
        # Querying
        best_of: int = 1,
        n_samples_per_abstraction: int = 5,
        top_p: float = 0.1,
        max_tokens: int = 256,
        n_function_generated: int = 10,
        # Prompt construction
        n_program_examples: int = 20,
        # Resume from prior runs
        resume_strategy: str = None,
        # Utilities
        verbose: bool = True,
    ):
        task_split = task_splits[0]
        assert task_split == TRAIN
        grammar = experiment_state.models[model_loaders.GRAMMAR]

        # Optional load from results JSON
        # if self._maybe_load_from_checkpoint(
        #     experiment_state, task_split, resume_strategy, add_primitive=True
        # ):
        #     return {
        #         SKIPPED_MODEL_FN: True,
        #     }

        gpt_abstraction_library = {}
        grammar = LAPSGrammar(
            logVariable=grammar.logVariable,  # TODO: Renormalize logVariable
            productions=grammar.productions,
            continuationType=grammar.continuationType,
            initialize_parameters_from_grammar=grammar,
        )
        experiment_state.models[model_loaders.GRAMMAR] = grammar

        prompt_dict = {}

        results_filepath = os.path.join(
            os.getcwd(),
            experiment_state.get_checkpoint_directory(),
            task_split,
            self.results_file,
        )
        os.makedirs(os.path.dirname(results_filepath), exist_ok=True)

        for function_num in range(n_function_generated):
            abstraction_definitions = self._get_abstraction_definitions(
                experiment_state, abstractions_only=False
            )
            task_examples, task_examples_id = self._get_task_examples(
                experiment_state, n_program_examples, n_function_generated, function_num
            )

            prompt = LibraryAbstractionPrompt2(
                abstraction_definitions=abstraction_definitions,
                task_examples=task_examples,
            )

            if verbose:
                prompt_dict[f"prompt{function_num}"] = str(prompt)
                print(prompt)

            completion = self.query_completion(
                prompt,
                n_samples=n_samples_per_abstraction,
                top_p=top_p,
                max_tokens=max_tokens,
                stop=None,
            )

            frontiers_dict = self.write_frontiers_to_file(
                experiment_state, task_splits, task_ids_in_splits
            )

            stitch_kwargs = stitch.from_dreamcoder(frontiers_dict)
            name_mapping = stitch_kwargs["name_mapping"]
            # Parse response
            parse_results = self._parse_completion(
                completion, experiment_state, verbose, name_mapping
            )
            # selected_result = self._select_result(parse_results)
            selected_results = [result for result in parse_results if result["valid"]]

            if verbose:
                print(f"GPT ({self.ENGINE}) completion:")
                print(json.dumps(parse_results, indent=4))

            # add the function to the grammar
            # if selected_result is not None:
            for selected_result in selected_results:
                readable_name = selected_result["data"]["readable_name"]
                function_expression = selected_result["data"]["function_expression"]
                grammar = experiment_state.models[model_loaders.GRAMMAR]
                function_name_classes = [
                    LAPSGrammar.HUMAN_READABLE,
                    LAPSGrammar.NUMERIC_FUNCTION_NAMES,
                ]
                program_str = "#" + grammar.show_program(
                    function_expression,
                    input_name_class=function_name_classes,
                    name_classes=[LAPSGrammar.DEFAULT_FUNCTION_NAMES],
                )
                p = Invented.parse(program_str)

                new_productions = (0.0, p.infer(), p)
                new_grammar = LAPSGrammar(
                    logVariable=grammar.logVariable,  # TODO: Renormalize logVariable
                    productions=grammar.productions + [new_productions],
                    continuationType=grammar.continuationType,
                    initialize_parameters_from_grammar=grammar,
                )

                p = stitch.Abstraction.from_dreamcoder(
                    name=readable_name,
                    dreamcoder_abstraction=program_str,
                    name_mapping=name_mapping,
                )

                rewrite_exception_occurred = False
                try:
                    rewrite_result = stitch.rewrite(
                        programs=stitch_kwargs["programs"],
                        abstractions=[p],
                    )
                except stitch.StitchException as e:
                    print(f"❌ Failed to create a function")
                    error_filepath = os.path.join(
                        os.getcwd(),
                        experiment_state.get_checkpoint_directory(),
                        task_split,
                        self.error_file,
                    )
                    with open(error_filepath, "w") as f:
                        json.dump({str(p): str(e)}, f, indent=4)
                    self.error_dict["rewrite"] += 1
                    rewrite_exception_occurred = True

                if not rewrite_exception_occurred:
                    gpt_abstraction_library[str(function_expression)] = selected_result[
                        "data"
                    ]
                    gpt_abstraction_library[str(function_expression)][
                        "compression_ratio"
                    ] = rewrite_result.json["compression_ratio"]
                    gpt_abstraction_library[str(function_expression)][
                        "original_cost"
                    ] = rewrite_result.json["original_cost"]
                    gpt_abstraction_library[str(function_expression)][
                        "final_cost"
                    ] = rewrite_result.json["final_cost"]
                    gpt_abstraction_library[str(function_expression)]["delta_cost"] = (
                        rewrite_result.json["original_cost"]
                        - rewrite_result.json["final_cost"]
                    )

                    if rewrite_result.json["compression_ratio"] != 1:
                        gpt_abstraction_library[str(function_expression)][
                            "utility"
                        ] = rewrite_result.json["abstractions"][-1]["utility"]
                        gpt_abstraction_library[str(function_expression)][
                            "num_uses"
                        ] = rewrite_result.json["abstractions"][-1]["num_uses"]
                    # name_mapping double check
                    programs_rewritten = stitch.stitch_to_dreamcoder(
                        rewrite_result.rewritten,
                        stitch_kwargs["name_mapping"]
                        + stitch.name_mapping_stitch(rewrite_result.json),
                    )
                    frontiers_rewritten = []
                    tasks = stitch_kwargs["tasks"]
                    for task_id, program in zip(tasks, programs_rewritten):
                        matching_tasks = experiment_state.get_tasks_for_ids(
                            task_splits[0],
                            [task_id],
                            include_samples=False,
                            include_ground_truth_tasks=True,
                        )
                        if matching_tasks:
                            if len(matching_tasks) > 1:
                                logging.warning(
                                    f"Found multiple ({len(matching_tasks)}) tasks associated with task_id {task_id}"
                                )
                                rng = experiment_state.metadata[RANDOM_GENERATOR]
                                task = rng.choice(matching_tasks)
                            else:
                                task = matching_tasks[0]
                            frontier = Frontier(
                                frontier=[
                                    FrontierEntry(
                                        program=Program.parse(program),
                                        logPrior=0.0,
                                        logLikelihood=0.0,
                                    )
                                ],
                                task=task,
                            )
                            frontiers_rewritten.append(frontier)

                    # Rescore frontiers under grammar
                    grammar = experiment_state.models[model_loaders.GRAMMAR]

                    # update name and description
                    new_grammar.set_function_name(
                        program_str,
                        name_class=LAPSGrammar.HUMAN_READABLE,
                        name=readable_name,
                    )
                    new_grammar.set_function_description(
                        name=program_str,
                        description=selected_result["data"]["description"],
                    )

                    new_grammar = new_grammar.insideOutside(
                        frontiers_rewritten, pseudoCounts=30, iterations=1
                    )
                    new_grammar = LAPSGrammar.fromGrammar(new_grammar)

                    experiment_state.models[model_loaders.GRAMMAR] = new_grammar
                    grammar = experiment_state.models[model_loaders.GRAMMAR]

                    frontiers_rewritten = [
                        grammar.rescoreFrontier(f) for f in frontiers_rewritten
                    ]

                    # Clear old frontiers and replace with rewritten
                    experiment_state.reset_task_frontiers(
                        task_split=task_split, task_ids=ALL
                    )
                    assert all(
                        [
                            t.empty
                            for t in experiment_state.task_frontiers[
                                task_split
                            ].values()
                        ]
                    )

                    experiment_state.update_frontiers(
                        new_frontiers=frontiers_rewritten,
                        maximum_frontier=grammar.maximum_frontier,
                        task_split=task_split,
                        is_sample=False,
                    )

                    print(
                        f"✅ Successfully created {readable_name}:{function_expression}"
                    )
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
                "n_program_examples": n_program_examples,
                "top_p": top_p,
                "max_tokens": max_tokens,
                "n_function_generated": n_function_generated,
            },
            "summary": {
                "n_abstractions_generated": n_abstractions_generated,
                "n_abstractions_total": len(gpt_abstraction_library),
            },
            "abstractions": gpt_abstraction_library,
            "error": self.error_dict,
        }

        # Save results to file

        with open(results_filepath, "w") as f:
            json.dump(results, f, indent=4)
        if verbose:
            print(f"Wrote results: {results_filepath}")

        prompt_filepath = os.path.join(
            os.getcwd(),
            experiment_state.get_checkpoint_directory(),
            task_split,
            self.prompt_file,
        )
        with open(prompt_filepath, "w") as f:
            json.dump(prompt_dict, f, ensure_ascii=False, indent=4)

    def _get_task_examples(
        self, experiment_state, n_task_examples, n_function_generated, function_num
    ):

        grammar = experiment_state.models[model_loaders.GRAMMAR]
        solved_task = []
        tasks = list(experiment_state.task_frontiers[TRAIN].keys())
        # get all solved tasks
        for task in tasks:
            frontier = experiment_state.task_frontiers[TRAIN][task]
            rng = experiment_state.metadata[RANDOM_GENERATOR]
            task_language = rng.choice(
                experiment_state.get_language_for_ids(TRAIN, [task.name])[0]
            )
            for e in frontier:
                # program = grammar.show_program(
                #     e.program,
                #     name_classes=[
                #         LAPSGrammar.HUMAN_READABLE,
                #         LAPSGrammar.NUMERIC_FUNCTION_NAMES,
                #     ],
                #     debug=True,
                # )
                # solved_task[task.name] = program
                solved_task += [
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
                break

        return solved_task, None
        # # get a dict with all task_id:task_embedding
        # task_language_loader = experiment_state.config["metadata"][
        #     "task_language_loader"
        # ]
        # embedding_filepath = get_embedding_directory_for_domain(task_language_loader)
        # try:
        #     with open(embedding_filepath, "r") as f:
        #         embedding_dict = json.load(f)
        # except FileNotFoundError:
        #     raise FileNotFoundError(
        #         f"The file '{embedding_filepath}' could not be found."
        #     )

        # # get a dict with all solved task_id:task_embedding
        # embedding_dict = {task_id: embedding_dict[task_id] for task_id in solved_task}

        # # cluster tasks
        # embeddings = np.array(list(embedding_dict.values()))
        # task_ids = np.array(list(embedding_dict.keys()))
        # kmeans = KMeans(n_clusters=n_function_generated, random_state=0)
        # clusters = kmeans.fit_predict(embeddings)
        # dict_cluster = dict(zip(task_ids, clusters))

        # task_examples_id = [
        #     task_id
        #     for task_id, cluster in dict_cluster.items()
        #     if cluster == function_num
        # ][:n_task_examples]

        # task_languages = [
        #     task_example[0]
        #     for task_example in experiment_state.get_language_for_ids(
        #         TRAIN, task_examples_id
        #     )
        # ]
        # task_programs = [solved_task[task_id] for task_id in task_examples_id]
        # task_programs = [{k: v} for k, v in zip(task_languages, task_programs)]
        # return task_programs, task_examples_id

    def _get_program(self, experiment_state):
        grammar = experiment_state.models[model_loaders.GRAMMAR]
        tasks = list(experiment_state.task_frontiers[TRAIN].keys())

        programs = {}

        for task in tasks:
            frontier = experiment_state.task_frontiers[TRAIN][task]
            experiment_state.get_language_for_ids(TRAIN, [task.name])[0]

            for e in frontier.entries:
                programs[task.name] = grammar.show_program(
                    e.program,
                    name_classes=[
                        LAPSGrammar.HUMAN_READABLE,
                        LAPSGrammar.NUMERIC_FUNCTION_NAMES,
                    ],
                    debug=True,
                )
        return programs

    # parse a new grammar
    def _parse_completion(
        self,
        completion,
        experiment_state,
        verbose,
        name_mapping,
        function_name_classes: list = [
            LAPSGrammar.HUMAN_READABLE,
            LAPSGrammar.NUMERIC_FUNCTION_NAMES,
        ],
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
                        "error": GPTLibraryLearner2.ERROR_JSON,
                    }
                )
                self.error_dict["json"] += 1
                continue

            if not (
                ("readable_name" in data)
                and ("function_expression" in data)
                and ("description" in data)
            ):
                parse_results.append(
                    {
                        "index": choice["index"],
                        "text": choice["text"],
                        "valid": False,
                        "error": GPTLibraryLearner2.ERROR_MISSING_FIELD,
                    }
                )
                continue

            if not data["readable_name"]:
                parse_results.append(
                    {
                        "index": choice["index"],
                        "text": choice["text"],
                        "valid": False,
                        "error": GPTLibraryLearner2.ERROR_NULL_NAME,
                    }
                )
                continue

            # add filter
            program_str_gpt = data["function_expression"]
            grammar = experiment_state.models[model_loaders.GRAMMAR]
            # CHECK 1: Does the program parse?
            try:
                # Write the program back into the DreamCoder form from whatever it was initially in.
                program_str = "#" + grammar.show_program(
                    program_str_gpt,
                    input_name_class=function_name_classes,
                    name_classes=[LAPSGrammar.DEFAULT_FUNCTION_NAMES],
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
                        "error": GPTLibraryLearner2.ERROR_PARSE,
                    }
                )
                self.error_dict["parse"] += 1
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
                        "error": GPTLibraryLearner2.ERROR_INFER,
                    }
                )
                self.error_dict["infer"] += 1
                continue

            # CHECK 3: Does the program have free variable
            try:
                stitch.Abstraction.from_dreamcoder(
                    name="name",
                    dreamcoder_abstraction=program_str,
                    name_mapping=name_mapping,
                )
            except Exception:
                # add lambda to program_str
                import re

                numbers = re.findall(r"\$(\d+)", program_str_gpt)
                num_variable = max(map(int, numbers))
                num_lambda = program_str_gpt.count("lambda")
                lambda_need = num_variable - num_lambda
                for i in range(lambda_need):
                    program_str_gpt = f"(lambda{program_str_gpt})"
                    program_str = "#" + grammar.show_program(
                        program_str_gpt,
                        input_name_class=function_name_classes,
                        name_classes=[LAPSGrammar.DEFAULT_FUNCTION_NAMES],
                    )
                    try:
                        p.parse()
                        p.infer()
                        stitch.Abstraction.from_dreamcoder(
                            name="name",
                            dreamcoder_abstraction=program_str,
                            name_mapping=name_mapping,
                        )
                    except Exception:
                        if verbose:
                            print(f"Free variable in: {str(p)}")
                        parse_results.append(
                            {
                                "index": choice["index"],
                                "text": choice["text"],
                                "valid": False,
                                "error": GPTLibraryLearner2.ERROR_FREE_VARIABLE,
                            }
                        )
                        self.error_dict["freevariable"] += 1
                        continue
                    choice["text"]["function_expression"] = program_str_gpt

            parse_results.append(
                {
                    "index": choice["index"],
                    "text": choice["text"],
                    "valid": True,
                    "data": data,
                }
            )
        return parse_results
