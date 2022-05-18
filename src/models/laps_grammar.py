"""
laps_grammmar.py | Author : Catherine Wong

Utility wrapper function around the DreamCoder Grammar. Elevates common functions to be class functions in order to support calling with an ExperimentState.
"""
import json
import os
import subprocess
from typing import Counter

import numpy as np

import src.experiment_iterator as experiment_iterator
import src.models.model_loaders as model_loaders
import src.task_loaders as task_loaders
from dreamcoder.compression import induceGrammar
from dreamcoder.dreaming import backgroundHelmholtzEnumeration
from dreamcoder.enumeration import multicoreEnumeration
from dreamcoder.frontier import Frontier, FrontierEntry
from dreamcoder.grammar import Grammar
from dreamcoder.program import Program
from dreamcoder.type import Type
from dreamcoder.utilities import ParseFailure, get_root_dir


class LAPSGrammar(Grammar):
    """LAPSGrammar: utility model wrapper around DreamCoder Grammar to support model functions (sampling, program inference, compression)."""

    name = "laps_grammar"

    DEFAULT_MAXIMUM_FRONTIER = 5  # Maximum top-programs to keep in frontier
    DEFAULT_CPUS = 12  # Parallel CPUs
    DEFAULT_ENUMERATION_SOLVER = "ocaml"  # OCaml, PyPy, or Python enumeration
    DEFAULT_SAMPLER = "helmholtz"
    DEFAULT_BINARY_DIRECTORY = os.path.join(DEFAULT_ENUMERATION_SOLVER, "bin")
    DEFAULT_EVALUATION_TIMEOUT = 1  # Timeout for evaluating a program on a task
    DEFAULT_MAX_MEM_PER_ENUMERATION_THREAD = 1000000000  # Max memory usage per thread
    DEFAULT_MAX_SAMPLES = 5000

    # Compression hyperparameters
    DEFAULT_TOP_K = 2
    DEFAULT_PSEUDOCOUNTS = 30.0
    DEFAULT_ARITY = 3
    DEFAULT_AIC = 1.0
    DEFAULT_STRUCTURE_PENALTY = 1.5
    DEFAULT_COMPRESSOR_TYPE = "ocaml"
    DEFAULT_COMPRESSOR = "compression"
    DEFAULT_API_COMPRESSOR = "compression_rescoring_api"
    DEFAULT_MAX_COMPRESSION = 1000

    # API keys for the OCaml compressor API.
    API_FN, REQUIRED_ARGS, KWARGS = "api_fn", "required_args", "kwargs"
    GRAMMAR, FRONTIERS, COMPRESSION_SCORES = (
        "grammar",
        "frontiers",
        "compression_scores",
    )
    TEST_API_FN = "test_send_receive_response"  # Test ping function to make sure the binary is avaiable.

    # Standard hyperparameters for the compressor.
    MAX_CANDIDATES_PER_COMPRESSION_STEP = "max_candidates_per_compression_step"
    MAX_CANDIDATES_TO_RETAIN_FOR_REWRITING = (
        "max_grammar_candidates_to_retain_for_rewriting"
    )
    MAX_COMPRESSION_STEPS = "max_compression_steps"
    ARITY = "arity"
    PSEUDOCOUNTS = "pseudocounts"
    AIC = "aic"
    CPUS = "cpus"
    STRUCTURE_PENALTY = "structure_penalty"
    TOP_K = "top_k"  # Compress with respect to the top K frontiers.

    DEFAULT_FUNCTION_NAMES = "default" # DC names. Uses inlined naming (#(lambda ...)) for inventions.
    DEFAULT_NO_INLINE_FUNCTION_NAMES = "default_no_inline" # DC names, but does not include inlined naming for inventions (inventions do not have a default name).
    NUMERIC_FUNCTION_NAMES = "numeric"
    EXCLUDE_NAME_INITIALIZATION = [DEFAULT_FUNCTION_NAMES, NUMERIC_FUNCTION_NAMES]
    # Other common naming schemes.
    HUMAN_READABLE = "human_readable"
    DEFAULT_LAMBDA = "lambda"

    NUMERIC_FUNCTION_NAMES_PREFIX = "fn_"
    ALT_NAME = "_alt_"

    def __init__(
        self,
        logVariable,
        productions,
        continuationType=None,
        initialize_parameters_from_grammar=None,
    ):
        self.function_prefix = ""  # String prefix for all functions in the grammar
        super(LAPSGrammar, self).__init__(logVariable, productions, continuationType)
        if initialize_parameters_from_grammar:
            self.function_prefix = initialize_parameters_from_grammar.function_prefix
        # Initialize other metadata about the productions.
        self.all_function_names_counts = Counter()
        self.all_function_names_to_productions = dict()
        self.function_names = self._init_function_names(
            initialize_parameters_from_grammar
        )
    
    def _add_base_primitive(self, base_primitive, use_default_as_human_readable=False):
        numeric_idx = len(self.function_names)
        self.function_names[str(base_primitive)] = {
            LAPSGrammar.DEFAULT_FUNCTION_NAMES: str(base_primitive),
            LAPSGrammar.DEFAULT_NO_INLINE_FUNCTION_NAMES : str(base_primitive),
            LAPSGrammar.NUMERIC_FUNCTION_NAMES: LAPSGrammar.NUMERIC_FUNCTION_NAMES_PREFIX
            + str(numeric_idx),
        }
        if use_default_as_human_readable:
            self.function_names[str(base_primitive)][LAPSGrammar.HUMAN_READABLE] = str(
                base_primitive
            )

        self.all_function_names_counts[str(base_primitive)] += 1
        self.all_function_names_to_productions[str(base_primitive)] = str(
            base_primitive
        )

    def _init_function_names(self, initialize_from_grammar=None):
        """
        Creates a {production_key : {name_class : name}} dictionary containing alternate names for productions in the grammar.
        """
        # Sort the function names such that the inventions are always at the end.
        inventions = sorted(
            [p for p in self.primitives if p.isInvented], key=lambda p: str(p)
        )
        base_dsl = sorted(
            [p for p in self.primitives if not p.isInvented], key=lambda p: str(p)
        )

        function_names = {
            str(p): {
                LAPSGrammar.DEFAULT_FUNCTION_NAMES: str(p),
                LAPSGrammar.NUMERIC_FUNCTION_NAMES: LAPSGrammar.NUMERIC_FUNCTION_NAMES_PREFIX
                + str(idx),
            }
            for idx, p in enumerate(base_dsl + inventions)
        }
        for p in base_dsl:
            # Only base inventions have non-inlined names.
            function_names[str(p)][LAPSGrammar.DEFAULT_NO_INLINE_FUNCTION_NAMES] = str(p)

            # Set any alternate names that exist.
            function_names[str(p)][LAPSGrammar.HUMAN_READABLE] = p.alternate_names[-1]

        # Retain any new names from the previous grammar.
        if initialize_from_grammar is not None:
            for p in initialize_from_grammar.function_names:
                for name_class in initialize_from_grammar.function_names[p]:
                    if name_class not in self.EXCLUDE_NAME_INITIALIZATION:
                        if p not in function_names:
                            function_names[p] = dict()

                        function_names[p][
                            name_class
                        ] = initialize_from_grammar.function_names[p][name_class]

        # Initialize counter to avoid function name duplication
        for p in function_names:
            for name_class in function_names[p]:
                function_name = function_names[p][name_class]
                if not (
                    self.all_function_names_counts[function_name] == 0
                    or self.all_function_names_to_productions[function_name] == p
                ):
                    assert False
                base_name = self._get_base_name(function_name)
                self.all_function_names_counts[function_name] += 1

                # Count the existing base names.
                if function_name != base_name:
                    self.all_function_names_counts[base_name] += 1
                self.all_function_names_to_productions[function_name] = p
        return function_names

    def get_name(self, production_key, name_classes):
        name_classes += [LAPSGrammar.DEFAULT_FUNCTION_NAMES]
        for n in name_classes:
            if n in self.function_names[production_key]:
                return self.function_names[production_key][n]
        assert False

    def has_alternate_name(self, production_key, name_class):
        """
        :ret: bool - whether the production has been assigned a function name for the class different from the original name.
        """
        production_key = str(production_key)
        return name_class in self.function_names[production_key]

    def _get_base_name(self, name):
        base_name = name.split(self.ALT_NAME)[0]
        return base_name

    def set_function_name(self, production_key, name_class, name):
        """
        production_key: which production to provide a name to.
        name_class: what class of names this is (eg. default, stitch_default, codex.)
        name: what alternate name to assign
        """
        base_name = self._get_base_name(name)

        if production_key not in self.function_names:
            raise Exception(f"Error: {production_key} not in grammar.")

        # Disallow duplicate names
        if (
            self.all_function_names_counts[base_name] > 0
            and not self.all_function_names_to_productions.get(base_name, "")
            == production_key
        ):
            name = f"{base_name}_alt_{self.all_function_names_counts[base_name]}"

        self.function_names[production_key][name_class] = name
        self.all_function_names_counts[name] += 1
        if name != base_name:
            self.all_function_names_counts[base_name] += 1
        self.all_function_names_to_productions[name] = production_key
        return name

    def show_program(
        self,
        program,
        name_classes=[DEFAULT_FUNCTION_NAMES],
        lam=DEFAULT_LAMBDA,
        input_name_class=[DEFAULT_FUNCTION_NAMES],
        input_lam=DEFAULT_LAMBDA,
        debug=False,
    ):
        if input_name_class == name_classes:
            unchanged = str(program)

            return unchanged

        if type(program) == str:
            program = Program.parse(program, allow_unknown_primitives=True)
        return self.show_program_from_tree(program, name_classes, lam, debug)

    def show_program_from_tree(
        self, program, name_classes, lam, debug=False,
    ):
        # Show a program, walking the tree and printing out alternate names as we go.
        class NameVisitor(object):
            def __init__(self, function_names, name_classes, lam, grammar):
                self.grammar = grammar
                self.name_classes = name_classes + [LAPSGrammar.DEFAULT_FUNCTION_NAMES]

                self.function_names = function_names
                self.lam = lam

            def invented(self, e, isFunction):
                original = "#" + str(e.body)
                for n in self.name_classes:
                    if n in self.function_names[original]:
                        return self.function_names[original][n]

            def primitive(self, e, isFunction):
                # First, find out what primitive we're talking about here.
                if not e.name in self.grammar.all_function_names_to_productions:
                    # Allow floats.
                    try:
                        float_primitive = float(e.name)  # If we can cast to float.
                        self.grammar._add_base_primitive(
                            e, use_default_as_human_readable=True
                        )
                        self.function_names = self.grammar.function_names
                    except:
                        raise ParseFailure((str(e), e))

                original_name = self.grammar.all_function_names_to_productions[e.name]
                for n in self.name_classes:
                    if n in self.function_names[original_name]:
                        return self.function_names[original_name][n]
                return e.name

            def index(self, e, isFunction):
                return e.show(isFunction)

            def application(self, e, isFunction):
                if isFunction:
                    return "%s %s" % (e.f.visit(self, True), e.x.visit(self, False))
                else:
                    return "(%s %s)" % (e.f.visit(self, True), e.x.visit(self, False),)

            def abstraction(self, e, isFunction):
                return "(%s %s)" % (self.lam, e.body.visit(self, False))

        return program.visit(
            NameVisitor(self.function_names, name_classes, lam, self), isFunction=False,
        )

    def infer_programs_for_tasks(
        self,
        experiment_state,
        task_split,
        task_batch_ids,
        enumeration_timeout,
        maximum_frontier=DEFAULT_MAXIMUM_FRONTIER,
        cpus=DEFAULT_CPUS,
        solver=DEFAULT_ENUMERATION_SOLVER,
        evaluation_timeout=DEFAULT_EVALUATION_TIMEOUT,
        max_mem_per_enumeration_thread=DEFAULT_MAX_MEM_PER_ENUMERATION_THREAD,
        solver_directory=DEFAULT_BINARY_DIRECTORY,
    ):
        """
        Infers programs for tasks via top-down enumerative search from the grammar.
        Updates Frontiers in experiment_state with discovered programs.

        Wrapper function around multicoreEnumeration from dreamcoder.enumeration.
        """
        tasks_to_attempt = experiment_state.get_tasks_for_ids(
            task_split=task_split, task_ids=task_batch_ids, include_samples=False
        )
        new_frontiers, _ = multicoreEnumeration(
            g=self,
            tasks=tasks_to_attempt,
            maximumFrontier=maximum_frontier,
            enumerationTimeout=enumeration_timeout,
            CPUs=cpus,
            solver=solver,
            evaluationTimeout=evaluation_timeout,
            max_mem_per_enumeration_thread=max_mem_per_enumeration_thread,
            solver_directory=solver_directory,
        )

        experiment_state.update_frontiers(
            new_frontiers=new_frontiers,
            maximum_frontier=maximum_frontier,
            task_split=task_split,
            is_sample=False,
        )

    def generative_sample_frontiers_for_tasks(
        self,
        experiment_state,
        task_split=None,
        task_batch_ids=None,
        enumeration_timeout=None,
        evaluation_timeout=DEFAULT_EVALUATION_TIMEOUT,
        max_samples=DEFAULT_MAX_SAMPLES,
        sampler=DEFAULT_SAMPLER,
        sampler_directory=DEFAULT_BINARY_DIRECTORY,
    ):
        """Samples frontiers via enumeration from the grammar.
        Samples according to task types in experiment_state.
        Wrapper around dreamcoder.dreaming.backgroundHelmholtzEnumeration.

        Updates experiment_state.sample_frontiers and experiment_state.sample_tasks to include samples."""

        # Samples typed programs bsaed on the distribution of training tasks,
        tasks_to_attempt = experiment_state.get_tasks_for_ids(
            task_split=task_loaders.TRAIN,
            task_ids=experiment_iterator.ExperimentState.ALL,
            include_samples=False,
        )

        # Serializer function for sending examples to Ocaml solver
        task_serializer_fn = tasks_to_attempt[0].ocaml_serializer

        sampled_frontiers = backgroundHelmholtzEnumeration(
            tasks_to_attempt,
            self,
            timeout=enumeration_timeout,
            evaluationTimeout=evaluation_timeout,
            special=experiment_state.metadata[
                experiment_iterator.OCAML_SPECIAL_HANDLER
            ],  # Domain-specific flag for handling enumeration
            executable=os.path.join(sampler_directory, sampler),
            serialize_special=task_serializer_fn,
            maximum_size=max_samples,
        )()

        for sampled_frontier in sampled_frontiers:
            experiment_state.sample_frontiers[sampled_frontier.task] = sampled_frontier
            experiment_state.sample_tasks[
                sampled_frontier.task.name
            ] = sampled_frontier.task

    def optimize_grammar_frontiers_for_frontiers(
        self,
        experiment_state,
        task_split,
        task_batch_ids,
        include_samples=False,
        top_k=DEFAULT_TOP_K,
        pseudocounts=DEFAULT_PSEUDOCOUNTS,
        arity=DEFAULT_ARITY,
        aic=DEFAULT_AIC,
        structure_penalty=DEFAULT_STRUCTURE_PENALTY,
        compressor_type=DEFAULT_COMPRESSOR_TYPE,
        compressor=DEFAULT_COMPRESSOR,
        compressor_directory=DEFAULT_BINARY_DIRECTORY,
        cpus=DEFAULT_CPUS,
        max_compression=DEFAULT_MAX_COMPRESSION,
    ):
        """Compresses grammar with respect to frontiers in task_batch_ids (or ALL).
        Updates the experiment_state.models[GRAMMAR] and experiment_state frontiers rewritten with respect to the grammar."""
        frontiers_to_optimize = experiment_state.get_frontiers_for_ids(
            task_split=task_split,
            task_ids=task_batch_ids,
            include_samples=include_samples,
        )

        optimized_grammar, optimized_frontiers = induceGrammar(
            self,
            frontiers_to_optimize,
            topK=top_k,
            pseudoCounts=pseudocounts,
            a=arity,
            aic=aic,
            structurePenalty=structure_penalty,
            topk_use_only_likelihood=False,
            backend=compressor_type,
            CPUs=cpus,
            iteration=experiment_state.curr_iteration,
            language_alignments=None,
            executable=os.path.join(compressor_directory, compressor),
            lc_score=0.0,
            max_compression=max_compression,
        )
        # Cast back to LAPSGrammar
        optimized_grammar = LAPSGrammar.fromGrammar(optimized_grammar)

        # Add samples.
        experiment_state.models[model_loaders.GRAMMAR] = optimized_grammar
        for optimized_frontier in optimized_frontiers:
            experiment_state.task_frontiers[task_split][
                optimized_frontier.task
            ] = optimized_frontier

    ## Utility functions for compressing over subsets of the programs and the grammar. This uses a different OCaml binary than the original compressor; it interfaces with compression_rescoring_api.
    def _send_receive_compressor_api_call(
        self,
        api_fn,
        grammar=None,
        frontiers={},
        kwargs={},
        compressor_type=DEFAULT_COMPRESSOR_TYPE,
        compressor=DEFAULT_API_COMPRESSOR,
        compressor_directory=DEFAULT_BINARY_DIRECTORY,
    ):
        """Caller for invoking an API function implemented in the the OCaml compression_rescoring_api binary. Function names and signatures are defined in compression_rescoring_api.ml.
        Expects:
            api_fn: string name of API function to call.
            grammar: Grammar object. If None, uses self/
            frontiers: {split : [array of Frontiers objects]}
            kwargs: additional pre-serialized objects dictionary.

        Returns: deserialized JSON response containing:
        {
            api_fn: API function that was called.
            required_args:
                {
                    grammar: [array of response Grammar objects.]
                    frontiers: [array of {
                        split: [array of response Frontier objects.]
                    }]
                }
            kwargs: any other response arguments.
        }
        """
        if compressor_type != self.DEFAULT_COMPRESSOR_TYPE:
            # Legacy Dreamcoder library supports Python enumeration
            raise NotImplementedError
        # Construct the JSON message.

        grammar = grammar if grammar is not None else self
        non_empty_frontiers = {
            split: [f for f in frontiers[split] if not f.empty] for split in frontiers
        }
        json_serialized_binary_message = {
            self.API_FN: str(api_fn),  # String name of the API function.
            self.REQUIRED_ARGS: {
                self.GRAMMAR: grammar.json(),
                self.FRONTIERS: {
                    split: [f.json() for f in non_empty_frontiers[split]]
                    for split in non_empty_frontiers
                },
            },
            self.KWARGS: kwargs,
        }
        json_serialized_binary_message = json.dumps(json_serialized_binary_message)

        ocaml_binary_path = os.path.join(
            get_root_dir(), compressor_directory, compressor
        )
        try:
            process = subprocess.Popen(
                ocaml_binary_path, stdin=subprocess.PIPE, stdout=subprocess.PIPE
            )
            json_response, json_error = process.communicate(
                bytes(json_serialized_binary_message, encoding="utf-8")
            )
            json_response = json.loads(json_response.decode("utf-8"))
        except Exception as e:
            print(f"Error in _send_receive_compressor_api_call: {e}")

        json_deserialized_response = json_response
        json_deserialized_response[self.REQUIRED_ARGS][self.GRAMMAR] = [
            self._deserialize_json_grammar(serialized_grammar)
            for serialized_grammar in json_response[self.REQUIRED_ARGS][self.GRAMMAR]
        ]

        new_grammars = json_deserialized_response[self.REQUIRED_ARGS][self.GRAMMAR]
        # Deserialize the frontiers with respect to their corresponding grammar.
        json_deserialized_response[self.REQUIRED_ARGS][self.FRONTIERS] = [
            self._deserialize_json_frontiers(
                new_grammars[grammar_idx],
                non_empty_frontiers=non_empty_frontiers,
                json_frontiers=json_frontiers,
            )
            for grammar_idx, json_frontiers in enumerate(
                json_deserialized_response[self.REQUIRED_ARGS][self.FRONTIERS]
            )
        ]
        json_serialized_binary_message = json.loads(json_serialized_binary_message)
        return json_deserialized_response, json_error, json_serialized_binary_message

    def _deserialize_json_grammar(self, json_grammar):
        return LAPSGrammar(
            json_grammar["logVariable"],
            [
                (l, p.infer(), p)
                for production in json_grammar["productions"]
                for l in [production["logProbability"]]
                for p in [Program.parse(production["expression"])]
            ],
            continuationType=self.continuationType,
        )

    def _deserialize_json_frontiers(self, grammar, non_empty_frontiers, json_frontiers):
        """Deserializes frontiers with respect to a grammar to evaluate likelihoods.
        non_empty_frontiers : non-empty frontiers, ordered as they were sent to the compressor {split : array of original Frontiers objects.}
        json_frontiers: {split: array of serialized json_frontier objects sent from the OCaml binary.} should be ordered exactly as non_empty_frontiers

        :ret: {split: array of Frontiers with entries scored with respect to the grammar.}
        """
        # Wrap this in a try catch in the case of an error
        def maybe_entry(program, json_entry, grammar, request):
            try:
                return FrontierEntry(
                    program,
                    logLikelihood=json_entry["logLikelihood"],
                    logPrior=grammar.logLikelihood(request, program),
                )
            except Exception as e:
                print(f"Error adding frontier entry: {str(program)}")
                print(e)
                return None

        deserialized_frontiers = {}
        for split in non_empty_frontiers:
            deserialized_frontiers[split] = []
            for original_frontier, serialized_frontier in zip(
                non_empty_frontiers[split], json_frontiers[split]
            ):

                entries = [
                    maybe_entry(p, e, grammar, original_frontier.task.request)
                    for e in serialized_frontier["programs"]
                    for p in [Program.parse(e["program"])]
                ]
                entries = [e for e in entries if e is not None]
                deserialized_frontiers[split].append(
                    Frontier(entries, task=original_frontier.task)
                )
        return deserialized_frontiers

    def _get_dummy_grammar_scores_frontiers(self, frontiers):
        grammar_frontier_score_candidates = [
            {
                self.GRAMMAR: self,
                self.FRONTIERS: frontiers,
                self.COMPRESSION_SCORES: 0.0,
            }
        ]
        return grammar_frontier_score_candidates

    def _get_compressed_grammar_candidates_and_rewritten_frontiers(
        self,
        experiment_state,
        task_splits,
        task_ids_in_splits,
        max_candidates_per_compression_step,  # How many to consider.
        max_grammar_candidates_to_retain_for_rewriting,  # How many to actually rewrite.
        top_k=DEFAULT_TOP_K,
        pseudocounts=DEFAULT_PSEUDOCOUNTS,
        arity=DEFAULT_ARITY,
        aic=DEFAULT_AIC,
        structure_penalty=DEFAULT_STRUCTURE_PENALTY,
        compressor_type=DEFAULT_COMPRESSOR_TYPE,
        compressor=DEFAULT_API_COMPRESSOR,
        compressor_directory=DEFAULT_BINARY_DIRECTORY,
        cpus=DEFAULT_CPUS,
        debug_get_dummy=False,
    ):
        """
        Corresponding OCaml API Function:
        get_compressed_grammar_candidates_and_rewritten_frontiers
        Call the compressor to return a set of grammar candidates augmneted with library functions {g_c_0, g_c_1...} and frontiers rewritten under each candidate.

        Evaluates max_candidates_per_compression_step under the compression score, and currently returns the max_candidates_per_compression_step under the compression score.

        Always assumes it should only optimize with respect to the TRAIN frontiers, but rewrites train/test frontiers under the compressed grammar.

        :ret:
            Array of {
                "grammar": Grammar candidate.
                "frontiers": {split:[rewritten frontiers]}.
                "compression_score": scalar score of grammar wrt. frontiers.
            }
        """
        api_fn = "get_compressed_grammar_candidates_and_rewritten_frontiers"
        frontiers = experiment_state.get_frontiers_for_ids_in_splits(
            task_splits=task_splits,
            task_ids_in_splits=task_ids_in_splits,
            include_samples=False,
        )

        # Debug mock-method for avoiding an expensive compression step.
        if debug_get_dummy:
            return self._get_dummy_grammar_scores_frontiers(frontiers)

        # Build the standard KWARGS.
        kwargs = {
            self.MAX_CANDIDATES_PER_COMPRESSION_STEP: max_candidates_per_compression_step,
            self.MAX_CANDIDATES_TO_RETAIN_FOR_REWRITING: max_grammar_candidates_to_retain_for_rewriting,
            self.ARITY: arity,
            self.PSEUDOCOUNTS: pseudocounts,
            self.AIC: aic,
            self.CPUS: cpus,
            self.STRUCTURE_PENALTY: structure_penalty,
            self.TOP_K: top_k,
        }

        json_deserialized_response, _, _ = self._send_receive_compressor_api_call(
            api_fn=api_fn,
            grammar=self,
            frontiers=frontiers,
            kwargs=kwargs,
            compressor_type=compressor_type,
            compressor=compressor,
            compressor_directory=compressor_directory,
        )

        grammar_candidates = json_deserialized_response[self.REQUIRED_ARGS][
            self.GRAMMAR
        ]

        rewritten_frontier_candidates = json_deserialized_response[self.REQUIRED_ARGS][
            self.FRONTIERS
        ]
        grammar_frontier_score_candidates = [
            {
                self.GRAMMAR: grammar_candidates[idx],
                self.FRONTIERS: rewritten_frontier_candidates[idx],
                self.COMPRESSION_SCORES: json_deserialized_response[self.REQUIRED_ARGS][
                    self.COMPRESSION_SCORES
                ][idx],
            }
            for idx in range(len(grammar_candidates))
        ]

        return grammar_frontier_score_candidates

    def _get_compressed_grammar_candidates_and_rewritten_frontiers_parallel(
        self,
        experiment_state,
        task_splits,
        task_ids_in_splits,
        max_candidates_per_compression_step,
        max_compression_steps,
        top_k=DEFAULT_TOP_K,
        pseudocounts=DEFAULT_PSEUDOCOUNTS,
        arity=DEFAULT_ARITY,
        aic=DEFAULT_AIC,
        structure_penalty=DEFAULT_STRUCTURE_PENALTY,
        compressor_type=DEFAULT_COMPRESSOR_TYPE,
        compressor=DEFAULT_API_COMPRESSOR,
        compressor_directory=DEFAULT_BINARY_DIRECTORY,
        cpus=DEFAULT_CPUS,
    ):
        """
        Parallel implementation for getting candidate compressed grammars and rewritten frontiers.
        Corresponding OCaml API Function:
        get_compressed_grammar_candidates_and_rewritten_frontiers
        Call the compressor to return a set of grammar candidates augmneted with library functions {g_c_0, g_c_1...} and frontiers rewritten under each candidate.

        Evaluates max_candidates_per_compression_step under the compression score, and currently returns the max_candidates_per_compression_step under the compression score.

        Always assumes it should only optimize with respect to the TRAIN frontiers, but rewrites train/test frontiers under the compressed grammar.

        :ret:
            Array of {
                "grammar": Grammar candidate.
                "frontiers": {split:[rewritten frontiers]}.
                "compression_score": scalar score of grammar wrt. frontiers.
            }
        """
        # NB  this could actually be done with separate threads.

    def _get_compressed_grammmar_and_rewritten_frontiers(
        self,
        experiment_state,
        task_splits,
        task_ids_in_splits,
        max_candidates_per_compression_step,
        max_compression_steps,
        top_k=DEFAULT_TOP_K,
        pseudocounts=DEFAULT_PSEUDOCOUNTS,
        arity=DEFAULT_ARITY,
        aic=DEFAULT_AIC,
        structure_penalty=DEFAULT_STRUCTURE_PENALTY,
        compressor_type=DEFAULT_COMPRESSOR_TYPE,
        compressor=DEFAULT_API_COMPRESSOR,
        compressor_directory=DEFAULT_BINARY_DIRECTORY,
        cpus=DEFAULT_CPUS,
    ):
        """
        Corresponding OCaml API Function:get_compressed_grammmar_and_rewritten_frontiers.
        Call the compressor to rewrite the grammar and the frontiers.

        Runs compression up to max_compression_steps, evaluating max_candidates_per_compression_step under the compression score, and greedily adding the top candidate each time.
        Always assumes it should only optimize with respect to the TRAIN frontiers, but rewrites train/test frontiers under the compressed grammar.

        Updates the experiment_state.models[GRAMMAR] and experiment_state frontiers rewritten with respect to the grammar.
        :ret:
            grammar: Grammar object containing the final DSL with up to max_compression_steps new library inventions.
            rewritten_train_test_frontiers: {split : [frontiers] with frontiers rewritten under the final grammar.} Note that this only returns non-empty frontiers for the tasks.

        """
        api_fn = "get_compressed_grammmar_and_rewritten_frontiers"
        frontiers = experiment_state.get_frontiers_for_ids_in_splits(
            task_splits=task_splits,
            task_ids_in_splits=task_ids_in_splits,
            include_samples=False,
        )

        # Build the standard KWARGS.
        kwargs = {
            self.MAX_CANDIDATES_PER_COMPRESSION_STEP: max_candidates_per_compression_step,
            self.MAX_COMPRESSION_STEPS: max_compression_steps,
            self.ARITY: arity,
            self.PSEUDOCOUNTS: pseudocounts,
            self.AIC: aic,
            self.CPUS: cpus,
            self.STRUCTURE_PENALTY: structure_penalty,
            self.TOP_K: top_k,
        }

        json_deserialized_response, _, _ = self._send_receive_compressor_api_call(
            api_fn=api_fn,
            grammar=self,
            frontiers=frontiers,
            kwargs=kwargs,
            compressor_type=compressor_type,
            compressor=compressor,
            compressor_directory=compressor_directory,
        )

        optimized_grammar = json_deserialized_response[self.REQUIRED_ARGS][
            self.GRAMMAR
        ][0]

        rewritten_train_test_frontiers = json_deserialized_response[self.REQUIRED_ARGS][
            self.FRONTIERS
        ][0]

        experiment_state.models[model_loaders.GRAMMAR] = optimized_grammar
        for task_split in rewritten_train_test_frontiers:
            for rewritten_frontier in rewritten_train_test_frontiers[task_split]:
                experiment_state.task_frontiers[task_split][
                    rewritten_frontier.task
                ] = rewritten_frontier

        return optimized_grammar, rewritten_train_test_frontiers

    def evaluate_frontiers(
        self,
        experiment_state,
        task_splits,
        task_ids_in_splits,
        compute_likelihoods: bool = True,
        compute_description_lengths: bool = True,
        include_samples: bool = False,
        save_filename: str = None,
    ):
        """
        Evaluates frontier likelihoods and/or description lengths with respect to the current grammar.
        """
        assert compute_likelihoods or compute_description_lengths

        frontiers = experiment_state.get_frontiers_for_ids_in_splits(
            task_splits=task_splits,
            task_ids_in_splits=task_ids_in_splits,
            include_samples=include_samples,
        )

        if compute_likelihoods:
            log_likelihoods_by_task = {}
            log_likelihoods = []
        if compute_description_lengths:
            description_lengths_by_task = {}
            description_lengths = []

        # Rescore all frontiers under current grammar.
        for task_split in task_splits:

            if compute_likelihoods:
                log_likelihoods_by_task[task_split] = {}
            if compute_description_lengths:
                description_lengths_by_task[task_split] = {}

            for f in frontiers[task_split]:
                # Compute log likelihood of each program
                if compute_likelihoods:
                    lls = [self.logLikelihood(f.task.request, e.program) for e in f]
                    log_likelihoods += lls
                    log_likelihoods_by_task[task_split][f.task.name] = lls

                # Additionally, compute description length of each program
                if compute_description_lengths:
                    dls = [
                        len(Program.left_order_tokens(e.program, show_vars=True))
                        for e in f
                    ]
                    description_lengths += dls
                    description_lengths_by_task[task_split][f.task.name] = dls

        if compute_likelihoods:
            print(
                f"EVALUATION: evaluate_frontiers : mean log likelihood of {len(log_likelihoods)} programs in splits: {task_splits} is: {np.mean(log_likelihoods)}"
            )
        if compute_description_lengths:
            print(
                f"EVALUATION: evaluate_frontiers : mean description length of {len(description_lengths)} programs in splits: {task_splits} is: {np.mean(description_lengths)}"
            )

        if save_filename is not None:
            save_filepath = os.path.join(
                os.getcwd(), experiment_state.get_checkpoint_directory(), save_filename,
            )
            os.makedirs(os.path.dirname(save_filepath), exist_ok=True)
            with open(save_filepath, "w") as f:
                data = {}
                if compute_likelihoods:
                    data.update(
                        {
                            "mean_log_likelihood": np.mean(log_likelihoods),
                            "log_likelihoods_by_task": log_likelihoods_by_task,
                        }
                    )
                if compute_description_lengths:
                    data.update(
                        {
                            "mean_description_length": np.mean(description_lengths),
                            "description_lengths_by_task": description_lengths_by_task,
                        }
                    )
                json.dump(data, f)

    def evaluate_frontier_likelihoods(
        self,
        experiment_state,
        task_splits,
        task_ids_in_splits,
        include_samples=False,
        save_filename: str = None,
    ):
        """
        Evaluates and reports the frontier likelihoods with respect to the current grammar.
        """
        return self.evaluate_frontiers(
            experiment_state,
            task_splits,
            task_ids_in_splits,
            compute_likelihoods=True,
            compute_description_lengths=False,
            include_samples=include_samples,
            save_filename=save_filename,
        )

    ## Elevate static methods to create correct class.
    @staticmethod
    def fromGrammar(grammar):
        return LAPSGrammar(
            grammar.logVariable,
            grammar.productions,
            continuationType=grammar.continuationType,
        )

    @staticmethod
    def fromProductions(productions, logVariable=0.0, continuationType=None):
        """Make a grammar from primitives and their relative logpriors."""
        return LAPSGrammar(
            logVariable,
            [(l, p.infer(), p) for l, p in productions],
            continuationType=continuationType,
        )

    @staticmethod
    def uniform(primitives, continuationType=None):
        return LAPSGrammar(
            0.0,
            [(0.0, p.infer(), p) for p in primitives],
            continuationType=continuationType,
        )

    def get_checkpoint_filepath(self, checkpoint_directory):
        return os.path.join(checkpoint_directory, f"{self.name}.json")

    def checkpoint(self, experiment_state, checkpoint_directory):
        f"=> Checkpointed grammar to: {self.get_checkpoint_filepath(checkpoint_directory)}==========="
        with open(self.get_checkpoint_filepath(checkpoint_directory), "w") as f:
            json.dump(self.json(), f)

    def load_model_from_checkpoint(self, experiment_state, checkpoint_directory):
        with open(self.get_checkpoint_filepath(checkpoint_directory)) as f:
            json_grammar = json.load(f)

        reloaded_primitives = [
            Program.parse(production["expression"])
            for production in json_grammar["productions"]
        ]
        log_probabilities = [
            production["logProbability"] for production in json_grammar["productions"]
        ]
        productions = [
            (probability, p.infer(), p)
            for (probability, p) in zip(log_probabilities, reloaded_primitives)
        ]
        grammar = LAPSGrammar(
            logVariable=json_grammar["logVariable"],
            productions=productions,
            continuationType=Type.fromjson(json_grammar["continuationType"]),
        )
        experiment_state.models[model_loaders.GRAMMAR] = grammar
        return grammar
