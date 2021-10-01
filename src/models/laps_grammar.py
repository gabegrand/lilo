"""
laps_grammmar.py | Author : Catherine Wong

Utility wrapper function around the DreamCoder Grammar. Elevates common functions to be class functions in order to support calling with an ExperimentState.
"""
import subprocess
import os, json
from dreamcoder.grammar import Grammar
from dreamcoder.frontier import Frontier, FrontierEntry
from dreamcoder.program import Program
from dreamcoder.enumeration import multicoreEnumeration
from dreamcoder.dreaming import backgroundHelmholtzEnumeration
from dreamcoder.compression import induceGrammar
from dreamcoder.utilities import get_root_dir

import src.task_loaders as task_loaders
import src.models.model_loaders as model_loaders
import src.experiment_iterator as experiment_iterator


class LAPSGrammar(Grammar):
    """LAPSGrammar: utility model wrapper around DreamCoder Grammar to support model functions (sampling, program inference, compression)."""

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
            task_split=task_split, task_ids=task_batch_ids, include_samples=False
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

        ## TODO: remove this -- test on train and test
        frontiers["test"] = frontiers["train"]

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
