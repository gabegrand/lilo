"""
laps_grammmar.py | Author : Catherine Wong

Utility wrapper function around the DreamCoder Grammar. Elevates common functions to be class functions in order to support calling with an ExperimentState.
"""
import subprocess
from dreamcoder.frontier import Frontier
import os, json
from dreamcoder.grammar import Grammar
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
    GRAMMAR, FRONTIERS = "grammar", "frontiers"
    TEST_API_FN = "test_send_receive_response"  # Test ping function to make sure the binary is avaiable.

    # Standard hyperparameters for the compressor.
    MAX_CANDIDATES_PER_COMPRESSION_STEP = "max_candidates_per_compression_step"
    MAX_COMPRESSION_STEPS = "max_compression_steps"
    ARITY = "arity"
    PSEUDOCOUNTS = "pseudocounts"
    AIC = "aic"
    CPUS = "cpus"
    STRUCTURE_PENALTY = "structure_penalty"

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
            frontiers: {split : Frontiers object}
            kwargs: additional pre-serialized objects dictionary.

        Returns: deserialized JSON response."""
        if compressor_type != self.DEFAULT_COMPRESSOR_TYPE:
            # Legacy Dreamcoder library supports Python enumeration
            raise NotImplementedError
        # Construct the JSON message.

        grammar = grammar if grammar is not None else self
        json_serialized_binary_message = {
            self.API_FN: str(api_fn),  # String name of the API function.
            self.REQUIRED_ARGS: {
                self.GRAMMAR: grammar.json(),
                self.FRONTIERS: {split: [] for split in frontiers},
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
            print("Error in _send_receive_compressor_api_call: {e}")

        # Deserialize the response.
        json_deserialized_response = json_response

        json_deserialized_response[self.REQUIRED_ARGS][self.GRAMMAR] = [
            self._deserialize_json_grammar(serialized_grammar)
            for serialized_grammar in json_response[self.REQUIRED_ARGS][self.GRAMMAR]
        ]
        # TODO: deserialize the frontiers

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

    def _get_compressed_grammmar_and_rewritten_frontiers(
        self,
        experiment_state,
        task_splits,
        task_ids_in_splits,
        max_candidates_per_compression_step,
        max_compression_steps,
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
        API Function: get_compressed_grammmar_and_rewritten_frontiers.
        Call the compressor to rewrite the grammar and the frontiers.

        Runs compression up to max_compression_steps, evaluating max_candidates_per_compression_step under the compression score, and greedily adding the top candidate each time.
        Always assumes it should only optimize with respect to the TRAIN frontiers, but rewrites train/test frontiers under the compressed grammar.
        :ret:
            grammar: Grammar object containing the final DSL with up to max_compression_steps new library inventions.
            frontiers: {split : [frontiers] with frontiers rewritten under the final grammar.}

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
        }

        api_response = self._send_receive_compressor_api_call(
            api_fn=api_fn,
            grammar=self,
            frontiers=frontiers,
            kwargs=kwargs,
            compressor_type=compressor_type,
            compressor=compressor,
            compressor_directory=compressor_directory,
        )
        # final_grammar = api_response[self.GRAMMAR][0]
        # rewritten_frontiers = api_response[self.FRONTIERS][0]
        # return final_grammar, rewritten_frontiers

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
