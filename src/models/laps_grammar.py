"""
laps_grammmar.py | Author : Catherine Wong

Utility wrapper function around the DreamCoder Grammar. Elevates common functions to be class functions in order to support calling with an ExperimentState.
"""
import os

from dreamcoder.grammar import Grammar
from dreamcoder.enumeration import multicoreEnumeration
from dreamcoder.dreaming import backgroundHelmholtzEnumeration
from dreamcoder.compression import induceGrammar

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
    DEFAULT_MAX_MEM_PER_ENUMERATION_THREAD = (
        1000000000  # Max memory usage per thread
    )
    DEFAULT_MAX_SAMPLES = 5000

    # Compression hyperparameters
    DEFAULT_TOP_K = 2
    DEFAULT_PSEUDOCOUNTS = 30.0
    DEFAULT_ARITY = 3
    DEFAULT_AIC = 1.0
    DEFAULT_STRUCTURE_PENALTY = 1.5
    DEFAULT_COMPRESSOR_TYPE = "ocaml"
    DEFAULT_COMPRESSOR = "compression"
    DEFAULT_MAX_COMPRESSION = 1000

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
            experiment_state.sample_frontiers[
                sampled_frontier.task
            ] = sampled_frontier
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
