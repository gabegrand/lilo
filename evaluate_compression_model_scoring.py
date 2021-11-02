"""
evaluate_compression_model_scoring.py | Author : Catherine Wong.

Evaluation for new models that score candidate library functions. Runs simulated iterations of compression using ground truth programs.

Usage: 
python evaluate_compression_model_scoring.py
    --config_file seq2seq_language_only_compositional_graphics_200_synthetic.json # Config to initialize the model and the dataset. Assumes the amortized synthesis model is the one with the scoring function.
    --library_candidates_scoring_fn # Which scoring function you want to use.
    -k # Pytest keywordsds for which test to run (substring of tests), otherwise runs all. 
       [test_discrimination_original_final_libraries_full,
       test_discrimination_candidate_alignments,
       test_heldout_scores_with_model_reranking]

# TODO: catwong: implement simple pickle caching based on the test name and iteration. (We can also re-use the cache between tests over candidates.)

"""
import argparse
import json
import os
import time
from collections import defaultdict

import numpy as np

from data.compositional_graphics.encoder import *

# All of the model loaders we import.
from data.compositional_graphics.grammar import *
from data.compositional_graphics.make_tasks import *
from src.experiment_iterator import ExperimentState
from src.models.laps_dreamcoder_recognition import *
from src.models.model_loaders import *
from src.models.seq2seq import *
from src.task_loaders import ALL, TEST, TRAIN
from src.utils import *

DEFAULT_CONFIG_DIR = "experiments/configs"
DEFAULT_OUTPUT_DIR = "experiments/outputs/evaluate_compression_model"

# Default hyperparamters for the evaluations
DEFAULT_NUM_TRAINING_BUCKETS = 10
DEFAULT_MAX_COMPRESSION_STEPS = 5
DEFAULT_MAX_CANDIDATES_PER_COMPRESSION_STEP = 100
DEFAULT_MAX_GRAMMAR_CANDIDATES_TO_RETAIN_FOR_REWRITING = 10
DEFAULT_ARITY = 2
DEFAULT_PARALLEL_GRAMMAR_CANDIDATES = 0

parser = argparse.ArgumentParser()
parser.add_argument(
    "--config_dir",
    default=DEFAULT_CONFIG_DIR,
    help="Top level directory containing experiment config files.",
)
parser.add_argument(
    "--config_file",
    required=True,
    help="File name of the config within the config directory.",
)
parser.add_argument(
    "--output_dir",
    default=DEFAULT_OUTPUT_DIR,
    help="Top level directory for where to output results.",
)
parser.add_argument(
    "--util_generate_cloud_command",
    default=None,
    help="If provided, generates a command for running this in the cloud instead of actually running locally.",
)
parser.add_argument(
    "--db_no_compression",
    action="store_true",
    help="Debug flag: avoids expensive compression runs.",
)
parser.add_argument(
    "--db_no_model_training",
    action="store_true",
    help="Debug flag: avoids training the model.",
)

# Hyperparameters for the tests and compressor.
parser.add_argument(
    "--hp_num_training_buckets",
    type=int,
    default=DEFAULT_NUM_TRAINING_BUCKETS,
    help="Test hyperparameter: how many buckets to make of the training tasks when evaluating the model on increasing batches of tasks.",
)
parser.add_argument(
    "--hp_max_compression_steps",
    type=int,
    default=DEFAULT_MAX_COMPRESSION_STEPS,
    help="Test hyperparameter: maximum number of candidates to add to the grammar each time.",
)
parser.add_argument(
    "--hp_max_candidates_per_compression_step",
    type=int,
    default=DEFAULT_MAX_CANDIDATES_PER_COMPRESSION_STEP,
    help="Test hyperparameter: how many candidates the compressor considers before choosing the top-k to rank the global compressor score.",
)
parser.add_argument(
    "--hp_max_grammar_candidates_to_retain_for_rewriting",
    type=int,
    default=DEFAULT_MAX_CANDIDATES_PER_COMPRESSION_STEP,
    help="Test hyperparameter: how many candidates to return in which we actually rewrite the grammar and the frontiers.",
)
parser.add_argument(
    "--hp_arity",
    type=int,
    default=DEFAULT_ARITY,
    help="Test hyperparameter: arity: maximum arity that we consider for candidate library functions.",
)
parser.add_argument(
    "--hp_parallel_grammar_candidate_rewriting",
    type=int,
    default=DEFAULT_PARALLEL_GRAMMAR_CANDIDATES,
    help="Test hyperparameter: whether or not to enumerate different grammar candidates in parallel. This consumes more memory but takes less time. Use int [0, 1] to indicate boolean True, False",
)
parser.add_argument("--k", nargs="+", help="Substring keywords of tests to run.")

TEST_FUNCTIONS_REGISTRY = {}

# String constants for reporting.
MODEL_SCORE, COMPRESSOR_SCORE = "model score", "compressor score"
MODEL_RANK, COMPRESSOR_RANK = "model rank", "compressor rank"
FRONTIERS = "frontiers"
MODEL_TEST_LIKELIHOODS = "model_test_likelihoods"
TOP_K_METRIC = "Top {} {}"
INITIAL, COMPRESSED = "initial", "compressed"
NUM_TRAIN_TASKS = "# training tasks"
TEST_CROSS_ENTROPY_LOSS = "Test cross entropy loss"


def register_test(test_fn):
    TEST_FUNCTIONS_REGISTRY[test_fn.__name__] = test_fn


def get_test_fns(args):
    test_fns = []
    if not args.k:
        test_fns = TEST_FUNCTIONS_REGISTRY.values()
        return test_fns
    for keyword in args.k:
        for test_fn_name, test_fn in TEST_FUNCTIONS_REGISTRY.items():
            if keyword in test_fn_name:
                test_fns.append(test_fn)
    return test_fns


@register_test
def test_discrimination_original_final_libraries_full(args, config):
    """Tests whether the model scoring function can discriminate at all between the initial DSL and the final DSL over all of the training and test programs.
    Formally: reports p(test_programs | model, language, L_0) vs. p(test_programs | model, language, L_f) where L_f is derived via the original DreamCoder compressor.
    """

    initial_ground_truth_experiment_state = get_initial_ground_truth_experiment_state(
        config
    )
    metrics_to_report = {
        model_header: defaultdict(list) for model_header in [INITIAL, COMPRESSED]
    }

    for train_iteration, train_task_subset in make_program_log_prior_buckets_iterator(
        initial_ground_truth_experiment_state,
        task_split=TRAIN,
        num_buckets=args.hp_num_training_buckets,
    ):
        print(
            f"Iteration: {train_iteration}. Train task subset: {len(train_task_subset)} tasks: up to {train_task_subset[-1].name}"
        )
        # Make the comparison experiment_state by compressing the frontiers.
        compressed_experiment_state = get_initial_ground_truth_experiment_state(config)

        if args.db_no_compression:
            print("[DEBUG]: skipping library compression.")
        else:
            # Get the compression candidates and rewrite the test set.
            start_time = time.time()
            compressed_experiment_state.models[
                GRAMMAR
            ]._get_compressed_grammmar_and_rewritten_frontiers(
                experiment_state=compressed_experiment_state,
                task_splits=[TRAIN, TEST],
                task_ids_in_splits={
                    TRAIN: [t.name for t in train_task_subset],
                    TEST: ALL,
                },
                max_candidates_per_compression_step=args.hp_max_candidates_per_compression_step,
                max_compression_steps=args.hp_max_compression_steps,
                arity=args.hp_arity,
            )
            print(
                f"[DEBUG]: compression - {train_iteration} - took {(time.time() - start_time)} s."
            )

        for (header, experiment_state) in [
            (INITIAL, initial_ground_truth_experiment_state),
            (COMPRESSED, compressed_experiment_state),
        ]:
            metrics_to_report[header][NUM_TRAIN_TASKS].append(len(train_task_subset))

            # Train the model with respect to the train task subset.
            model = experiment_state.models[AMORTIZED_SYNTHESIS]

            if args.db_no_model_training:
                print("[DEBUG]: skipping model training.")
                test_frontier_log_likelihoods = [0.0]
            else:
                start_time = time.time()
                run_results_per_epoch = model.optimize_model_for_frontiers(
                    experiment_state,
                    task_split=TRAIN,
                    task_batch_ids=[t.name for t in train_task_subset],
                    # TODO: @gg - add any other hyperparameters you need here.
                )
                print(
                    f"[DEBUG]: Model training ({header}, {len(train_task_subset)} tasks) took {(time.time() - start_time)} s."
                )

                # Save training run results to disk
                results_path = os.path.join(
                    args.output_dir,
                    "training",
                    header,
                    f"{str(len(train_task_subset)).zfill(4)}_tasks.json",
                )
                os.makedirs(os.path.dirname(results_path), exist_ok=True)
                with open(results_path, "w") as f:
                    json.dump(run_results_per_epoch, f)

                # Evaluate it with respect to the test tasks.
                test_results = model.score_frontier_avg_conditional_log_likelihoods(
                    experiment_state, task_split=TEST, task_batch_ids=ALL
                )
                test_frontier_log_likelihoods = list(
                    test_results["loss_per_task"].values()
                )

                # Save test run results to disk
                results_path = os.path.join(
                    args.output_dir,
                    "test",
                    header,
                    f"{str(len(train_task_subset)).zfill(4)}_tasks.json",
                )
                os.makedirs(os.path.dirname(results_path), exist_ok=True)
                with open(results_path, "w") as f:
                    json.dump(test_results, f)

            # Report on likelihoods.
            print(
                f"Evaluated model on {header} library: test likelihoods = {np.mean(test_frontier_log_likelihoods)}"
            )
            metrics_to_report[header][TEST_CROSS_ENTROPY_LOSS].append(
                (test_frontier_log_likelihoods)
            )

            # Generate intermediate curve.
            experiment_id = experiment_state.metadata["experiment_id"]
            generate_rel_plot(
                args,
                metrics_to_report,
                x_titles=[NUM_TRAIN_TASKS],
                y_titles=[TEST_CROSS_ENTROPY_LOSS],
                plot_title="test_discrimination_original_final_libraries_full"
                + experiment_id,
                y_lim=None,
            )


@register_test
def test_discrimination_candidate_alignments(args, config):
    """Tests whether the model scoring function can meaningfully rerank a set of proposed DSL candidates and compare this ranking to that produced by the compressor.

    Formally: ranks candidate DSLs L_i_0...L_i_n (where n=max_grammar_candidates_to_retain_for_rewriting) according to the compressor score and the model score.

    Reports:
        What ranking was given by the model for the top-k best compressor candidates.
        What the ranking was given by the model for the top-k best model candidates.
    """
    initial_ground_truth_experiment_state = get_initial_ground_truth_experiment_state(
        config
    )
    for train_iteration, train_task_subset in make_program_log_prior_buckets_iterator(
        initial_ground_truth_experiment_state,
        task_split=TRAIN,
        num_buckets=args.hp_num_training_buckets,
    ):
        print(
            f"Iteration: {train_iteration}. Train task subset: {len(train_task_subset)} tasks: up to {train_task_subset[-1].name}"
        )

        (
            candidate_grammars_to_scores,
            _,
            _,
        ) = get_compressor_candidates_and_model_reranking(
            args,
            config,
            train_iteration,
            train_task_subset,
            args.hp_max_candidates_per_compression_step,
            args.hp_max_grammar_candidates_to_retain_for_rewriting,
            args.hp_arity,
            compress_test_frontiers=False,
            parallel_grammar_candidate_rewriting=bool(
                args.hp_parallel_grammar_candidate_rewriting
            ),
        )

        report_model_compressor_score_agreement(
            train_iteration, candidate_grammars_to_scores
        )


@register_test
def test_heldout_scores_with_model_reranking(
    args,
    config,
    top_k_candidates_to_evaluate_on_heldout=10,  # Compare this many candidates.
):
    """
    Evaluates the top-k grammar candidates proposed by the model vs. the top-k grammar candidates proposed by the compressor.

    Tests whether the candidates ranked by the model (a) improve model likelihood scores over the test set and (b) improve grammar likelihood scores over the test set. Note that this only tests the top-k, which may not diverge.

    TODO(@catwong): we could alternately evaluate the first one where they diverge.
    """
    # Report best and top-K
    metrics_to_report = {
        model_header: defaultdict(list)
        for model_header in [
            str.format(TOP_K_METRIC, 1, COMPRESSOR_SCORE),
            str.format(TOP_K_METRIC, 1, MODEL_SCORE),
            str.format(
                TOP_K_METRIC, top_k_candidates_to_evaluate_on_heldout, COMPRESSOR_SCORE
            ),
            str.format(
                TOP_K_METRIC, top_k_candidates_to_evaluate_on_heldout, MODEL_SCORE
            ),
        ]
    }

    initial_ground_truth_experiment_state = get_initial_ground_truth_experiment_state(
        config
    )
    experiment_id = initial_ground_truth_experiment_state.metadata["experiment_id"]
    for train_iteration, train_task_subset in make_program_log_prior_buckets_iterator(
        initial_ground_truth_experiment_state,
        task_split=TRAIN,
        num_buckets=args.hp_num_training_buckets,
    ):
        print(
            f"Iteration: {train_iteration}. Train task subset: {len(train_task_subset)} tasks: up to {train_task_subset[-1].name}"
        )

        (
            candidate_grammars_to_scores,
            compressor_sorted_grammars,
            model_sorted_grammars,
        ) = get_compressor_candidates_and_model_reranking(
            args,
            config,
            train_iteration,
            train_task_subset,
            args.hp_max_candidates_per_compression_step,
            args.hp_max_grammar_candidates_to_retain_for_rewriting,
            args.hp_arity,
            compress_test_frontiers=True,
            evaluate_test_model_likelihoods=True,
        )

        report_model_baseline_top_k_candidates_heldout_likelihoods(
            args,
            experiment_id,
            metrics_to_report,
            len(train_task_subset),
            candidate_grammars_to_scores,
            compressor_sorted_grammars,
            model_sorted_grammars,
            top_k_candidates_to_evaluate_on_heldout,
        )


def report_model_baseline_top_k_candidates_heldout_likelihoods(
    args,
    experiment_id,
    metrics_to_report,
    num_iteration_train_tasks,  # How many training tasks at this iteration.
    candidate_grammars_to_scores,
    compressor_sorted_grammars,
    model_sorted_grammars,
    top_k_candidates_to_evaluate_on_heldout,
):
    """
    Reports the top-k heldout likelihoods under the model vs. the top-k under the baseline.
    """
    for top_k_candidates, score_type in [
        (
            compressor_sorted_grammars[:top_k_candidates_to_evaluate_on_heldout],
            COMPRESSOR_SCORE,
        ),
        (model_sorted_grammars[:top_k_candidates_to_evaluate_on_heldout], MODEL_SCORE),
    ]:
        mean_heldout_likelihoods = [
            np.mean(candidate_grammars_to_scores[candidate][MODEL_TEST_LIKELIHOODS])
            for candidate in top_k_candidates
        ]
        # Report best
        best_header = str.format(TOP_K_METRIC, 1, score_type)
        metrics_to_report[best_header][MODEL_TEST_LIKELIHOODS].append(
            [mean_heldout_likelihoods[0]]
        )
        metrics_to_report[best_header][NUM_TRAIN_TASKS].append(
            num_iteration_train_tasks
        )

        # Report mean over top-k
        top_k_header = str.format(
            TOP_K_METRIC, top_k_candidates_to_evaluate_on_heldout, score_type
        )
        metrics_to_report[top_k_header][MODEL_TEST_LIKELIHOODS].append(
            mean_heldout_likelihoods
        )
        metrics_to_report[top_k_header][NUM_TRAIN_TASKS].append(
            num_iteration_train_tasks
        )
    # Generate intermediate curve.
    generate_rel_plot(
        args,
        metrics_to_report,
        x_titles=[NUM_TRAIN_TASKS],
        y_titles=[MODEL_TEST_LIKELIHOODS],
        plot_title="test_heldout_scores_with_model_reranking" + experiment_id,
    )


def make_program_log_prior_buckets_iterator(
    experiment_state,
    task_split,
    num_buckets,
):
    """Iterator over num_buckets buckets of tasks with ground truth programs by log_prior under the grammar (as a corollary for description length)."""

    def best_log_prior(task):
        frontier = experiment_state.task_frontiers[task_split][task]
        return min([e.logPrior for e in frontier.entries])

    sorted_log_prior = sorted(
        experiment_state.task_frontiers[task_split],
        key=lambda task: best_log_prior(task),
        reverse=True,
    )

    batch_size = int(len(sorted_log_prior) / num_buckets)
    for bucket_idx in range(num_buckets + 1):
        end = (bucket_idx + 1) * batch_size
        yield bucket_idx, sorted_log_prior[:end]


def get_initial_ground_truth_experiment_state(config):
    experiment_state = ExperimentState(config)
    experiment_state.initialize_ground_truth_task_frontiers(task_split=TRAIN)
    experiment_state.initialize_ground_truth_task_frontiers(task_split=TEST)
    return experiment_state


def get_experiment_state_grammar_frontiers(config, grammar, frontiers):
    initial_experiment_state = get_initial_ground_truth_experiment_state(config)
    initial_experiment_state.models[model_loaders.GRAMMAR] = grammar
    for task_split in frontiers:
        for rewritten_frontier in frontiers[task_split]:
            initial_experiment_state.task_frontiers[task_split][
                rewritten_frontier.task
            ] = rewritten_frontier
    return initial_experiment_state


def get_compressor_candidates_and_model_reranking(
    args,
    config,
    train_iteration,
    train_task_subset,
    max_candidates_per_compression_step,
    max_grammar_candidates_to_retain_for_rewriting,  # How many candidates to actually return for evaluating.
    arity,
    compress_test_frontiers=False,
    evaluate_test_model_likelihoods=False,
    parallel_grammar_candidate_rewriting=False,
):
    """
    Compresses grammar with respect to frontiers in train_task_subset to produce max_grammar_candidates_to_retain_for_rewriting grammar candidates.
    Evaluates the model to produce a score with respect to each candidate.

    Returns:
        {
            grammar: {
                model_score, model_ranking, compressor_score, compressor_ranking, rewritten_frontiers
            }
        }
        sorted_compressor_grammars: grammars sorted by the compressor ranking.
        sorted_model_grammars: grammars sorted by the model ranking.

    # TODOs (@CathyWong): de-couple the compressor from the
    # Wrap it in a thread and re-call if it seems stuck? Or allow it to send responses? Since it seems like OM interrupts.
    """
    candidate_grammars_to_scores = defaultdict(
        lambda: {
            k: None
            for k in [
                MODEL_SCORE,
                MODEL_RANK,
                COMPRESSOR_SCORE,
                COMPRESSOR_RANK,
                FRONTIERS,
                MODEL_TEST_LIKELIHOODS,
            ]
        }
    )
    # Make the comparison experiment_state by compressing the frontiers.
    compressed_experiment_state = get_initial_ground_truth_experiment_state(config)

    # Get the compression candidates.
    if args.db_no_compression:
        print("[DEBUG]: skipping library compression.")

    # Get the compression candidates and rewrite the test set.
    start_time = time.time()
    # Use the memory intensive
    if parallel_grammar_candidate_rewriting:
        print(
            "Using the parallel implementation to rewrite the grammar candidates and frontiers."
        )
        candidate_rewriting_fn = compressed_experiment_state.models[
            GRAMMAR
        ]._get_compressed_grammar_candidates_and_rewritten_frontiers_parallel
    else:
        print(
            "Using the non-parallel implementation to rewrite the grammar candidates and frontiers."
        )
        candidate_rewriting_fn = compressed_experiment_state.models[
            GRAMMAR
        ]._get_compressed_grammar_candidates_and_rewritten_frontiers

    grammars_scores_frontiers = candidate_rewriting_fn(
        experiment_state=compressed_experiment_state,
        task_splits=[TRAIN, TEST],
        task_ids_in_splits={
            TRAIN: [t.name for t in train_task_subset],
            TEST: [] if not compress_test_frontiers else ALL,
        },
        max_candidates_per_compression_step=max_candidates_per_compression_step,
        max_grammar_candidates_to_retain_for_rewriting=max_grammar_candidates_to_retain_for_rewriting,
        arity=arity,
        debug_get_dummy=args.db_no_compression,
    )
    print(
        f"[DEBUG]: compression - {train_iteration} - took {(time.time() - start_time)} s."
    )

    # Train the model against each of these candidates and get a score.
    for candidate_idx, candidate_grammar_score_frontier in enumerate(
        grammars_scores_frontiers
    ):
        print(
            f"Training model to evaluate candidate grammar {candidate_idx}/{len(grammars_scores_frontiers)}"
        )
        candidate_grammar, candidate_frontiers, candidate_score = (
            candidate_grammar_score_frontier["grammar"],
            candidate_grammar_score_frontier["frontiers"],
            candidate_grammar_score_frontier["compression_scores"],
        )
        candidate_experiment_state = get_experiment_state_grammar_frontiers(
            config, grammar=candidate_grammar, frontiers=candidate_frontiers
        )

        model = candidate_experiment_state.models[AMORTIZED_SYNTHESIS]
        model_test_likelihoods = None
        if args.db_no_model_training:
            print("[DEBUG]: skipping model training.")
            model_candidate_score = -1.0
            model_test_likelihoods = [0.0]
        else:
            # TODO: @gg: this should actually run cross-validation on the training frontiers and produce a score for the resulting grammar. Swap out the below.
            model_candidate_score = model.optimize_model_for_frontiers(
                candidate_experiment_state,
                task_split=TRAIN,
                task_batch_ids=[t.name for t in train_task_subset],
                # TODO: @gg - add any other hyperparameters you need here.
            )

            # Evaluate how the model actually would have done on the heldout programs.
            if evaluate_test_model_likelihoods:
                model_test_likelihoods = (
                    model.score_frontier_avg_conditional_log_likelihoods(
                        candidate_experiment_state,
                        task_split=TEST,
                        task_batch_ids=ALL,
                    )
                )
        # Add their scores under the model vs. the compressor.
        candidate_grammars_to_scores[candidate_grammar][
            MODEL_SCORE
        ] = model_candidate_score
        candidate_grammars_to_scores[candidate_grammar][
            COMPRESSOR_SCORE
        ] = candidate_score
        candidate_grammars_to_scores[candidate_grammar][FRONTIERS] = candidate_frontiers

        # Add the model likelihoods over the heldout test set if we are evaluating this.
        candidate_grammars_to_scores[candidate_grammar][
            MODEL_TEST_LIKELIHOODS
        ] = model_test_likelihoods
    # Rank them with respect to their scores.
    all_sorted_grammars = {}
    for score_type, rank_type in [
        (MODEL_SCORE, MODEL_RANK),
        (COMPRESSOR_SCORE, COMPRESSOR_RANK),
    ]:
        sorted_grammars = sorted(
            candidate_grammars_to_scores,
            key=lambda candidate_grammar: candidate_grammars_to_scores[
                candidate_grammar
            ][score_type],
        )
        all_sorted_grammars[score_type] = sorted_grammars
        for (idx, g) in enumerate(sorted_grammars):
            candidate_grammars_to_scores[g][rank_type] = idx
    return (
        candidate_grammars_to_scores,
        all_sorted_grammars[COMPRESSOR_SCORE],
        all_sorted_grammars[MODEL_SCORE],
    )


def report_model_compressor_score_agreement(
    train_iteration, candidate_grammars_to_scores, report_top_k=3
):
    print(
        f"Iteration: {train_iteration} Reporting model vs. compressor score agreement"
    )
    for score_type, rank_type, comparison_score_type, comparison_rank_type in [
        (MODEL_SCORE, MODEL_RANK, COMPRESSOR_SCORE, COMPRESSOR_RANK),
        (COMPRESSOR_SCORE, COMPRESSOR_RANK, MODEL_SCORE, MODEL_RANK),
    ]:
        sorted_grammars = sorted(
            candidate_grammars_to_scores,
            key=lambda candidate_grammar: candidate_grammars_to_scores[
                candidate_grammar
            ][score_type],
        )
        print(f"Reporting top-k by {score_type}")
        for (idx, g) in enumerate(sorted_grammars[:report_top_k]):
            main_score = candidate_grammars_to_scores[g][score_type]
            comparison_score = candidate_grammars_to_scores[g][comparison_score_type]
            comparison_rank = candidate_grammars_to_scores[g][comparison_rank_type]
            print(
                f"]\tRank {idx} w/ {score_type}: {main_score} | comparison: rank = {comparison_rank} w/ {comparison_score_type} : {comparison_score}"
            )


def load_config_from_file(args):
    config_full_path = os.path.join(args.config_dir, args.config_file)
    with open(config_full_path) as f:
        return json.load(f)


def build_cloud_job_name():
    """
    Builds job and logfile name as: {CONCATENATED_TEST_NAMES}_{EXPERIMENT_ID}_{TIMESTAMP}.
    """
    test_fns = TEST_FUNCTIONS_REGISTRY.values() if not args.k else args.k
    concatenated_test_names = "_".join(test_fns)

    config = load_config_from_file(args)
    experiment_id = config["metadata"]["experiment_id"]
    timestamp = escaped_timestamp()
    return f"{concatenated_test_names}_{experiment_id}_{timestamp}"


def main(args):
    config = load_config_from_file(args)

    test_fns = get_test_fns(args)

    print(f"Now running {len(test_fns)} tests...")
    for idx, test_fn in enumerate(test_fns):
        print(f"Running {idx} / {len(test_fns)}: {test_fn.__name__}")
        print_hyperparameter_arguments(args)
        test_fn(args, config)


if __name__ == "__main__":
    args = parser.parse_args()
    if args.util_generate_cloud_command is not None:
        generate_cloud_command(
            source_python_file=os.path.basename(__file__),
            output_dir=args.output_dir,
            job_name=build_cloud_job_name(),
            args=args,
        )
    else:
        main(args)
