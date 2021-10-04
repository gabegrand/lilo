"""
evaluate_compression_model_scoring.py | Author : Catherine Wong.

Evaluation for new models that score candidate library functions. Runs simulated iterations of compression using ground truth programs.

Usage: 
python evaluate_compression_model_scoring.py
    --config_file syntax_robustfill_language_only_compositional_graphics_200_synthetic.json # Config to initialize the model and the dataset. Assumes the amortized synthesis model is the one with the scoring function.
    --library_candidates_scoring_fn # Which scoring function you want to use.
    -k # Pytest keywordsds for which test to run (substring of tests), otherwise runs all. 
       [test_discrimination_original_final_libraries_full,
       test_discrimination_candidate_alignments,
       test_heldout_scores_with_model_reranking]

# TODO: catwong: implement simple pickle caching based on the test name and iteration. (We can also re-use the cache between tests over candidates.)

"""
import time
import os, json, argparse
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

from collections import defaultdict

from src.experiment_iterator import ExperimentState
from src.task_loaders import TRAIN, TEST, ALL
from src.models.model_loaders import *

from data.compositional_graphics.make_tasks import *

# All of the model loaders we import.
from data.compositional_graphics.grammar import *
from data.compositional_graphics.encoder import *

from src.models.laps_dreamcoder_recognition import *
from src.models.syntax_robustfill import *

DEFAULT_CONFIG_DIR = "experiments/configs"
DEFAULT_OUTPUT_DIR = "experiments/outputs/evaluate_compression_model"

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
    "--db_no_compression",
    action="store_true",
    help="Debug flag: avoids expensive compression runs.",
)
parser.add_argument(
    "--db_no_model_training",
    action="store_true",
    help="Debug flag: avoids training the model.",
)
parser.add_argument("-k", nargs="+", help="Substring keywords of tests to run.")

TEST_FUNCTIONS_REGISTRY = {}

MODEL_SCORE, COMPRESSOR_SCORE = "model score", "compressor score"
MODEL_RANK, COMPRESSOR_RANK = "model rank", "compressor rank"
FRONTIERS = "frontiers"


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
def test_discrimination_original_final_libraries_full(
    args,
    config,
    num_training_buckets=5,  # How many buckets to make of the training programs.
    max_candidates_per_compression_step=100,
    arity=2,
    max_compression_steps=3,
):
    """Tests whether the model scoring function can discriminate at all between the initial DSL and the final DSL over all of the training and test programs."""

    initial_ground_truth_experiment_state = get_initial_ground_truth_experiment_state(
        config
    )

    INITIAL, COMPRESSED = "initial", "compressed"
    NUM_TRAIN_TASKS = "# training tasks"
    TEST_LOG_LIKELIHOOD = "Test log likelihood"
    metrics_to_report = {
        model_header: defaultdict(list) for model_header in [INITIAL, COMPRESSED]
    }

    for train_iteration, train_task_subset in make_program_log_prior_buckets_iterator(
        initial_ground_truth_experiment_state,
        task_split=TRAIN,
        num_buckets=num_training_buckets,
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
                max_candidates_per_compression_step=max_candidates_per_compression_step,
                max_compression_steps=max_compression_steps,
                arity=arity,
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
                model.optimize_model_for_frontiers(
                    experiment_state,
                    task_split=TRAIN,
                    task_batch_ids=[t.name for t in train_task_subset],
                    # TODO: @gg - add any other hyperparameters you need here.
                )

                # Evaluate it with respect to the test tasks.
                test_frontier_log_likelihoods = (
                    model.score_frontier_avg_conditional_log_likelihoods(
                        experiment_state, task_split=TEST, task_batch_ids=ALL
                    )
                )

            # Report on likelihoods.
            print(
                f"Evaluated model on {header} library: test likelihoods = {np.mean(test_frontier_log_likelihoods)}"
            )
            metrics_to_report[header][TEST_LOG_LIKELIHOOD].append(
                (test_frontier_log_likelihoods)
            )

            # Generate intermediate curve.
            experiment_id = experiment_state.metadata["experiment_id"]
            generate_rel_plot(
                args,
                metrics_to_report,
                x_titles=[NUM_TRAIN_TASKS],
                y_titles=[TEST_LOG_LIKELIHOOD],
                plot_title="test_discrimination_original_final_libraries_full"
                + experiment_id,
            )


@register_test
def test_discrimination_candidate_alignments(
    args,
    config,
    num_training_buckets=5,  # How many buckets to make of the training programs.
    max_candidates_per_compression_step=100,
    max_grammar_candidates_to_retain_for_rewriting=4,  # How many candidates to actually return for evaluating.
    arity=2,
):
    """Tests whether the model scoring function can meaningfully rerank a set of proposed DSL candidates and compare this ranking to that produced by the compressor.

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
        num_buckets=num_training_buckets,
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
            max_candidates_per_compression_step,
            max_grammar_candidates_to_retain_for_rewriting,
            arity,
            compress_test_frontiers=False,
        )

        report_model_compressor_score_agreement(
            train_iteration, candidate_grammars_to_scores
        )


@register_test
def test_heldout_scores_with_model_reranking(
    args,
    config,
    num_training_buckets=5,  # How many buckets to make of the training programs.
    max_candidates_per_compression_step=100,
    max_grammar_candidates_to_retain_for_rewriting=4,  # How many candidates to actually return for evaluating.
    arity=2,
):
    """
    Tests whether the candidates ranked by the model (a) improve model likelihood scores over the test set and (b) improve grammar likelihood scores over the test set. Note that this only tests adding the single top candidate to the model -- alternatively we could add the first one where they diverge.
    """
    initial_ground_truth_experiment_state = get_initial_ground_truth_experiment_state(
        config
    )
    for train_iteration, train_task_subset in make_program_log_prior_buckets_iterator(
        initial_ground_truth_experiment_state,
        task_split=TRAIN,
        num_buckets=num_training_buckets,
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
            max_candidates_per_compression_step,
            max_grammar_candidates_to_retain_for_rewriting,
            arity,
            compress_test_frontiers=False,
        )

        report_model_compressor_score_agreement(
            train_iteration, candidate_grammars_to_scores
        )

        # TODO (catwong): we should also return the best MODEL and then evaluate it on the test tasks.


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


def generate_rel_plot(
    args, metrics_to_report, x_titles, y_titles, plot_title, y_lim=1.0
):
    for y_title in y_titles:
        for x_title in x_titles:

            def build_dataframe(metrics_to_report, x_title, y_title):
                xs = []
                ys = []
                model = []
                for legend in metrics_to_report:
                    num_iterations = len(metrics_to_report[legend][x_title])

                    for iteration in range(num_iterations):
                        iter_ys = metrics_to_report[legend][y_title][iteration]
                        iter_xs = [metrics_to_report[legend][x_title][iteration]] * len(
                            iter_ys
                        )

                        xs += iter_xs
                        ys += iter_ys
                        model += [legend] * len(iter_ys)
                d = {
                    f"{x_title}": xs,
                    f"{y_title}": ys,
                    "Model": model,
                }
                return pd.DataFrame(data=d)

            plt.figure(figsize=(6, 3))
            df = build_dataframe(metrics_to_report, x_title, y_title)
            ax = sns.relplot(
                x=f"{x_title}",
                y=f"{y_title}",
                hue="Model",
                style="Model",
                kind="line",
                data=df,
            )
            ax.fig.set_size_inches(12, 3)
            ax.axes[0, 0].set_ylim(0, y_lim)
            plt.title(f"{y_title}")

            escaped_y_title = y_title.lower().replace(" ", "_")
            output_title = f"{plot_title}_{escaped_y_title}.png"
            output_name = os.path.join(args.output_dir, output_title)

            print(f"Writing plot out to: {output_name}")
            plt.savefig(output_name)


def get_compressor_candidates_and_model_reranking(
    args,
    config,
    train_iteration,
    train_task_subset,
    max_candidates_per_compression_step,
    max_grammar_candidates_to_retain_for_rewriting,  # How many candidates to actually return for evaluating.
    arity,
    compress_test_frontiers=False,
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
    grammars_scores_frontiers = compressed_experiment_state.models[
        GRAMMAR
    ]._get_compressed_grammar_candidates_and_rewritten_frontiers(
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
        if args.db_no_model_training:
            print("[DEBUG]: skipping model training.")
            test_model_candidate_score = -1.0
        else:
            # TODO: @gg: this should actually run cross-validation on the training frontiers and produce a score for the resulting grammar.
            model.optimize_model_for_frontiers(
                candidate_experiment_state,
                task_split=TRAIN,
                task_batch_ids=[t.name for t in train_task_subset],
                # TODO: @gg - add any other hyperparameters you need here.
            )

            test_model_candidate_score = (
                model.score_frontier_avg_conditional_log_likelihoods(
                    candidate_experiment_state,
                    task_split=TRAIN,
                    task_batch_ids=[t.name for t in train_task_subset],
                )
            )
        # Add their score.
        candidate_grammars_to_scores[candidate_grammar][
            MODEL_SCORE
        ] = test_model_candidate_score
        candidate_grammars_to_scores[candidate_grammar][
            COMPRESSOR_SCORE
        ] = candidate_score
        candidate_grammars_to_scores[candidate_grammar][FRONTIERS] = candidate_frontiers
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


def main(args):
    config = load_config_from_file(args)

    test_fns = get_test_fns(args)

    print(f"Now running {len(test_fns)} tests...")
    for idx, test_fn in enumerate(test_fns):
        print(f"Running {idx} / {len(test_fns)}: {test_fn.__name__}")
        test_fn(args, config)


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
