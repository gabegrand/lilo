"""
evaluate_compression_model_scoring.py | Author : Catherine Wong.

Evaluation for new models that score candidate library functions. Runs simulated iterations of compression using ground truth programs.

Usage: 
python evaluate_compression_model_scoring.py
    --config_file syntax_robustfill_language_only_compositional_graphics_200_synthetic.json # Config to initialize the model and the dataset. Assumes the amortized synthesis model is the one with the scoring function.
    --library_candidates_scoring_fn # Which scoring function you want to use.
    -k # Pytest keywrods for which test to run (substring of tests), otherwise runs all. 

"""
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
parser.add_argument("-k", nargs="+", help="Substring keywords of tests to run.")

TEST_FUNCTIONS_REGISTRY = {}


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
        yield sorted_log_prior[:end]


def get_initial_ground_truth_experiment_state(config):
    experiment_state = ExperimentState(config)
    experiment_state.initialize_ground_truth_task_frontiers(task_split=TRAIN)
    experiment_state.initialize_ground_truth_task_frontiers(task_split=TEST)
    return experiment_state


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


@register_test
def test_discrimination_original_final_libraries_full(
    args,
    config,
    num_training_buckets=5,  # How many buckets to make of the training programs.
    max_candidates_per_compression_step=100,
    arity=2,
    max_compression_steps=1,
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

    for train_task_subset in make_program_log_prior_buckets_iterator(
        initial_ground_truth_experiment_state,
        task_split=TRAIN,
        num_buckets=num_training_buckets,
    ):
        print(
            f"Train task subset: {len(train_task_subset)} tasks: up to {train_task_subset[-1].name}"
        )
        # Make the comparison experiment_state by compressing the frontiers.
        compressed_experiment_state = get_initial_ground_truth_experiment_state(config)
        # Get the compression candidates and rewrite the test set.
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

        for (header, experiment_state) in [
            (INITIAL, initial_ground_truth_experiment_state),
            (COMPRESSED, compressed_experiment_state),
        ]:
            metrics_to_report[header][NUM_TRAIN_TASKS].append(len(train_task_subset))

            # Train the model with respect to the train task subset.
            model = experiment_state.models[AMORTIZED_SYNTHESIS]
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
            generate_rel_plot(
                args,
                config,
                metrics_to_report,
                x_titles=[NUM_TRAIN_TASKS],
                y_titles=[TEST_LOG_LIKELIHOOD],
                plot_title="test_discrimination_original_final_libraries_full",
            )


@register_test
def test_discrimination_candidate_alignments(
    args,
    config,
    num_training_buckets=10,  # How many buckets to make of the training programs.
    max_candidates_per_compression_step=10,
    max_grammar_candidates_to_retain_for_rewriting=4,
    arity=2,
    debug=True,
    report_top_k=5,
):
    """Tests whether the model scoring function can meaningfully rerank a set of proposed DSL candidates and compare this ranking to that produced by the compressor.."""

    initial_ground_truth_experiment_state = get_initial_ground_truth_experiment_state(
        config
    )
    for train_task_subset in make_program_log_prior_buckets_iterator(
        initial_ground_truth_experiment_state,
        task_split=TRAIN,
        num_buckets=num_training_buckets,
    ):
        print(
            f"Train task subset: {len(train_task_subset)} tasks: up to {train_task_subset[-1].name}"
        )
        # Make the comparison experiment_state by compressing the frontiers.
        compressed_experiment_state = get_initial_ground_truth_experiment_state(config)

        # Get the compression candidates.
        grammars_scores_frontiers = compressed_experiment_state.models[
            GRAMMAR
        ]._get_compressed_grammar_candidates_and_rewritten_frontiers(
            experiment_state=compressed_experiment_state,
            task_splits=[TRAIN, TEST],
            task_ids_in_splits={
                TRAIN: [t.name for t in train_task_subset],
                TEST: ALL,  # TODO: why does this not work with fewer -- fix this.
            },
            max_candidates_per_compression_step=max_candidates_per_compression_step,
            max_grammar_candidates_to_retain_for_rewriting=max_grammar_candidates_to_retain_for_rewriting,
            arity=arity,
        )

        print(
            f"Received {len(grammars_scores_frontiers)} candidates; reporting the first K:"
        )
        top_k_grammars_scores_frontiers = grammars_scores_frontiers[:report_top_k]
        for idx, grammar_score_frontier in enumerate(top_k_grammars_scores_frontiers):
            print(f"Reporting {idx}: ")
            print(grammar_score_frontier["grammar"])
            print(f"Score: {grammar_score_frontier['compression_scores']}")


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
