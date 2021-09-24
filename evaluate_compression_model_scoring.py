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


@register_test
def test_discrimination_original_final_libraries_full(experiment_state):
    """Tests whether the model scoring function can discriminate at all between the initial DSL and the final DSL over all of the training and test programs."""
    experiment_state.initialize_ground_truth_task_frontiers(task_split=TRAIN)
    experiment_state.initialize_ground_truth_task_frontiers(task_split=TEST)

    initial_grammar = experiment_state.models[GRAMMAR]
    for train_task_subset in make_program_log_prior_buckets_iterator(
        experiment_state, task_split=TRAIN, num_buckets=10
    ):
        print(
            f"Train task subset: {len(train_task_subset)} tasks: up to {train_task_subset[-1].name}"
        )
        # Get the compression candidates and rewrite the test set.
        (
            compressed_grammar,
            rewritten_train_test_frontiers,
        ) = initial_grammar._get_compressed_grammmar_and_rewritten_frontiers(
            experiment_state=experiment_state,
            task_splits=[TRAIN, TEST],
            task_ids_in_splits={
                TRAIN: [t.name for t in train_task_subset],
                TEST: ALL,
            },
            max_candidates_per_compression_step=200,
            max_compression_steps=5,
        )

        # First, how well would we have done under the initial grammar with these training tasks?
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

        # As comparison, how well can we do under the compressed grammar with these training tasks?
        # TODO (catwong): retrain with respect to the training tasks.

        # TODO (catwong): report the results.


def load_config_from_file(args):
    config_full_path = os.path.join(args.config_dir, args.config_file)
    with open(config_full_path) as f:
        return json.load(f)


def main(args):
    config = load_config_from_file(args)
    experiment_state = ExperimentState(config)

    test_fns = get_test_fns(args)

    print(f"Now running {len(test_fns)} tests...")
    for idx, test_fn in enumerate(test_fns):
        print(f"Running {idx} / {len(test_fns)}: {test_fn.__name__}")
        test_fn(experiment_state)


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
