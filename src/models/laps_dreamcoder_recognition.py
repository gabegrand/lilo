"""
laps_dreamcoder_recognition.py | Author : Catherine Wong.

Utility wrapper function around the DreamCoder recognition model. Elevates common functions to be class functions and allows them to be called with an ExperimentState.
"""

# TODO: implement a model loader so that we can load it.


class LAPSDreamCoderRecognition:
    def infer_programs_for_tasks(self):
        pass

    def optimize_model_for_frontiers(
        self,
        experiment_state,
        task_split,
        task_batch_ids,
        task_encoders,
        matrix_rank=None,
        mask=False,
        activation=None,
        contextual=True,
        bias_optimal=True,
        recognition_steps=None,
        timeout=None,
        helmholtz_ratio=None,
        auxiliary_loss=None,
        cuda=None,
        cpus=None,
        max_mem_per_enumeration_thread=1000000,
        require_ground_truth_frontiers=False,
    ):
        """Trains the recognition model with respect to the frontiers. Updates the experiment_state.models[AMORTIZED_SYNTHESIS] with respect to the model."""

        # Skip training if no non-empty frontiers.
        if (
            require_ground_truth_frontiers
            and len(experiment_state.get_non_empty_frontiers_for_split(task_split)) < 1
        ):
            print(
                f"require_ground_truth_frontiers=True and no non-empty frontiers in {task_split}. skipping optimize_model_for_frontiers"
            )
            return
