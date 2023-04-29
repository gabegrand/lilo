"""
task_loaders.py | Author : Gabe Grand
Utilities for autogenerating configs for experiments based on templates.

"""

import json
import os
import subprocess
from enum import Enum

# import data.drawings.make_tasks as drawing_tasks # zyzzyva@ Temporarily disable this domain., which is causing
from src.experiment_iterator import (
    AWS_S3_SYNC_BASE_PATH,
    CURR_ITERATION,
    EXPERIMENT_BLOCK_TYPE,
    EXPERIMENT_BLOCK_TYPE_CHECKPOINT,
)
from src.models.laps_dreamcoder_recognition import LAPSDreamCoderRecognition
from src.models.laps_grammar import LAPSGrammar
from src.models.model_loaders import (
    AMORTIZED_SYNTHESIS,
    GRAMMAR,
    INITIALIZE_GROUND_TRUTH,
    LIBRARY_LEARNER,
    LIBRARY_NAMER,
    LLM_SOLVER,
    PROGRAM_REWRITER,
    SAMPLE_GENERATOR,
)
from src.task_loaders import ALL, RandomShuffleOrderedTaskBatcher

# @zyzzyva (April 19): Temporarily disable the drawings domain, which is causing Primitive import conflicts.


DEFAULT_EXPERIMENT_DIR = "experiments_iterative"
DEFAULT_TEMPLATE_DIR = os.path.join(DEFAULT_EXPERIMENT_DIR, "templates")


DEFAULT_STITCH_PARAMS = {
    "max_arity": 3,
    "iterations": 10,
    "candidates_per_iteration": 1,
}

DEFAULT_GPT_PARAMS = {
    "debug": False,
    "use_cached": False,
    "n_samples": 50,
    "n_samples_per_query": 5,
    "temperature": 0.40,
    "max_tokens_completion_beta": 2.0,
    "function_name_classes": ["human_readable", "default"],
    "final_task_origin": "default",
    "body_task_types": ["programs"],
    "final_task_types": ["programs"],
    "prepend_dsl_description": False,
}

DEFAULT_GPT_SOLVER_PARAMS = {
    "temperature": 0.90,
    "max_tokens_completion_beta": 4.0,
    "function_name_classes": ["human_readable", "default"],
}


class ExperimentType(str, Enum):
    ORACLE = "oracle"
    ORACLE_TRAIN_TEST = "oracle_train_test"
    STITCH = "stitch"
    STITCH_CODEX = "stitch_codex"
    STITCH_CODEX_LANGUAGE = "stitch_codex_language"
    STITCH_CODEX_LANGUAGE_ORIGIN_RANDOM_TEST = (
        "stitch_codex_language_origin_random_test"
    )
    STITCH_CODEX_DSL_DESCRIPTION = "stitch_codex_dsl_description"


def get_domain_metadata(domain: str):
    METADATA = {
        "logo": {
            "tasks_loader": "compositional_graphics_200",
            "task_language_loader": "compositional_graphics_200_synthetic",
            "ocaml_special_handler": "LOGO",
            "dsl_description_prefix": "This is a domain-specific language for Logo turtle graphics.",
            "global_batch_sizes": [5, 10, 15, 25, 50, 100, 200],
            "n_tasks_train": 200,
            "n_tasks_test": 111,
        },
        "clevr": {
            "tasks_loader": "clevr",
            "task_language_loader": "clevr_synthetic",
            "ocaml_special_handler": "clevr",
            "dsl_description_prefix": "This is a domain-specific language for CLEVR: A Diagnostic Dataset for Compositional Language and Elementary Visual Reasoning.",
            "global_batch_sizes": [5, 10, 15, 25, 50, 100, 191],
            "n_tasks_train": 191,
            "n_tasks_test": 103,
        },
        "re2": {
            "tasks_loader": "re2",
            "task_language_loader": "re2_synthetic",
            "ocaml_special_handler": "re2",
            "dsl_description_prefix": "This is a domain-specific language for regular expressions that specify string transformations.",
            "global_batch_sizes": [5, 10, 15, 25, 50, 100, 200, 300, 400, 491],
            "n_tasks_train": 491,
            "n_tasks_test": 500,
        },
    }

    # # Metadata for each drawing task domain
    # METADATA["drawings"] = {
    #     "tasks_loader": "drawings",
    #     "task_language_loader": f"drawings_human",
    #     "ocaml_special_handler": "drawings",
    #     "global_batch_sizes": [
    #         5,
    #         10,
    #         15,
    #         25,
    #         50,
    #         100,
    #         200,
    #         300,
    #         400,
    #         500,
    #         600,
    #         700,
    #         800,
    #     ],
    # }
    # for drawing_domain in drawing_tasks.TASK_DOMAINS:
    #     drawing_domain_name = "drawings_" + drawing_domain
    #     drawing_domain_metadata = {
    #         "tasks_loader": drawing_domain_name,
    #         "task_language_loader": f"drawings_human_{drawing_domain}",
    #         "ocaml_special_handler": "drawings",
    #         "dsl_description_prefix": "",
    #         "global_batch_sizes": [5, 10, 15, 25, 50, 100, 150, 200],
    #     }
    #     METADATA[drawing_domain_name] = drawing_domain_metadata

    return METADATA[domain]


def build_config(
    experiment_name: str,
    experiment_type: str,
    domain: str,
    output_directory: str = DEFAULT_EXPERIMENT_DIR,
    random_seed: int = 0,
    iterations: int = 1,
    init_iteration: int = 0,
    task_batcher: str = RandomShuffleOrderedTaskBatcher.name,
    global_batch_size: int = ALL,
    enumeration_timeout: int = None,
    recognition_train_steps: int = None,
    encoder: str = None,
    stitch_params: dict = DEFAULT_STITCH_PARAMS,
    gpt_params: dict = DEFAULT_GPT_PARAMS,
    compute_likelihoods: bool = True,
    compute_description_lengths: bool = True,
    increment_task_batcher: bool = True,
    init_frontiers_from_checkpoint: bool = False,
    init_frontiers_every_iteration: bool = False,
    resume_checkpoint_directory: bool = False,
    s3_sync: bool = True,
):
    config = {}
    config.update(
        build_config_body(
            experiment_type=experiment_type,
            domain=domain,
            iterations=iterations,
            task_batcher=task_batcher,
            global_batch_size=global_batch_size,
            enumeration_timeout=enumeration_timeout,
            recognition_train_steps=recognition_train_steps,
            encoder=encoder,
            stitch_params=stitch_params,
            gpt_params=gpt_params,
            compute_likelihoods=compute_likelihoods,
            compute_description_lengths=compute_description_lengths,
            increment_task_batcher=increment_task_batcher,
            s3_sync=s3_sync,
        )
    )
    config.update(
        build_config_metadata(
            experiment_name=experiment_name,
            domain=domain,
            experiment_type=experiment_type,
            global_batch_size=global_batch_size,
            enumeration_timeout=enumeration_timeout,
            recognition_train_steps=recognition_train_steps,
            encoder=encoder,
            output_directory=output_directory,
            init_iteration=init_iteration,
            init_frontiers_from_checkpoint=init_frontiers_from_checkpoint,
            init_frontiers_every_iteration=init_frontiers_every_iteration,
            resume_checkpoint_directory=resume_checkpoint_directory,
            random_seed=random_seed,
        )
    )
    return config


def build_config_metadata(
    experiment_name: str,
    domain: str,
    experiment_type: str,
    global_batch_size: int = ALL,
    enumeration_timeout: int = None,
    recognition_train_steps: int = None,
    encoder: str = None,
    output_directory: str = DEFAULT_EXPERIMENT_DIR,
    init_iteration: int = 0,
    init_frontiers_from_checkpoint: bool = False,
    init_frontiers_every_iteration: bool = False,
    resume_checkpoint_directory: bool = False,
    random_seed: int = 0,
):
    domain_meta = get_domain_metadata(domain)

    export_directory = os.path.join(
        output_directory,
        "outputs",
        experiment_name,
        "domains",
        domain,
        experiment_type,
        f"seed_{random_seed}",
    )
    log_directory = os.path.join(
        output_directory,
        "logs",
        experiment_name,
        "domains",
        domain,
        experiment_type,
        f"seed_{random_seed}",
    )

    return {
        "metadata": {
            "experiment_name": experiment_name,
            "experiment_id": f"{experiment_type}_{global_batch_size}",
            "human_readable": "Autogenerated iterative experiment.",
            "export_directory": export_directory,
            "log_directory": log_directory,
            "tasks_loader": domain_meta["tasks_loader"],
            "task_language_loader": domain_meta["task_language_loader"],
            "dsl_description_prefix": domain_meta["dsl_description_prefix"],
            "export_with_timestamp": False,
            "resume_checkpoint_directory": resume_checkpoint_directory,
            "init_frontiers_from_checkpoint": init_frontiers_from_checkpoint,
            "init_frontiers_every_iteration": init_frontiers_every_iteration,
            "ocaml_special_handler": domain_meta["ocaml_special_handler"],
            "global_batch_size": global_batch_size,
            "enumeration_timeout": enumeration_timeout,
            "recognition_train_steps": recognition_train_steps,
            "encoder": encoder,
            "random_seed": random_seed,
            "curr_iteration": init_iteration,
        }
    }


def build_config_body(
    experiment_type: str,
    domain: str,
    iterations: int = 1,
    task_batcher: str = RandomShuffleOrderedTaskBatcher.name,
    global_batch_size: int = ALL,
    enumeration_timeout: int = None,
    recognition_train_steps: int = None,
    encoder: str = None,
    stitch_params: dict = DEFAULT_STITCH_PARAMS,
    gpt_params: dict = DEFAULT_GPT_PARAMS,
    compute_likelihoods: bool = True,
    compute_description_lengths: bool = True,
    increment_task_batcher: bool = True,
    s3_sync: bool = True,
):
    template_path = os.path.join(
        DEFAULT_TEMPLATE_DIR, f"template_{experiment_type}.json"
    )
    with open(template_path, "r") as f:
        config = json.load(f)

    domain_meta = get_domain_metadata(domain)

    model_initializers = config["model_initializers"]
    model_initializers[0]["model_loader"] = domain_meta["ocaml_special_handler"]
    config["model_initializers"] = model_initializers

    # Update recognition model params to match domain if there is a recognition model and
    # it is set on the command-line.
    recognition_encoder_initializer = next(
        (
            initializer
            for initializer in model_initializers
            if initializer["model_type"] == "examples_encoder"
        ),
        None,
    )

    if encoder:
        if not recognition_encoder_initializer:
            raise ValueError(
                "Encoder is provided by command-line arguments but there is no encoder being initialized in the template."
            )
        recognition_encoder_initializer["model_loader"] = encoder
    elif (
        recognition_encoder_initializer
        and recognition_encoder_initializer["model_loader"] is None
    ):
        raise ValueError(
            "Encoder is not provided by command-line arguments but there is an encoder being initialized in the template."
        )

    config["experiment_iterator"]["max_iterations"] = iterations
    config["experiment_iterator"]["task_batcher"]["model_type"] = task_batcher
    config["experiment_iterator"]["task_batcher"]["params"][
        "global_batch_size"
    ] = global_batch_size

    config["experiment_iterator"]["task_batcher"]["params"][
        "increment_at_global_iteration"
    ] = increment_task_batcher

    # params updates use the following precedence order (highest to lowest):
    # 1. params from CLI (e.g., stitch_params)
    # 2. params from template (e.g., block["params"])
    # 3. params from config_builder globals (e.g., DEFAULT_STITCH_PARAMS)

    loop_blocks = []
    for block in config["experiment_iterator"]["loop_blocks"]:
        if (
            block.get("model_type") == LAPSGrammar.GRAMMAR
            and block.get("model_fn") == LAPSGrammar.infer_programs_for_tasks.__name__
            and enumeration_timeout is not None
        ) or (
            block.get("model_type") == AMORTIZED_SYNTHESIS
            and block.get("model_fn")
            == LAPSDreamCoderRecognition.infer_programs_for_tasks.__name__
            and enumeration_timeout is not None
        ):
            block["params"].update(
                {
                    "enumeration_timeout": enumeration_timeout,
                }
            )
        if block.get("model_type") == SAMPLE_GENERATOR:
            _gpt_params = DEFAULT_GPT_PARAMS
            _gpt_params.update(block["params"])
            _gpt_params.update(gpt_params)
            block["params"] = _gpt_params
        if block.get("model_type") == LLM_SOLVER:
            _gpt_params = DEFAULT_GPT_SOLVER_PARAMS
            _gpt_params.update(block["params"])
            _gpt_params.update(gpt_params)
            block["params"] = _gpt_params
        if block.get("model_type") == LIBRARY_NAMER:
            _gpt_params = block["params"]
            _gpt_params.update(gpt_params)
        if block.get("model_type") == LIBRARY_LEARNER:
            _stitch_params = DEFAULT_STITCH_PARAMS
            _stitch_params.update(block["params"])
            _stitch_params.update(stitch_params)
            block["params"] = _stitch_params
        if (
            block.get("model_type")
            in [
                LAPSGrammar.GRAMMAR,
                SAMPLE_GENERATOR,
                PROGRAM_REWRITER,
            ]
            or block.get("state_fn") == INITIALIZE_GROUND_TRUTH
        ):
            block["params"].update(
                {
                    "compute_likelihoods": compute_likelihoods,
                }
            )
        if (
            block.get("model_type") == AMORTIZED_SYNTHESIS
            and block.get("model_fn")
            == LAPSDreamCoderRecognition.optimize_model_for_frontiers.__name__
            and recognition_train_steps is not None
        ):
            block["params"].update(
                {
                    "recognition_train_steps": recognition_train_steps,
                }
            )
        if (
            block.get(EXPERIMENT_BLOCK_TYPE) == EXPERIMENT_BLOCK_TYPE_CHECKPOINT
        ) and block.get(AWS_S3_SYNC_BASE_PATH):
            if s3_sync:
                # Verify that AWS CLI is configured on the machine
                subprocess.run(
                    "aws sts get-caller-identity",
                    shell=True,
                    capture_output=True,
                    check=True,
                )
                # Verify that the bucket exists
                subprocess.run(
                    f"aws s3 ls {block[AWS_S3_SYNC_BASE_PATH]}",
                    shell=True,
                    capture_output=True,
                    check=True,
                )
            else:
                # Disable S3 upload
                block[AWS_S3_SYNC_BASE_PATH] = None

        loop_blocks.append(block)
    config["experiment_iterator"]["loop_blocks"] = loop_blocks

    return config
