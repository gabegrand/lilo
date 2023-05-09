"""
analysis_utilities.py | Author: Gabe Grand.

Class containing utilities for analyzing results from run_iterative_experiment.py.
Usage examples in analysis/analyze_experiment.ipynb.
"""

import glob
import json
import os
from typing import List

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.config_builder import (
    DEFAULT_EXPERIMENT_DIR,
    ExperimentType,
    get_domain_metadata,
)
from src.experiment_iterator import (
    EXPERIMENT_BLOCK_TYPE_MODEL_FN,
    LOOP_BLOCKS,
    METRICS_CHECKPOINT,
    METRICS_LOOP_BLOCK_RUNTIMES,
    MODEL_TYPE,
    RUN_EVERY_N_ITERATIONS,
)
from src.models.model_loaders import AMORTIZED_SYNTHESIS, GRAMMAR, LLM_SOLVER
from src.task_loaders import TASK_SPLIT, TEST, TRAIN


class IterativeExperimentAnalyzer:

    COL_NAMES_CAMERA = {
        "batch_size": "Number of training tasks",
        "description_length": "Test program description length",
        "experiment_type": "Model",
        "n_frontiers": "Number of training programs (including samples)",
        "percent_solved": "Tasks solved (%)",
    }
    DOMAIN_NAMES_CAMERA = {
        "drawings_nuts_bolts": "nuts & bolts",
        "drawings_wheels": "vehicles",
        "drawings_dials": "gadgets",
        "drawings_furniture": "furniture",
        "re2": "REGEX",
        "clevr": "CLEVR",
        "logo": "LOGO",
    }
    EXPERIMENT_TYPES_CAMERA = {
        ExperimentType.ORACLE: "oracle (test)",
        ExperimentType.ORACLE_TRAIN_TEST: "oracle (train + test)",
        ExperimentType.STITCH: "stitch",
        ExperimentType.STITCH_CODEX: "stitch + lilo [programs]",
        ExperimentType.STITCH_CODEX_LANGUAGE: "stitch + lilo [programs, language (train)]",
        ExperimentType.STITCH_CODEX_LANGUAGE_ORIGIN_RANDOM_TEST: "stitch + lilo [programs, language (test)]",
        ExperimentType.STITCH_CODEX_DSL_DESCRIPTION: "stitch + lilo [programs, language (test), dsl description]",
    }
    EXPERIMENT_TYPES_PALETTE = {
        EXPERIMENT_TYPES_CAMERA[ExperimentType.ORACLE]: "#3F553A",
        EXPERIMENT_TYPES_CAMERA[ExperimentType.ORACLE_TRAIN_TEST]: "#8FAD88",
        EXPERIMENT_TYPES_CAMERA[ExperimentType.STITCH]: "#306BAC",
        EXPERIMENT_TYPES_CAMERA[ExperimentType.STITCH_CODEX]: "#B56576",
        EXPERIMENT_TYPES_CAMERA[ExperimentType.STITCH_CODEX_LANGUAGE]: "#E56B6F",
        EXPERIMENT_TYPES_CAMERA[
            ExperimentType.STITCH_CODEX_LANGUAGE_ORIGIN_RANDOM_TEST
        ]: "#EAAC8B",
        EXPERIMENT_TYPES_CAMERA[ExperimentType.STITCH_CODEX_DSL_DESCRIPTION]: "#EAAC8B",
    }

    def __init__(
        self,
        experiment_name,
        experiment_dir: str = DEFAULT_EXPERIMENT_DIR,
        experiment_types: List = None,
        batch_size="all",
        seeds: List[str] = None,
        compute_likelihoods: bool = True,
        allow_incomplete_results: bool = False,
    ):

        self.dir_base = os.path.join("../", experiment_dir, "outputs", experiment_name)
        print(f"Experiment directory: {self.dir_base}")

        self.dir_domains = os.path.join(self.dir_base, "domains")
        self.domains = [
            os.path.split(path)[-1]
            for path in glob.glob(os.path.join(self.dir_domains, "*"))
        ]
        # Reorder the domains by DOMAIN_NAMES_CAMERA
        self.domains = [d for d in self.DOMAIN_NAMES_CAMERA.keys() if d in self.domains]
        print(f"Available domains: {self.domains}")

        self.experiment_types = experiment_types
        self.batch_size = batch_size
        self.seeds = seeds

        self.compute_likelihoods = compute_likelihoods
        self.allow_incomplete_results = allow_incomplete_results

        self.COL_ERROR_BAR = "random_seed"

    def get_available_experiment_types(self, domain):
        experiment_type_paths = sorted(
            glob.glob(os.path.join(self.dir_domains, domain, "*"))
        )
        experiment_types = []
        for path in experiment_type_paths:
            e_type = os.path.split(path)[-1]
            if self.experiment_types is None or e_type in self.experiment_types:
                experiment_types.append(e_type)
        return experiment_types

    def get_available_seeds(self, domain, experiment_type):
        seed_paths = sorted(
            glob.glob(os.path.join(self.dir_domains, domain, experiment_type, "*"))
        )
        seeds = []
        for path in seed_paths:
            with open(os.path.join(path, "config_base.json"), "r") as f:
                config_base = json.load(f)
            seeds.append(config_base["metadata"]["random_seed"])

        # Filter seeds
        if self.seeds is not None:
            seeds = [s for s in seeds if s in self.seeds]

        return seeds

    def get_available_iterations(self, domain, experiment_type, seed):
        path = os.path.join(
            self.dir_domains,
            domain,
            experiment_type,
            f"seed_{seed}",
            f"{experiment_type}_{self.batch_size}",
        )
        return sorted(
            [int(d) for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
        )

    def get_config(self, domain, experiment_type, seed, batch_size: int = None):
        if batch_size is None:
            config_json_path = os.path.join(
                self.dir_domains,
                domain,
                experiment_type,
                f"seed_{seed}",
                "config_base.json",
            )
        else:
            config_json_path = os.path.join(
                self.dir_domains,
                domain,
                experiment_type,
                f"seed_{seed}",
                f"{experiment_type}_{batch_size}",
                "config.json",
            )
        if self.allow_incomplete_results and not os.path.exists(config_json_path):
            return None
        with open(config_json_path, "r") as f:
            config = json.load(f)
        return config

    def get_run_path(self, domain, experiment_type, seed, batch_size):
        return os.path.join(
            self.dir_domains,
            domain,
            experiment_type,
            f"seed_{seed}",
            f"{experiment_type}_{batch_size}",
        )

    def get_runtime_metrics(self):
        df_list = []
        for domain in self.domains:
            for experiment_type in self.get_available_experiment_types(domain):
                for seed in self.get_available_seeds(domain, experiment_type):
                    for iteration in self.get_available_iterations(
                        domain, experiment_type, seed
                    ):
                        run_path = self.get_run_path(
                            domain, experiment_type, seed, self.batch_size
                        )
                        metrics_path = os.path.join(
                            run_path, str(iteration), METRICS_CHECKPOINT
                        )

                        if self.allow_incomplete_results and not os.path.exists(
                            metrics_path
                        ):
                            print(f"Not found: {metrics_path}")
                            continue

                        with open(metrics_path, "r") as f:
                            metrics_json = json.load(f)

                        df = pd.DataFrame(metrics_json[METRICS_LOOP_BLOCK_RUNTIMES])
                        df["domain"] = domain
                        df["experiment_type"] = experiment_type
                        df["seed"] = seed
                        df["iteration"] = iteration
                        df_list.append(df)

        df_runtime = pd.concat(df_list, axis=0).reset_index(drop=True)
        df_runtime["time_start"] = pd.to_datetime(
            df_runtime["time_start"], unit="s", utc=True
        )
        df_runtime["time_end"] = pd.to_datetime(
            df_runtime["time_end"], unit="s", utc=True
        )
        return df_runtime

    def get_default_split(self, experiment_type):
        if experiment_type == ExperimentType.ORACLE:
            split = "test"
        elif experiment_type == ExperimentType.ORACLE_TRAIN_TEST:
            split = "train_test"
        else:
            split = "train"
        return split

    def get_results_for_domain(
        self, domain, experiment_types: List[ExperimentType] = None
    ):
        if experiment_types is None:
            experiment_types = self.get_available_experiment_types(domain)
        df_list = []
        for experiment_type in experiment_types:
            df_experiment_type = self.get_results_for_experiment_type(
                domain, experiment_type
            )
            df_experiment_type["experiment_type"] = experiment_type
            df_list.append(df_experiment_type)
        df = pd.concat(df_list, axis=0).reset_index(drop=True)
        df["batch_size"] = df["batch_size"].astype(int)
        return df

    def get_results_for_experiment_type(self, domain, experiment_type):
        seeds = self.get_available_seeds(domain, experiment_type)

        df_list = []
        for seed in seeds:
            print(f"experiment_type: {experiment_type}, seed: {seed}")
            df = self.get_results_for_run(domain, experiment_type, seed)
            df["random_seed"] = seed
            df_list.append(df)

        df_concat = pd.concat(df_list, axis=0).reset_index(drop=True)

        # Compute the mean over replications
        AGG_COLS = ["n_frontiers", "description_length"]
        if self.compute_likelihoods:
            AGG_COLS += ["log_likelihood"]
        df_out = df_concat.groupby(
            ["batch_size", self.COL_ERROR_BAR], sort=False, as_index=False
        )[AGG_COLS].mean()

        return df_out

    def get_results_for_run(self, domain, experiment_type, random_seed):
        split = self.get_default_split(experiment_type)

        data = []

        config_base = self.get_config(domain, experiment_type, random_seed)
        global_batch_sizes = config_base["metadata"]["global_batch_sizes"]

        for batch_size in global_batch_sizes:
            run_path = self.get_run_path(
                domain, experiment_type, random_seed, batch_size
            )
            config = self.get_config(domain, experiment_type, random_seed, batch_size)
            if self.allow_incomplete_results and config is None:
                print(
                    f"Skipping incomplete results for {domain}, {experiment_type}, {random_seed}, {batch_size}"
                )
                continue

            global_batch_size = config["experiment_iterator"]["task_batcher"]["params"][
                "global_batch_size"
            ]

            test_likelihoods_json_path = os.path.join(
                run_path, "0", "test_likelihoods.json"
            )
            if self.allow_incomplete_results and not os.path.exists(
                test_likelihoods_json_path
            ):
                print(
                    f"Skipping incomplete results for {domain}, {experiment_type}, {random_seed}, {batch_size}"
                )
                continue
            with open(test_likelihoods_json_path, "r") as f:
                likelihoods_data = json.load(f)

            stitch_frontiers_json_path = os.path.join(
                run_path, "0", split, "stitch_compress_input.json"
            )
            if self.allow_incomplete_results and not os.path.exists(
                stitch_frontiers_json_path
            ):
                print(
                    f"Skipping incomplete results for {domain}, {experiment_type}, {random_seed}, {batch_size}"
                )
                continue
            with open(stitch_frontiers_json_path, "r") as f:
                stitch_frontiers_data = json.load(f)

            for task_name, dl_list in likelihoods_data["description_lengths_by_task"][
                "test"
            ].items():
                d = {
                    "batch_size": global_batch_size,
                    "task_name": task_name,
                    "description_length": dl_list[0],
                    "n_frontiers": len(stitch_frontiers_data["frontiers"]),
                }
                if self.compute_likelihoods:
                    d["log_likelihood"] = likelihoods_data["log_likelihoods_by_task"][
                        "test"
                    ][task_name][0]

                data.append(d)

        df = pd.DataFrame(data)
        return df

    def get_codex_programs(self, use_results_by_query: bool = False):
        df_list = []
        for domain in self.domains:
            df_domain = self.get_codex_programs_for_domain(domain, use_results_by_query)
            df_domain["domain"] = domain
            df_list.append(df_domain)
        df = pd.concat(df_list, axis=0).reset_index(drop=True)
        return df

    def get_codex_programs_for_domain(self, domain, use_results_by_query: bool = False):
        df_list = []
        for experiment_type in self.get_available_experiment_types(domain):
            if experiment_type in [
                ExperimentType.STITCH_CODEX,
                ExperimentType.STITCH_CODEX_LANGUAGE,
                ExperimentType.STITCH_CODEX_LANGUAGE_ORIGIN_RANDOM_TEST,
                ExperimentType.STITCH_CODEX_DSL_DESCRIPTION,
            ]:
                if use_results_by_query:
                    df = self.get_codex_programs_by_query_for_experiment_type(
                        domain, experiment_type
                    )
                else:
                    df = self.get_codex_programs_for_experiment_type(
                        domain, experiment_type
                    )
                df["experiment_type"] = experiment_type
                df_list.append(df)
        return pd.concat(df_list, axis=0).reset_index(drop=True)

    def get_codex_programs_for_experiment_type(
        self, domain, experiment_type: ExperimentType = ExperimentType.STITCH_CODEX
    ):
        seeds = self.get_available_seeds(domain, experiment_type)

        df_list = []

        for random_seed in seeds:
            config_base = self.get_config(domain, experiment_type, random_seed)
            global_batch_sizes = config_base["metadata"]["global_batch_sizes"]

            for batch_size in global_batch_sizes:
                path = os.path.join(
                    self.get_run_path(domain, experiment_type, random_seed, batch_size),
                    "0",
                    "gpt_query_results.json",
                )

                if self.allow_incomplete_results and not os.path.exists(path):
                    print(
                        f"Skipping incomplete results for {domain}, {experiment_type}, {random_seed}, {batch_size}"
                    )
                    continue
                with open(path, "r") as f:
                    codex_query_results = json.load(f)

                data = []
                for program_data in codex_query_results["results"]["programs_valid"]:
                    prompt_programs = [
                        d["task_program"]
                        for d in codex_query_results["results_by_query"][
                            program_data["query_id"]
                        ]["prompt"]["body_task_data"]
                    ]
                    prompt_programs.append(
                        codex_query_results["results_by_query"][
                            program_data["query_id"]
                        ]["prompt"]["final_task_data"]["task_program"]
                    )

                    program_data["match_prompt"] = (
                        program_data["program"] in prompt_programs
                    )
                    data.append(program_data)

                df = pd.DataFrame(data)
                df["batch_size"] = batch_size
                df["random_seed"] = random_seed
                df_list.append(df)

        return pd.concat(df_list, axis=0).reset_index(drop=True)

    def get_codex_programs_by_query_for_experiment_type(
        self, domain, experiment_type: ExperimentType = ExperimentType.STITCH_CODEX
    ):
        seeds = self.get_available_seeds(domain, experiment_type)

        df_list = []

        for random_seed in seeds:
            config_base = self.get_config(domain, experiment_type, random_seed)
            global_batch_sizes = config_base["metadata"]["global_batch_sizes"]

            for batch_size in global_batch_sizes:
                path = os.path.join(
                    self.get_run_path(domain, experiment_type, random_seed, batch_size),
                    "0",
                    "gpt_query_results.json",
                )

                if self.allow_incomplete_results and not os.path.exists(path):
                    print(
                        f"Skipping incomplete results for {domain}, {experiment_type}, {random_seed}, {batch_size}"
                    )
                    continue
                with open(path, "r") as f:
                    codex_query_results = json.load(f)

                data = []
                for query_data in codex_query_results["results_by_query"]:
                    prompt_programs = []

                    for body_task_data in query_data["prompt"]["body_task_data"]:
                        body_task_data["query_id"] = query_data["query_id"]
                        body_task_data["program"] = body_task_data["task_program"]
                        body_task_data.pop("task_program")
                        body_task_data["origin"] = "train"
                        body_task_data["final_task"] = False
                        body_task_data["valid"] = True
                        body_task_data["seed"] = random_seed
                        body_task_data["match_prompt"] = False
                        data.append(body_task_data)
                        prompt_programs.append(body_task_data["program"])

                    if "programs" in codex_query_results["params"]["final_task_types"]:
                        final_task_data = query_data["prompt"]["final_task_data"]
                        final_task_data["query_id"] = query_data["query_id"]
                        final_task_data["program"] = final_task_data["task_program"]
                        final_task_data.pop("task_program")
                        final_task_data["origin"] = "train"
                        final_task_data["final_task"] = True
                        final_task_data["valid"] = True
                        final_task_data["seed"] = random_seed
                        final_task_data["match_prompt"] = False
                        data.append(final_task_data)
                        prompt_programs.append(final_task_data["program"])

                    for parse_result_data in query_data["parse_results"]:
                        parse_result_data["origin"] = "codex"
                        parse_result_data["final_task"] = False
                        parse_result_data["seed"] = random_seed
                        if not parse_result_data["valid"]:
                            parse_result_data["program"] = parse_result_data["text"]
                        parse_result_data["match_prompt"] = (
                            parse_result_data["program"] in prompt_programs
                        )
                        data.append(parse_result_data)

                df = pd.DataFrame(data)
                df["program_str_len"] = df.program.str.len()
                df["batch_size"] = batch_size
                df["random_seed"] = random_seed

                train_programs = set(df[df["origin"] == "train"]["program"].tolist())
                df["match_train"] = [
                    (row["origin"] == "codex") and (row["program"] in train_programs)
                    for _, row in df.iterrows()
                ]

                df_list.append(df)

        return pd.concat(df_list).reset_index(drop=True)

    def get_abstractions_for_domain(
        self,
        domain,
        experiment_types: List[ExperimentType] = None,
    ):
        if experiment_types is None:
            experiment_types = self.get_available_experiment_types(domain)

        df_list = []
        for experiment_type in experiment_types:
            split = self.get_default_split(experiment_type)

            for seed in self.get_available_seeds(domain, experiment_type):
                config_base = self.get_config(domain, experiment_type, seed)
                global_batch_sizes = config_base["metadata"]["global_batch_sizes"]
                for iteration in self.get_available_iterations(
                    domain, experiment_type, seed
                ):
                    for batch_size in global_batch_sizes:
                        path = os.path.join(
                            self.dir_domains,
                            domain,
                            experiment_type,
                            f"seed_{seed}",
                            f"{experiment_type}_{batch_size}",
                            str(iteration),
                            split,
                            "stitch_compress_output.json",
                        )

                        with open(path, "r") as f:
                            stitch_output_data = json.load(f)

                        df = pd.DataFrame(stitch_output_data["abstractions"])[
                            [
                                "name",
                                "arity",
                                "utility",
                                "compression_ratio",
                                "cumulative_compression_ratio",
                                "body",
                                "dreamcoder",
                            ]
                        ]
                        df["experiment_type"] = experiment_type
                        df["random_seed"] = seed
                        df["iteration"] = iteration
                        df["batch_size"] = batch_size
                        df_list.append(df)

        return pd.concat(df_list, axis=0).reset_index(drop=True)

    def format_dataframe_camera(self, df):
        df = df.rename(mapper=self.COL_NAMES_CAMERA, axis="columns")

        # Sort experiment types
        experiment_dtype = pd.CategoricalDtype(
            categories=[x.value for x in list(ExperimentType)], ordered=True
        )
        df[self.COL_NAMES_CAMERA["experiment_type"]] = df[
            self.COL_NAMES_CAMERA["experiment_type"]
        ].astype(experiment_dtype)
        df = df.sort_values(
            by=[
                self.COL_NAMES_CAMERA["experiment_type"],
                "domain",
                "seed",
                "split",
                "iteration",
            ],
            ascending=[True, False, True, False, True],
        )

        # Replace experiment type names
        df[self.COL_NAMES_CAMERA["experiment_type"]] = df[
            self.COL_NAMES_CAMERA["experiment_type"]
        ].replace({k.value: v for k, v in self.EXPERIMENT_TYPES_CAMERA.items()})

        # Replace domain names
        if "domain" in df.columns:
            df["domain"] = df["domain"].replace(self.DOMAIN_NAMES_CAMERA)

        # Convert percentages
        if self.COL_NAMES_CAMERA["percent_solved"] in df.columns:
            df[self.COL_NAMES_CAMERA["percent_solved"]] *= 100
        return df

    def plot_description_length(
        self,
        domain: str,
        df: pd.DataFrame,
        plot_type: str = "pointplot",
        logscale: bool = False,
    ):
        df = self.format_dataframe_camera(df)

        plt.figure(figsize=(12, 6))

        if plot_type == "pointplot":
            fig = sns.pointplot(
                data=df,
                x=self.COL_NAMES_CAMERA["batch_size"],
                y=self.COL_NAMES_CAMERA["description_length"],
                hue=self.COL_NAMES_CAMERA["experiment_type"],
                palette=self.EXPERIMENT_TYPES_PALETTE,
            )
        elif plot_type == "lineplot":
            fig = sns.lineplot(
                data=df,
                x=self.COL_NAMES_CAMERA["batch_size"],
                y=self.COL_NAMES_CAMERA["description_length"],
                hue=self.COL_NAMES_CAMERA["experiment_type"],
                palette=self.EXPERIMENT_TYPES_PALETTE,
            )
        else:
            raise ValueError(f"Unknown plot_type: {plot_type}")

        if logscale:
            fig.set(xscale="log")

        plt.title(domain)
        return fig

    def plot_n_frontiers(self, domain: str, df: pd.DataFrame):
        df = self.format_dataframe_camera(df)
        plt.figure(figsize=(12, 6))
        fig = sns.barplot(
            data=df,
            x=self.COL_NAMES_CAMERA["batch_size"],
            y=self.COL_NAMES_CAMERA["n_frontiers"],
            hue=self.COL_NAMES_CAMERA["experiment_type"],
            palette=self.EXPERIMENT_TYPES_PALETTE,
        )
        plt.title(domain)
        return fig


class SynthesisExperimentAnalyzer(IterativeExperimentAnalyzer):
    DOMAIN_NAMES_CAMERA = {
        "re2": "REGEX",
        "clevr": "CLEVR",
        "logo": "LOGO",
    }
    EXPERIMENT_TYPES_CAMERA = {
        ExperimentType.BASE_DSL: "Base DSL",
        ExperimentType.DREAMCODER: "DreamCoder",
        ExperimentType.GPT_SOLVER: "LLM Solver",
        ExperimentType.GPT_SOLVER_STITCH: "LLM Solver (+ Stitch)",
        ExperimentType.GPT_SOLVER_STITCH_NAMER: "LILO",
        ExperimentType.GPT_SOLVER_STITCH_NAMER_HYBRID_DSL: "LILO (+ Hybrid DSL)",
        ExperimentType.GPT_SOLVER_STITCH_NAMER_SEARCH: "LILO (+ Search)",
    }
    EXPERIMENT_TYPES_PALETTE = {
        EXPERIMENT_TYPES_CAMERA[ExperimentType.BASE_DSL]: "#8FAD88",
        EXPERIMENT_TYPES_CAMERA[ExperimentType.DREAMCODER]: "#1E8531",
        EXPERIMENT_TYPES_CAMERA[ExperimentType.GPT_SOLVER]: "#306BAC",
        EXPERIMENT_TYPES_CAMERA[ExperimentType.GPT_SOLVER_STITCH]: "#8999D2",
        EXPERIMENT_TYPES_CAMERA[ExperimentType.GPT_SOLVER_STITCH_NAMER]: "#B56576",
        EXPERIMENT_TYPES_CAMERA[
            ExperimentType.GPT_SOLVER_STITCH_NAMER_HYBRID_DSL
        ]: "#E56B6F",
        EXPERIMENT_TYPES_CAMERA[
            ExperimentType.GPT_SOLVER_STITCH_NAMER_SEARCH
        ]: "#EAAC8B",
    }

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.DOMAIN_NAMES_CAMERA = SynthesisExperimentAnalyzer.DOMAIN_NAMES_CAMERA
        self.EXPERIMENT_TYPES_CAMERA = (
            SynthesisExperimentAnalyzer.EXPERIMENT_TYPES_CAMERA
        )
        self.EXPERIMENT_TYPES_PALETTE = (
            SynthesisExperimentAnalyzer.EXPERIMENT_TYPES_PALETTE
        )

    def get_enumeration_timeout(self, domain, experiment_type, seed):
        config_base = self.get_config(domain, experiment_type, seed)
        return config_base["metadata"]["enumeration_timeout"]

    def get_test_solver_run_every_n(self, domain, experiment_type, seed):
        config_base = self.get_config(domain, experiment_type, seed)
        loop_blocks = config_base["experiment_iterator"][LOOP_BLOCKS]
        test_solver_block = list(
            filter(
                lambda x: x.get(MODEL_TYPE)
                in [GRAMMAR, AMORTIZED_SYNTHESIS, LLM_SOLVER]
                and x.get(EXPERIMENT_BLOCK_TYPE_MODEL_FN) == "infer_programs_for_tasks"
                and x.get(TASK_SPLIT) == TEST,
                loop_blocks,
            )
        )
        assert len(test_solver_block) > 0
        # if len(test_solver_block) > 1:
        #     logging.warning(
        #         f"Found {len(test_solver_block)} test solver blocks; using first one to compute {RUN_EVERY_N_ITERATIONS}"
        #     )
        test_solver_block = test_solver_block[0]
        return test_solver_block.get(RUN_EVERY_N_ITERATIONS, 1)

    def get_synthesis_summary(self):
        # Create n_solved column
        df = self.get_synthesis_results()
        df_summary = (
            df[df.programs.astype(bool)]
            .groupby(["domain", "experiment_type", "seed", "iteration", "split"])
            .task.count()
            .reset_index(name="n_solved")
        )

        # Compute % solved
        percent_solved = []
        for _, row in df_summary.iterrows():
            domain_meta = get_domain_metadata(row["domain"])
            percent_solved.append(
                row["n_solved"] / domain_meta[f"n_tasks_{row['split']}"]
            )
        df_summary["percent_solved"] = percent_solved

        return df_summary

    def get_synthesis_results(self):
        df_list = []
        for domain in self.domains:
            df = self.get_synthesis_results_for_domain(domain)
            df["domain"] = domain
            df_list.append(df)
        return pd.concat(df_list).reset_index(drop=True)

    def get_synthesis_results_for_domain(self, domain):
        experiment_types = self.get_available_experiment_types(domain)
        df_list = []
        for experiment_type in experiment_types:
            df = self.get_synthesis_results_for_experiment_type(domain, experiment_type)
            df["experiment_type"] = experiment_type
            df_list.append(df)
        return pd.concat(df_list).reset_index(drop=True)

    def get_synthesis_results_for_experiment_type(self, domain, experiment_type):
        df_list = []
        for seed in self.get_available_seeds(domain, experiment_type):
            for iteration in self.get_available_iterations(
                domain, experiment_type, seed
            ):
                df = self.get_synthesis_results_for_iteration(
                    domain, experiment_type, seed, iteration
                )
                if df is None:
                    continue
                df["seed"] = seed
                df["iteration"] = iteration
                df_list.append(df)
        return pd.concat(df_list).reset_index(drop=True)

    def get_synthesis_results_for_iteration(
        self, domain, experiment_type, seed, iteration
    ):
        path = os.path.join(
            self.dir_domains,
            domain,
            experiment_type,
            f"seed_{seed}",
            f"{experiment_type}_{self.batch_size}",
            str(iteration),
            "frontiers.json",
        )
        if not os.path.exists(path):
            print(f"WARNING: Missing path {path}")
            if self.allow_incomplete_results:
                return None

        with open(path, "r") as f:
            frontiers_json = json.load(f)

        run_every_n = self.get_test_solver_run_every_n(domain, experiment_type, seed)

        df_list = []
        for split, data in frontiers_json.items():
            # Skip metadata
            if split not in [TRAIN, TEST]:
                continue
            # Skip iterations where the test solver didn't run
            if split == TEST and iteration % run_every_n != 0:
                continue

            df = pd.DataFrame.from_dict(data, orient="index")
            df["task"] = df.index
            df["split"] = split
            df_list.append(df)

        df_out = pd.concat(df_list).reset_index(drop=True)
        df_out["solved"] = df_out["programs"].apply(lambda x: len(x) > 0)
        return df_out

    def get_search_time_results(self, time_interval=1):
        df_list = []
        for domain in self.domains:
            df = self.get_search_time_results_for_domain(
                domain=domain, time_interval=time_interval
            )
            df["domain"] = domain
            df_list.append(df)
        return pd.concat(df_list).reset_index(drop=True)

    def get_search_time_results_for_domain(self, domain, time_interval=1):
        df = self.get_synthesis_results_for_domain(domain)

        enumeration_timeouts = [
            self.get_enumeration_timeout(domain, experiment_type, seed)
            for experiment_type, seed in df[["experiment_type", "seed"]]
            .drop_duplicates()
            .values
        ]
        if len(set(enumeration_timeouts)) != 1:
            raise ValueError(
                f"Enumeration timeouts are inconsistent across conditions: {enumeration_timeouts}"
            )
        enumeration_timeout = enumeration_timeouts[0]
        print(f"Using enumeration_timeout: {enumeration_timeout}")

        df_list = []
        for (experiment_type, seed, iteration, split), df_tmp in df.groupby(
            ["experiment_type", "seed", "iteration", "split"]
        ):

            n_tasks_split = get_domain_metadata(domain)[f"n_tasks_{split}"]

            ts = range(0, enumeration_timeout + time_interval, time_interval)
            n_solved = []

            for t in ts:
                n_solved.append(len(df_tmp[df_tmp.best_search_time <= t]))

            df_tmp_results = pd.DataFrame({"time": list(ts), "n_solved": n_solved})
            df_tmp_results["percent_solved"] = (
                df_tmp_results["n_solved"] / n_tasks_split
            )
            df_tmp_results["experiment_type"] = experiment_type
            df_tmp_results["seed"] = seed
            df_tmp_results["iteration"] = iteration
            df_tmp_results["split"] = split

            df_list.append(df_tmp_results)

        return pd.concat(df_list).reset_index(drop=True)
