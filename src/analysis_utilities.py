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

from src.config_builder import DEFAULT_EXPERIMENT_DIR, ExperimentType


class IterativeExperimentAnalyzer:

    COL_NAMES_CAMERA = {
        "batch_size": "Number of training tasks",
        "description_length": "Test program description length",
        "experiment_type": "Model",
        "n_frontiers": "Number of training programs (including samples)",
    }
    DOMAIN_NAMES_CAMERA = {
        "drawings_nuts_bolts": "nuts & bolts",
        "drawings_wheels": "vehicles",
        "drawings_dials": "gadgets",
        "drawings_furniture": "furniture",
        "re2": "REGEX",
        "clevr": "CLEVR",
    }
    EXPERIMENT_TYPES_CAMERA = {
        ExperimentType.ORACLE: "oracle (test)",
        ExperimentType.ORACLE_TRAIN_TEST: "oracle (train + test)",
        ExperimentType.STITCH: "stitch",
        ExperimentType.STITCH_CODEX: "stitch + lilo [programs]",
        ExperimentType.STITCH_CODEX_LANGUAGE: "stitch + lilo [programs, language (train)]",
        ExperimentType.STITCH_CODEX_LANGUAGE_ORIGIN_RANDOM_TEST: "stitch + lilo [programs, language (test)]",
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
    }

    def __init__(
        self,
        experiment_name,
        experiment_dir: str = DEFAULT_EXPERIMENT_DIR,
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

        self.compute_likelihoods = compute_likelihoods
        self.allow_incomplete_results = allow_incomplete_results

        self.COL_ERROR_BAR = "random_seed"

    def get_available_experiment_types(self, domain):
        experiment_type_paths = sorted(
            glob.glob(os.path.join(self.dir_domains, domain, "*"))
        )
        experiment_types = []
        for path in experiment_type_paths:
            experiment_types.append(os.path.split(path)[-1])
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
        return seeds

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
                run_path, "0", split, "stitch_frontiers.json"
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
                    "codex_query_results.json",
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
                    "codex_query_results.json",
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

    def get_library_inventions(
        self, domain, experiment_types: List[ExperimentType] = None
    ):
        if experiment_types is None:
            experiment_types = self.get_available_experiment_types(domain)

        df_list = []
        for experiment_type in experiment_types:
            split = self.get_default_split(experiment_type)

            for seed in self.get_available_seeds(domain, experiment_type):
                config_base = self.get_config(domain, experiment_type, seed)
                global_batch_sizes = config_base["metadata"]["global_batch_sizes"]
                for batch_size in global_batch_sizes:
                    path = os.path.join(
                        self.dir_domains,
                        domain,
                        experiment_type,
                        f"seed_{seed}",
                        f"{experiment_type}_{batch_size}",
                        "0",
                        split,
                        "stitch_output.json",
                    )

                    with open(path, "r") as f:
                        stitch_output_data = json.load(f)

                    df = pd.DataFrame(stitch_output_data["invs"])[
                        ["name", "arity", "utility", "multiplier", "body", "dreamcoder"]
                    ]
                    df["experiment_type"] = experiment_type
                    df["random_seed"] = seed
                    df["batch_size"] = batch_size
                    df_list.append(df)

        return pd.concat(df_list, axis=0).reset_index(drop=True)

    def format_dataframe_camera(self, df):
        df = df.rename(mapper=self.COL_NAMES_CAMERA, axis="columns")
        df[self.COL_NAMES_CAMERA["experiment_type"]] = df[
            self.COL_NAMES_CAMERA["experiment_type"]
        ].replace({k.value: v for k, v in self.EXPERIMENT_TYPES_CAMERA.items()})
        if "domain" in df.columns:
            df["domain"] = df["domain"].replace(self.DOMAIN_NAMES_CAMERA)
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
