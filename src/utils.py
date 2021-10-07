"""
utils.py | Author : Catherine Wong
General purpose utilities.
"""
import subprocess
import datetime
from pathlib import Path

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from collections import defaultdict


def escaped_timestamp():
    """[ret]: escaped string timestamp."""
    timestamp = datetime.datetime.now().isoformat()
    # Escape the timestamp.
    timestamp = timestamp.replace(":", "-")
    timestamp = timestamp.replace(".", "-")
    return timestamp


def mkdir_if_necessary(path):
    """Creates a directory if necessary.
    [ret]: string to directory path."""
    Path(path).mkdir(parents=True, exist_ok=True)
    return path


## Utilities for plotting.


def generate_rel_plot(
    args, metrics_to_report, x_titles, y_titles, plot_title, y_lim=1.0
):
    """
    Generates seaborn rel_plot from a metrics_to_report dict. See evaluate_compression_model_scoring.py for an example.
    """
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


## Utilities for running experiments.
DEFAULT_SINGULARITY_CONTAINER = "../containers/laps-dev-container.img"


def cleaned_args_dict(args):
    # Cleans up an args dict back into a string form.
    cleaned_args_dict = {}
    for (arg_name, arg_value) in args.items():
        if arg_value in [True, False]:
            print(
                f"!Warning: you used a boolean flag for {arg_name}. Using a heuristic to determine if we should keep this."
            )
            if "no" in arg_name:  # This is probably a store true.
                arg_name = None if not arg_value else arg_name
                arg_value = ""
            else:  # This is probably a store False.
                arg_name = None if arg_value else arg_name
                arg_value = ""
        elif type(arg_value) == list:
            arg_value = " ".join(arg_value)

        if arg_name is not None:
            arg_name = "--" + arg_name
            cleaned_args_dict[arg_name] = arg_value
    return cleaned_args_dict


def generate_singularity_command(
    source_python_file, args, singularity_container_location
):
    args_string = " ".join(
        [
            f"{arg_name} {arg_value}"
            for (arg_name, arg_value) in cleaned_args_dict(vars(args)).items()
        ]
    )
    wrapped_command = f"python {source_python_file}  {args_string}"
    singularity_command = f"singularity exec -B /om2 --nv {singularity_container_location} {wrapped_command}"
    return wrapped_command, singularity_command


DEFAULT_OM_MEMORY = "30G"
DEFAULT_CPUS_PER_TASK = "12"
DEFAULT_TIME_HRS = "24"


def generate_om_singularity_command(
    source_python_file,
    output_dir,
    job_name,
    args,
    singularity_container_location=DEFAULT_SINGULARITY_CONTAINER,
    memory=DEFAULT_OM_MEMORY,
    cpus=DEFAULT_CPUS_PER_TASK,
    time=DEFAULT_TIME_HRS,
):
    """
    Generates a SLURM command specifically for running on the OpenMind computing cluster from the Tenenbaum lab. This assumes that the Python file will be run with a singularity container.
    """

    print(f"To run on OM, run the following command: \n")
    wrapped_command, singularity_command = generate_singularity_command(
        source_python_file, args, singularity_container_location
    )
    logfile = f"{output_dir}/{job_name}"
    om_command = f"sbatch --job-name={job_name} --output={logfile} --ntasks=1 --mem={memory} --cpus-per-task {cpus} --time={time}:00:00 --qos=tenenbaum --partition=tenenbaum --wrap='{singularity_command}'"

    print(f"\n{om_command}\n")
    return logfile, wrapped_command, om_command


CLOUD_OM = "om"
CLOUD_COMMAND_GENERATORS = {CLOUD_OM: generate_om_singularity_command}


def generate_cloud_command(source_python_file, job_name, args, output_dir):
    print(
        f"Generating cloud command for {source_python_file} to run on: {args.util_generate_cloud_command} \n"
    )

    # Generate the cloud command.
    cloud_command_generator = CLOUD_COMMAND_GENERATORS[args.util_generate_cloud_command]
    # Remove the cloud command argument.
    del args.util_generate_cloud_command

    logfile, wrapped_command, cloud_command = cloud_command_generator(
        source_python_file, output_dir, job_name, args
    )

    # Print the argument hyperparameters.
    print_hyperparameter_arguments(args)

    # Generate the experiment replication log information.
    generate_experiment_replication_log_information(logfile, wrapped_command)


def print_hyperparameter_arguments(args):
    HYPERPARAMETER_PREFIX = "hp"
    print("Running with hyperparameters: ")
    for arg_name in sorted(list(vars(args).keys())):
        if arg_name.startswith(HYPERPARAMETER_PREFIX):
            print(f"\t{arg_name}: {vars(args)[arg_name]}")


def get_git_commit_sha():
    return (
        subprocess.check_output(["git", "rev-parse", "--short", "HEAD"])
        .decode("ascii")
        .strip()
    )


def generate_experiment_replication_log_information(logfile, wrapped_command):
    """
    Generates log-information for replicating this exact experiment.
    """
    print("\nAdd this information to your logfile for replication: ")
    log_information = {}
    # Git SHA commit.
    log_information["git_commit_sha"] = get_git_commit_sha()

    # Time and date.
    log_information["time_and_date"] = escaped_timestamp()

    # What command was actually run.
    log_information["wrapped_command"] = wrapped_command

    # Logfile.
    log_information["logfile"] = logfile

    # Print the log information.
    for arg_name in sorted(list(log_information.keys())):
        print(f"\t{arg_name}: {log_information[arg_name]}")
