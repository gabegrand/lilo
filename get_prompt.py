import json


def get_prompt(filepath, task_n):
    with open(filepath, "r") as f:
        metadata = json.load(f)

    results_by_query = metadata["results_by_query"]
    prompt = results_by_query[task_n]["prompt"]
    final_task_language = prompt["final_task_data"]["task_language"]
    training_task_language = "\n".join(
        [task["task_language"] for task in prompt["body_task_data"]]
    )
    print(
        f"final task: {final_task_language}\ntask examples language: \n{training_task_language}"
    )


filepath = "experiments_iterative/outputs/runs_human_3/domains/re2_human/gpt_solver/seed_111/gpt_solver_96/8/train/gpt_solver_results.json"
get_prompt(filepath, 134)
