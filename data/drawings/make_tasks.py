"""
make_tasks.py | Author: Catherine Wong
Loads tasks and language for the drawings domain.
"""
import csv
from dreamcoder.task import Task
from src.task_loaders import *

from data.drawings.grammar import *
from data.drawings.drawings_primitives import *
from dreamcoder.program import EtaLongVisitor, Program

DOMAIN_NAME = "drawings"

TASK_DOMAINS = [
    "nuts_bolts",
    "furniture",
    "dials",
    "wheels",
]
ROOT_DIR = os.getcwd()
DEFAULT_TASK_DIR = os.path.join(ROOT_DIR, f"data/{DOMAIN_NAME}/tasks")
DEFAULT_TASK_CSV_SUFFIX = "_programs_all.csv"

# Rows to read in the CSVs
CSV_PROGRAM = "dreamcoder_program_dsl_0"
CSV_S3_STIMULI = "s3_stimuli"
CSV_TASK_GENERATOR = "task_generator"


@TaskLoaderRegistry.register
class DrawingsLoader(TaskDataLoader):
    name = DOMAIN_NAME

    def make_drawing_task(self, csv_row):
        program = csv_row[CSV_PROGRAM]
        s3_stimuli = csv_row[CSV_S3_STIMULI]
        split = TRAIN if TRAIN in csv_row[CSV_TASK_GENERATOR] else TEST

        name = s3_stimuli.split("/")[-1]
        program = Program.parse(program)
        assert program.infer() == tstroke
        # Add the rendered image.
        rendered_image = DrawingGrammar.render_program(program)
        drawing_task = Task(
            name=name, request=tstroke, examples=[(([0]), rendered_image)]
        )

        drawing_task.supervisedSolution = program
        drawing_task.groundTruthProgram = program
        # Source image for reference.
        drawing_task.image_url = s3_stimuli
        drawing_task.highresolution = rendered_image

        return split, drawing_task

    def load_tasks_for_domain(self, domain):
        domain_csv_path = os.path.join(
            DEFAULT_TASK_DIR, domain + DEFAULT_TASK_CSV_SUFFIX
        )
        tasks = {TRAIN: [], TEST: []}
        with open(domain_csv_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                split, task = self.make_drawing_task(row)
                tasks[split].append(task)
        return tasks

    def load_tasks(self, domains=TASK_DOMAINS):
        tasks = {TRAIN: [], TEST: []}
        for domain in domains:
            domain_tasks = self.load_tasks_for_domain(domain)
            for split in domain_tasks:
                tasks[split] += domain_tasks[split]
        return tasks
