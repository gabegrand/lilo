"""
make_tasks.py | Author: Catherine Wong
Loads tasks and language for the drawings domain.
"""
from collections import defaultdict
import csv
from email.policy import default
import itertools
import pathlib
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
DEFAULT_LANGUAGE_DIR = os.path.join(ROOT_DIR, f"data/{DOMAIN_NAME}/language")
RAW_LANGUAGE_CSV = "lax_corpus_1k_trial.csv"
DEFAULT_TASK_CSV_SUFFIX = "_programs_all.csv"

# Rows to read in the CSVs
CSV_PROGRAM = "dreamcoder_program_dsl_0"
CSV_S3_STIMULI = "s3_stimuli"
CSV_TASK_GENERATOR = "task_generator"

WHAT_TOKEN, WHERE_TOKEN = "<WHAT>", "<WHERE>"


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

    def load_tasks(self):
        tasks = {TRAIN: [], TEST: []}
        for domain in TASK_DOMAINS:
            domain_tasks = self.load_tasks_for_domain(domain)
            for split in domain_tasks:
                tasks[split] += domain_tasks[split]
        return tasks


@TaskLanguageLoaderRegistry.register
class DrawingsHumanLanguageLoader(TaskDataLoader):
    name = DOMAIN_NAME + "_human"

    def get_language_for_row(self, row, domain):
        domain = domain.replace("_", "-")
        task_url = row["stimURL"]
        if domain not in task_url:
            return None, None, None
        whats, wheres = (
            [s.strip() for s in eval(row["whats"])],
            [s.strip() for s in eval(row["wheres"])],
        )
        zipped = list(zip(whats, wheres))
        language = "\n".join(
            [f"{WHAT_TOKEN} {what} {WHERE_TOKEN} {where}" for (what, where) in zipped]
        )

        all_strings = list(itertools.chain(*zipped))
        all_tokens = list(itertools.chain(*[s.split() for s in all_strings]))
        tokens = set([WHAT_TOKEN, WHERE_TOKEN] + all_tokens)
        return task_url, language, tokens

    def build_language_vocabulary_jsons(self, domain):
        # Load the tasks.
        tasks_loader = DrawingsLoader()
        tasks_for_domain = tasks_loader.load_tasks_for_domain(domain)
        task_url_to_split_name = {
            t.image_url: (split, t.name)
            for split in tasks_for_domain
            for t in tasks_for_domain[split]
        }
        vocabulary = defaultdict(set)
        language_dataset = defaultdict(lambda: defaultdict(list))

        language_csv_path = os.path.join(DEFAULT_LANGUAGE_DIR, RAW_LANGUAGE_CSV)
        with open(language_csv_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                task_url, language, tokens = self.get_language_for_row(row, domain)
                if task_url is not None:
                    split, task_name = task_url_to_split_name[task_url]
                    language_dataset[split][task_name].append(language)
                    vocabulary[split].update(tokens)
        # Assert we have enough.
        for split in language_dataset:
            assert len(language_dataset[split]) == len(tasks_for_domain[split])

        # Write them out to the directory.
        domain_dataset_path = os.path.join(DEFAULT_LANGUAGE_DIR + "_human", domain)
        for split in language_dataset:
            dataset_path = os.path.join(domain_dataset_path, split)
            pathlib.Path(dataset_path).mkdir(parents=True, exist_ok=True)
            language_file = os.path.join(dataset_path, f"{LANGUAGE}.json")
            vocab_file = os.path.join(dataset_path, f"{VOCAB}.json")
            with open(language_file, "w") as f:
                json.dump(language_dataset[split], f)

            with open(vocab_file, "w") as f:
                json.dump(list(vocabulary[split]), f)
        return domain_dataset_path

    def load_task_language_domain(self, domain):
        # If never loaded before, prepare the language.
        dataset_path = os.path.join(DEFAULT_LANGUAGE_DIR + "_human", domain)
        if not os.path.exists(dataset_path):
            self.build_language_vocabulary_jsons(domain)
        # Use the default loader.
        return self.load_task_language_for_directory(dataset_path, splits=[TRAIN, TEST])

    def load_task_language(self):
        task_dataset = DrawingsHumanLanguageLoader.name
        language = defaultdict(lambda: defaultdict(list))
        vocab = defaultdict(set)
        for domain in TASK_DOMAINS:
            domain_language, domain_vocab = self.load_task_language_domain(domain)
            for split in domain_language:
                language[split].update(domain_language[split])
                vocab[split].update(domain_vocab[split])
        for split in vocab:
            vocab[split] = list(vocab[split])
        return language, vocab


# Individual task and language loaders for all specific domains.
@TaskLoaderRegistry.register
class DrawingsNutsBoltsLoader(DrawingsLoader):
    domain_name = "nuts_bolts"
    name = DOMAIN_NAME + "_" + domain_name

    def load_tasks(self):
        return self.load_tasks_for_domain(self.domain_name)


@TaskLanguageLoaderRegistry.register
class DrawingsHumanNutsBoltsLanguageLoader(DrawingsHumanLanguageLoader):
    domain_name = "nuts_bolts"
    name = DrawingsHumanLanguageLoader.name + "_" + domain_name

    def load_task_language(self):
        return self.load_task_language_domain(self.domain_name)


@TaskLoaderRegistry.register
class DrawingsDialsLoader(DrawingsLoader):
    domain_name = "dials"
    name = DOMAIN_NAME + "_" + domain_name

    def load_tasks(self):
        return self.load_tasks_for_domain(self.domain_name)


@TaskLanguageLoaderRegistry.register
class DrawingsHumanDialsLanguageLoader(DrawingsHumanLanguageLoader):
    domain_name = "dials"
    name = DrawingsHumanLanguageLoader.name + "_" + domain_name

    def load_task_language(self):
        return self.load_task_language_domain(self.domain_name)


@TaskLoaderRegistry.register
class DrawingsFurnitureLoader(DrawingsLoader):
    domain_name = "furniture"
    name = DOMAIN_NAME + "_" + domain_name

    def load_tasks(self):
        return self.load_tasks_for_domain(self.domain_name)


@TaskLanguageLoaderRegistry.register
class DrawingsHumanFurnitureLanguageLoader(DrawingsHumanLanguageLoader):
    domain_name = "furniture"
    name = DrawingsHumanLanguageLoader.name + "_" + domain_name

    def load_task_language(self):
        return self.load_task_language_domain(self.domain_name)


@TaskLoaderRegistry.register
class DrawingsWheelsLoader(DrawingsLoader):
    domain_name = "wheels"
    name = DOMAIN_NAME + "_" + domain_name

    def load_tasks(self):
        return self.load_tasks_for_domain(self.domain_name)


@TaskLanguageLoaderRegistry.register
class DrawingsHumanWheelsLanguageLoader(DrawingsHumanLanguageLoader):
    domain_name = "wheels"
    name = DrawingsHumanLanguageLoader.name + "_" + domain_name

    def load_task_language(self):
        return self.load_task_language_domain(self.domain_name)
