import json
import os
import queue
from random import choice, randrange


class Evaluator:
    score = int  # maps client names/IDs to scores
    num_guesses = int
    curr_stroke = -1
    answer = str
    categories = list[str]

    def __init__(self, directory: str):
        self.stroke_q = None
        self.directory = directory
        self.categories = self.get_categories()
        if not self.categories:
            raise ValueError("No non-empty categories found!")

    def load_new_case(self):
        # Ensure there are categories to choose from
        if not self.categories:
            raise ValueError("No categories available!")

        # Choose random category
        category = choice(self.categories)

        # Within category, choose random case
        case = self.pick_case_from_file(category)
        case = json.loads(case)

        # Load strokes from case into the queue
        self.stroke_q = queue.Queue()
        for c in case["strokes"]:
            self.stroke_q.put(list(c))

        self.curr_stroke = 0
        self.answer = category

        # Return total number of strokes in the image
        return self.stroke_q.qsize()

    def get_next_stroke(self) -> list[list[int], list[int]]:
        if not self.stroke_q.empty():
            self.curr_stroke += 1
            return self.stroke_q.get()
        return False

    def validate(self, guess: str):
        if guess == self.answer:
            return self.get_score()
        return False

    def get_score(self) -> float:
        return 100 / (self.curr_stroke ** 2)

    def get_categories(self):
        categories = os.listdir(self.directory)
        # Filter out empty categories
        non_empty_categories = [i.replace(".ndjson", "").strip() for i in categories if ".ndjson" in i and os.path.getsize(os.path.join(self.directory, i)) > 0]
        return non_empty_categories

    def pick_case_from_file(self, category: str):
        with open(self.directory + "/{}.ndjson".format(category)) as file:
            line = next(file)
            for num, aline in enumerate(file, 2):
                if randrange(num):
                    continue
                line = aline
        return line
