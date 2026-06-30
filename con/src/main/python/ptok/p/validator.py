from pathlib import Path


class VietnameseValidator:
    def __init__(self, dictionary_path: str):
        script_dir = Path(__file__).resolve().parent
        self.file_path = script_dir / dictionary_path
        # Initialize by calling a helper method right away
        self.word_set = self._load_dictionary()

    def _load_dictionary(self) -> set[str]:
        try:
            with open(self.file_path, 'r', encoding='utf-8') as file:
                return {line.strip() for line in file if line.strip()}
        except FileNotFoundError:
            print(f"Warning: {self.file_path} not found. Starting with an empty set.")
            return set()

    def is_valid_syllable(self, word: str) -> bool:
        return word.lower() in self.word_set

