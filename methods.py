import json
import re


class DictBasedRestoration:
    def __init__(self, dict_file_path):
        with open(dict_file_path) as f:
            self.ocr_dict = json.load(f)
        self.ocr_dict = dict((re.escape(k), v) for k, v in self.ocr_dict.items())
        self.ocr_pattern = re.compile("|".join(self.ocr_dict.keys()))

    def dict_pattern_match(self, match):
        return self.ocr_dict[re.escape(match.group(0))]

    def __call__(self, text):
        return self.ocr_pattern.sub(self.dict_pattern_match, text)


class OCR(DictBasedRestoration):
    def __init__(self, dict_file_path="data/ocr_dict.json"):
        super().__init__(dict_file_path)


class Simchar(DictBasedRestoration):
    def __init__(self, dict_file_path="data/simchar_dict.json"):
        super().__init__(dict_file_path)
