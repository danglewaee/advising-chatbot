import json
import os
from pathlib import Path

def txt_to_json(input_file, output_file, flags_as_key=[]):
    """
    Converts a .txt file into a .json file.
    Each line starting with an element of flags_as_key is treated as a key.
    Each line between 2 keys is treated as part of the value for the preceding key.
    Values are list of strings.
    """
    data = {}
    current_key = "default"
    data[current_key] = []
    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            is_key_line = False
            for flag in flags_as_key:
                if line.startswith(flag):
                    current_key = line
                    data[current_key] = []
                    is_key_line = True
                    break

            if not is_key_line:
                data[current_key].append(line)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)

def mass_txt_to_json(input_dir, output_dir, flags_as_key=[]):
    """
    Converts all .txt files in the input directory to .json files in the output directory.
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for txt_file in input_path.glob("*.txt"):
        json_file = output_path / (txt_file.stem + ".json")
        txt_to_json(txt_file, json_file, flags_as_key)

if __name__ == "__main__":
    input_directory = "data/raw/bulk"
    output_directory = "data/processed/json"
    mass_txt_to_json(input_directory, output_directory, flags_as_key=["CICS", "COMPSCI", "CS", "INFO"])