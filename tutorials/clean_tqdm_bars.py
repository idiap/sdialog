import os
import json
import sys


def clean_tqdm_bars_in_notebook(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        notebook = json.load(f)

    modified = False
    for cell in notebook.get('cells', []):
        if cell.get('cell_type') == 'code':
            original_outputs = cell.get('outputs', [])
            cleaned_outputs = [
                output for output in original_outputs
                if "data" not in output or 'application/vnd.jupyter.widget-view+json' not in output['data']
            ]
            if len(cleaned_outputs) != len(original_outputs):
                cell['outputs'] = cleaned_outputs
                modified = True

    if modified:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(notebook, f, indent=1)
        print(f"Cleaned tqdm bars in: {file_path}")
    else:
        print(f"No tqdm bars found in: {file_path}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python clean_tqdm_bars.py <path_to_folder_with_notebooks>")
        sys.exit(1)

    notebook_folder = sys.argv[1]
    for filename in os.listdir(notebook_folder):
        if filename.endswith(".ipynb"):
            clean_tqdm_bars_in_notebook(os.path.join(notebook_folder, filename))
