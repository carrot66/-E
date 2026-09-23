"""Download the frozen model snapshots for transfer to an offline server.

Run this on a machine that can reach Hugging Face, then upload the resulting
models/ directory to the server. It does not download video data.
"""
import argparse
from pathlib import Path
from huggingface_hub import snapshot_download
from extractors import MODEL_IDS

PATTERNS = ['*.json', '*.txt', '*.safetensors', 'pytorch_model.bin', '*.model']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parent/'models')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    for key, repo in MODEL_IDS.items():
        destination = args.output / key
        destination.mkdir(parents=True, exist_ok=True)
        path = snapshot_download(repo_id=repo, local_dir=str(destination),
                                 allow_patterns=PATTERNS)
        print(f'{key}: {path}')
    print(f'Upload the complete directory: {args.output.resolve()}')


if __name__ == '__main__':
    main()
