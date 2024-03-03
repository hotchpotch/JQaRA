"""
GitHub の最新の README.md を Hugging Face にアップロードするスクリプト

"""

import difflib
from io import StringIO

import requests
from huggingface_hub import HfApi


def upload_readme_to_hf(readme: str):
    readme_io = StringIO(readme).getvalue().encode("utf-8")

    api = HfApi()
    api.upload_file(
        path_or_fileobj=readme_io,
        path_in_repo="README.md",
        repo_id="hotchpotch/JQaRA",
        repo_type="dataset",
    )


def main():
    github_raw_readme_url = (
        "https://raw.githubusercontent.com/hotchpotch/JQaRA/main/README.md"
    )
    hf_raw_readme_url = (
        "https://huggingface.co/datasets/hotchpotch/JQaRA/resolve/main/README.md"
    )

    github_readme_text = requests.get(github_raw_readme_url).text
    hf_readme_text: str = requests.get(hf_raw_readme_url).text
    _, hf_metadata, hf_text = hf_readme_text.split("---", maxsplit=2)

    print(f"GitHub README: text / {len(github_readme_text)}")
    print(f"Hugging Face README: metadata / {len(hf_metadata)},  text / {len(hf_text)}")

    new_hf_readme_text = "---\n" + hf_metadata + "---\n" + github_readme_text

    differ = difflib.Differ()
    diff = differ.compare(hf_readme_text.splitlines(), new_hf_readme_text.splitlines())

    diff_text = "\n".join(
        [line for line in diff if line.startswith("+ ") or line.startswith("- ")]
    )
    print("# ---- Diff ----")
    print(diff_text)

    ok = input("Upload the README to Hugging Face? [y/N]: ")
    if ok.lower() == "y":
        upload_readme_to_hf(new_hf_readme_text)
        print("Uploaded the README to Hugging Face.")
    else:
        print("Canceled.")


if __name__ == "__main__":
    main()
