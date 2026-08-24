from pathlib import Path


def load_markdown_files(folder):

    documents = []

    folder = Path(folder)

    for file in folder.rglob("*.md"):

        content = file.read_text(
            encoding="utf-8"
        )

        documents.append(
            {
                "path": str(file),
                "content": content
            }
        )

    return documents



if __name__ == "__main__":

    docs = load_markdown_files(
        "data/raw"
    )

    print(
        "Documents loaded:",
        len(docs)
    )

    print(docs[0]["path"])