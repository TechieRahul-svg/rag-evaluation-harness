import json
from loader import load_markdown_files
from cleaner import clean_markdown
from metadata import extract_metadata


RAW_PATH = "data/raw"

OUTPUT_PATH = "data/processed/documents.json"


def process_documents():

    documents = load_markdown_files(RAW_PATH)

    processed_documents = []


    for index, doc in enumerate(documents):

        cleaned_text = clean_markdown(
            doc["content"]
        )

        metadata = extract_metadata(
            doc["path"]
        )


        processed_documents.append(
            {
                "id": index + 1,

                "text": cleaned_text,

                "metadata": metadata
            }
        )


    return processed_documents



if __name__ == "__main__":

    documents = process_documents()


    with open(
        OUTPUT_PATH,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            documents,
            file,
            indent=4,
            ensure_ascii=False
        )


    print(
        f"Saved {len(documents)} documents"
    )

    print(
        f"Output: {OUTPUT_PATH}"
    )