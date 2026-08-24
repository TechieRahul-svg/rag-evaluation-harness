from pathlib import Path


def extract_metadata(path):

    path = Path(path)

    parts = path.parts

    if "raw" in parts:

        raw_index = parts.index("raw")

        if len(parts) > raw_index + 2:
            category = parts[raw_index + 1]
        else:
            category = "general"

        source_parts = parts[raw_index + 1:]

        source = "/".join(source_parts)

    else:

        category = "unknown"

        source = path.as_posix()


    return {
        "source": source,
        "category": category,
        "title": path.stem
    }