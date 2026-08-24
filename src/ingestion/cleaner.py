import re


def clean_markdown(text):

    # Remove images
    text = re.sub(
        r'!\[.*?\]\(.*?\)',
        '',
        text
    )


    # Convert markdown links
    # [text](url) -> text
    text = re.sub(
        r'\[(.*?)\]\(.*?\)',
        r'\1',
        text
    )


    # Remove markdown symbols
    text = re.sub(
        r'[#>*_`]',
        '',
        text
    )


    # Remove extra spaces
    text = re.sub(
        r'\n\s*\n',
        '\n\n',
        text
    )


    return text.strip()


if __name__ == "__main__":

    sample = """
# FastAPI

FastAPI is a modern framework.

[Docs](https://fastapi.tiangolo.com)

![image](abc.png)
"""


    result = clean_markdown(sample)

    print(result)