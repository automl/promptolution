"""Utils for formatting prompts and outputs."""
from typing import List, Union, overload


@overload
def extract_from_tag(text: str, start_tag: str, end_tag: str) -> str:
    ...


@overload
def extract_from_tag(text: List[str], start_tag: str, end_tag: str) -> List[str]:
    ...


def extract_from_tag(text: Union[str, List[str]], start_tag: str, end_tag: str) -> Union[List[str], str]:
    """Extract content from a string between specified start and end tags.

    Args:
        text (str): The input text to extract from.
        start_tag (str): The start tag to look for.
        end_tag (str): The end tag to look for.

    Returns:
        Union[List[str], str]: The extracted content, either as a list or a single string.
    """
    was_list = True
    if isinstance(text, str):
        text = [text]
        was_list = False

    outs = []
    for t in text:
        out = t.split(start_tag)[-1].split(end_tag)[0].strip()
        outs.append(out)
    if was_list:
        return outs
    return outs[0]
