import requests
import os
import re
import argparse

from collections import OrderedDict
from os.path import join
from elifetools import parseJATS as parser


def remove_angle_brackets(text):
    """
    Removes text in between <> in a string.
    """
    return re.sub(r"<.*?>", "", text)


def get_elife_dois(limit=1000):
    """
    Fetch the DOIs of papers published in eLife using the Europe PMC API.
    Uses the default sort method of Europe PMC ("relevance").

    Args:
    - limit (int): Number of papers to retrieve (max is 1000).

    Returns:
    - List of DOIs.
    """

    base_url = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
    query = 'JOURNAL:"eLife"'
    result_type = "core"
    format_type = "json"

    # request with pagination

    dois = []
    params = {
        "query": query,
        "format": format_type,
        "resultType": result_type,
        "pageSize": limit,
        "cursorMark": "*",
    }
    response = requests.get(base_url, params=params)

    data = response.json()
    for result in data.get("resultList", {}).get("result", []):
        if "doi" in result:
            dois.append(result["doi"])

    return dois


def dois_to_xml_filenames(doi_list):
    """
    Converts DOIs into xml filename in elife-article-xml GitHub.
    We use v1 because this version is the initially submitted manuscript.

    Args:
    - doi_list (list): List of DOIs of elife papers.

    Returns:
    - List of XML filenames.
    """
    xml_filenames = [f"elife-{doi.split('.')[-1]}-v1.xml" for doi in doi_list]
    return xml_filenames


def fetch_elife_article_xml(
    file_to_download, save_dir="./elife_unparsed/", verbose=True
):
    """
    Downloads xml for an eLife article given XML filename.

    Args:
    - file_to_download (str): XML filename.

    Returns:
    - None
    """

    # make save_dir if does not exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    output_path = join(save_dir, file_to_download)

    # GitHub API URL for the contents of the folder
    repo_owner = "elifesciences"
    repo_name = "elife-article-xml"
    folder_path = "articles"
    raw_url = (
        f"https://raw.githubusercontent.com/{repo_owner}/"
        f"{repo_name}/master/{folder_path}/{file_to_download}"
    )

    # Download the selected file
    file_response = requests.get(raw_url)
    if file_response.status_code == 200:
        with open(output_path, "wb") as f:
            f.write(file_response.content)
        if verbose:
            print(f"Downloaded: {file_to_download}")
    else:
        print(file_response)
        if verbose:
            print("Error downloading file.")


def save_paper(paper_text, save_dir, file_to_download):
    """
    Saves the parsed paper.

    Args:
    - paper_text (str): Paper content.
    - save_dir (str): Directory to save the paper.
    - file_to_download (str): XML filename of the paper.

    Returns:
    - None
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    output_path = join(save_dir, file_to_download[:-4] + ".txt")

    with open(output_path, "w") as f:
        f.write(paper_text)


def append_ratings(file_to_download, strength, significance):
    """
    Adds strength and significance score to ratings.tsv.

    Args:
    - file_to_download (str): XML filename of the paper.
    - strength (str): eLife strength score.
    - significance (str): eLife significance score.

    Returns:
    - None
    """
    with open("ratings.tsv", "a") as f:
        f.write(f"{file_to_download[:-4]}\t{strength}\t{significance}\n")


def extract_text_from_section(section_json, section_title):
    """
    Extracts the complete text from a JSON-encoded section, including
    figure captions, and table captions, but excluding figure and table data.

    Parameters:
        section_json (list): JSON representation of a paper section.

    Returns:
        str: The complete extracted text.
    """

    if section_title != "":
        extracted_text = [f"\n\n{section_title.upper()}"]
    else:
        extracted_text = []
    for item in section_json:
        if isinstance(item, OrderedDict):
            if item.get("type") == "section":
                extracted_text.append(f"{item.get('title', '')}")
                extracted_text.append(
                    extract_text_from_section(item.get("content", []), "")
                    + "\n"
                )
            elif item.get("type") == "paragraph":
                extracted_text.append(f'{item.get("text", "")}')
            elif item.get("type") == "figure":
                for asset in item.get("assets", []):
                    if asset.get("type") == "image":
                        extracted_text.append(
                          f"{asset.get('label', '')}: {asset.get('title', '')}"
                        )
                        for caption in asset.get("caption", []):
                            if caption.get("type") == "paragraph":
                                extracted_text[-1] = (
                                    extracted_text[-1]
                                    + " "
                                    + (f'{caption.get("text", "")}')
                                )
            elif item.get("type") == "table":
                extracted_text.append(
                    f"{item.get('label', '')}: {item.get('title', '')}"
                )
                for caption in item.get("caption", []):
                    if caption.get("type") == "paragraph":
                        extracted_text[-1] = (
                            extracted_text[-1]
                            + " "
                            + (f'{caption.get("text", "")}')
                        )

    return "\n".join(extracted_text)


def parse_paper(file_to_download, folder="./elife_unparsed"):
    """
    Parses all sections of the paper from XML.

    Args:
    - file_to_download (str): XML filename.
    - folder (str): Location of unparsed XML files.

    Returns:
    - success (bool): True if parsing was succesful.
    - paper (str): Text of the parsed paper.
    - strength (str): eLife strength score for the paper.
    - significance (str): eLife significance score for the paper.
    """
    soup = parser.parse_document(f"{folder}/{file_to_download}")

    success = True

    try:
        if "strength" not in parser.elife_assessment(soup):
            raise ValueError("Paper has not been reviewed.")

        strength = parser.elife_assessment(soup)["strength"][0]
        significance = parser.elife_assessment(soup)["significance"][0]

        title = "TITLE\n" + parser.title(soup) + "\n\n"
        abstract = ("ABSTRACT\n"
                    + parser.abstract_json(soup)["content"][0]["text"])

        paper = title + abstract

        for elem in parser.body_json(soup):
            elem_json = elem["content"]
            elem = extract_text_from_section(
                elem_json, elem["title"]
            )  # parse_section(elem_json, elem["title"])
            paper += elem

        paper = remove_angle_brackets(paper)

    except Exception as e:
        print("Skipping paper... ", e)
        success = False
        paper = ""
        strength = ""
        significance = ""

    return success, paper, strength, significance


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "-n_articles",
        "--num_articles",
        type=int,
        default=10,
        help="Number of articles to fetch",
    )
    args = argparser.parse_args()
    num_articles = args.num_articles

    elife_dois = get_elife_dois(num_articles)
    xml_filenames = dois_to_xml_filenames(elife_dois)

    for file_to_download in xml_filenames:
        fetch_elife_article_xml(file_to_download)

    # delete ratings file if it exists
    if os.path.exists("ratings.tsv"):
        os.remove("ratings.tsv")

    with open("ratings.tsv", "a") as f:
        f.write("paper_id\tstrength\tsignificance\n")

    total = 0
    for file_to_download in xml_filenames:
        print(f"Parsing {file_to_download}...")
        success, paper, strength, significance = parse_paper(file_to_download)

        if success:
            save_paper(paper, "elife_parsed", file_to_download)
            append_ratings(file_to_download, strength, significance)
            total += 1

    print(f"Total successfully parsed: {int(total)}/{len(xml_filenames)}")
