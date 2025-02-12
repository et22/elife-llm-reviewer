import os
import numpy as np
import json

from ai_scientist.llm import (
    get_response_from_llm,
    get_batch_responses_from_llm,
    extract_json_between_markers,
)

reviewer_system_prompt_base = (
    "You are a researcher who is reviewing a paper that was submitted to the journal eLife."
    "Be critical and cautious in your decision."
)

reviewer_system_prompt_neg = (
    reviewer_system_prompt_base
    + "If a paper is bad or you are unsure, give it bad scores."
)
reviewer_system_prompt_pos = (
    reviewer_system_prompt_base
    + "If a paper is good or you are unsure, give it good scores."
)

template_instructions = """
Respond in the following format:

THOUGHT:
<THOUGHT>

REVIEW JSON:
```json
<JSON>
```

In <THOUGHT>, first briefly discuss your intuitions and reasoning for the evaluation.
Detail your high-level arguments, necessary choices and desired outcomes of the review.
Do not make generic comments here, but be specific to your current paper.
Treat this as the note-taking phase of your review.

In <JSON>, provide the review in JSON format with the following fields in the order:
- "Assessment": A very short summary of your assessment of the work and its likely impact in the field, outside the field, or in society. Please summarise the significance of the findings and the strength of the evidence.
- "Significance": A one-word rating of the significance of the findings (Useful, Valuable, Important, Fundamental or Landmark).
- "Strength of Evidence": A one-word rating of the strength of evidence (Inadequate, Incomplete, Solid, Convincing, Compelling or Exceptional).
- "Summary": A brief summary of what the authors were trying to achieve.
- "Strengths": A list of strengths of the methods and results.
- "Weaknesses": A list of weaknesses of the methods and results.
- "Achievement": An brief appraisal of whether the authors achieved their aims, and whether the results support their conclusions.
- "Impact": A brief discussion of the likely impact of the work on the field, and the utility of the methods and data to the community.
- "Context": Any brief additional context you think would help readers interpret or understand the significance of the work.

This JSON will be automatically parsed, so ensure the format is precise.
"""

elife_form = (
    """
## Review Form
Below is a description of the questions you will be asked on the review form for each paper and some guidelines on what to consider when answering these questions.
When writing your review, please keep in mind that after decisions have been made, reviews of papers will be made public. 

The review form includes the following main sections:

1. Assessment (between two and four sentences)
This is a very short summary of the reviewer’s assessment of the work and its likely impact in the field, outside the field, or in society. This assessment should be easily readable by non-experts, and should clearly convey the judgment of the reviewer about whether the papers’ primary claims are supported by the data, and to whom the manuscript will be of interest or use. 

In one or two sentences, please summarise the significance of the findings and the strength of the evidence presented in support of them, using using the most appropriate terms and some of the phrases below (edited as needed):

A series of example eLife assessments follow:

This fundamental work substantially advances our understanding of protein import into peroxisomes by identifying a novel player in this process and uncovering its mode of action. The evidence supporting the conclusions is compelling, with rigorous biochemical assays and state-of-the-art microscopy. The work will be of broad interest to cell biologists and biochemists.
This study presents a valuable finding on the increased activity of two well-studied signal transduction pathways in a specific subtype of breast cancer. The evidence supporting the claims of the authors is solid, although inclusion of a larger number of patient samples and an animal model would have strengthened the study. The work will be of interest to medical biologists working on breast cancer.
This important study combines experiments and theory to quantify the force exerted on chromosomes during cell division. The new method for force measurements is highly compelling and goes beyond the current state of the art, but the theoretical analysis is incomplete and would benefit from more rigorous approaches. With the theoretical part strengthened, this paper would be of interest to cell biologists and biophysicists working on the cytoskeleton and cell division.
This study presents a useful inventory of genes that are up- and down regulated in human heart tissue during aging. The data were collected and analyzed using solid and validated methodology and can be used as a starting point for functional studies of heart development and disease.
This paper reports the fundamental discovery of a new mode of mammalian cell migration, which does not involve either actin or microtubule cytoskeleton. If confirmed, the study will change the way we think about cell motility and would be of very broad general interest. However, whereas some of the imaging data are compelling, the functional analyses are inadequate as they rely on a very limited set of pharmacological treatments.
This landmark study provides a comprehensive morphological and molecular description of the majority of documented neuronal cell types in the mouse cortex. This provides an extraordinary resource that will be invaluable to the whole neuroscience community. The methodology for combining expansion microscopy with spatially resolved transcriptomics across tissues is exceptional and establishes a new standard in the field.

2. Significance (one word). Please describe the significance of the findings. Choices: 
    Landmark: findings with profound implications that are expected to have widespread influence
    Fundamental: findings that substantially advance our understanding of major research questions
    Important: findings that have theoretical or practical implications beyond a single subfield
    Valuable: findings that have theoretical or practical implications for a subfield
    Useful: findings that have focused importance and scope

3. Strength of Evidence (one word). Please describe the strength of evidence. Choices: 
    Exceptional: exemplary use of existing approaches that establish new standards for a field
    Compelling: evidence that features methods, data and analyses more rigorous than the current state of the art
    Convincing: appropriate and validated methodology in line with current state-of-the-art
    Solid: methods, data and analyses broadly support the claims with only minor weaknesses
    Incomplete: main claims are only partially supported
    Inadequate: methods, data and analyses do not support the primary claims

4. Summary (two to three sentences). A summary of what the authors were trying to achieve. 

5. Strengths (one to two sentences). An account of the major strengths of the methods and results.

6. Weaknesses (one to two sentences). An account of the major weaknesses of the methods and results.

7. Achievement (one to two sentences). An appraisal of whether the authors achieved their aims, and whether the results support their conclusions.

8. Impact (one sentence). A discussion of the likely impact of the work on the field, and the utility of the methods and data to the community.

9. Context (one sentence). Any additional context you think would help readers interpret or understand the significance of the work.

"""
    + template_instructions
)

dir_path = os.path.dirname(os.path.realpath(__file__))

fewshot_papers = [
    os.path.join(dir_path, "fewshot_examples/motorcortex.txt"),
]

fewshot_reviews = [
    os.path.join(dir_path, "fewshot_examples/motorcortex.json"),
]

def get_review_fewshot_examples(num_fs_examples=1):
    fewshot_prompt = """
Below is a sample review, copied from previous eLife papers. Note that the review provided below is not neccesarily formatted in the way that you should respond. It is merely meant to provide general guidance on the potential content of a review. 
"""
    for paper, review in zip(
        fewshot_papers[:num_fs_examples], fewshot_reviews[:num_fs_examples]
    ):
        with open(paper, "r") as f:
            paper_text = f.read()

        with open(review, "r") as json_file:
            review_text = json.load(json_file)

        fewshot_prompt += f"""
Paper:

```
{paper_text}
```

Review:

```
{review_text}
```
"""

    return fewshot_prompt


def perform_review(
    text,
    model,
    client,
    num_reflections=1,
    num_fs_examples=1,
    num_reviews_ensemble=1,
    temperature=0.75,
    msg_history=None,
    return_msg_history=False,
    reviewer_system_prompt=reviewer_system_prompt_base,
    review_instruction_form=elife_form,
):
    if num_fs_examples > 0:
        fs_prompt = get_review_fewshot_examples(num_fs_examples)
        base_prompt = review_instruction_form + fs_prompt
    else:
        base_prompt = review_instruction_form

    base_prompt += f"""
Here is the paper you are asked to review:
```
{text}
```"""

    if num_reviews_ensemble > 1:
        llm_review, msg_histories = get_batch_responses_from_llm(
            base_prompt,
            model=model,
            client=client,
            system_message=reviewer_system_prompt,
            print_debug=False,
            msg_history=msg_history,
            # Higher temperature to encourage diversity.
            temperature=0.75,
            n_responses=num_reviews_ensemble,
        )
        parsed_reviews = []
        for idx, rev in enumerate(llm_review):
            try:
                parsed_reviews.append(extract_json_between_markers(rev))
            except Exception as e:
                print(f"Ensemble review {idx} failed: {e}")
        parsed_reviews = [r for r in parsed_reviews if r is not None]
        review = get_meta_review(model, client, temperature, parsed_reviews)

        # take first valid in case meta-reviewer fails
        if review is None:
            review = parsed_reviews[0]

        # Replace numerical scores with the average of the ensemble.
        for score, values in [
            (
                "Significance",
                ["useful", "valuable", "important", "fundamental", "landmark"],
            ),
            (
                "Strength of Evidence",
                [
                    "inadequate",
                    "incomplete",
                    "solid",
                    "convincing",
                    "compelling",
                    "exceptional",
                ],
            ),
        ]:
            scores = []
            for r in parsed_reviews:
                if score in r and r[score].lower().strip() in values:
                    scores.append(values.index(r[score].lower().strip()))
            review[score] = values[int(round(np.mean(scores)))]

        # Rewrite the message history with the valid one and new aggregated review.
        msg_history = msg_histories[0][:-1]
        msg_history += [
            {
                "role": "assistant",
                "content": f"""
THOUGHT:
I will start by aggregating the opinions of {num_reviews_ensemble} reviewers that I previously obtained.

REVIEW JSON:
```json
{json.dumps(review)}
```
""",
            }
        ]
    else:
        llm_review, msg_history = get_response_from_llm(
            base_prompt,
            model=model,
            client=client,
            system_message=reviewer_system_prompt,
            print_debug=False,
            msg_history=msg_history,
            temperature=temperature,
        )
        review = extract_json_between_markers(llm_review)

    if num_reflections > 1:
        for j in range(num_reflections - 1):
            text, msg_history = get_response_from_llm(
                reviewer_reflection_prompt,
                client=client,
                model=model,
                system_message=reviewer_system_prompt,
                msg_history=msg_history,
                temperature=temperature,
            )
            review = extract_json_between_markers(text)
            assert review is not None, "Failed to extract JSON from LLM output"

            if "I am done" in text:
                break

    if return_msg_history:
        return review, msg_history
    else:
        return review


reviewer_reflection_prompt = """Round {current_round}/{num_reflections}.
In your thoughts, first carefully consider the accuracy and soundness of the review you just created.
Include any other factors that you think are important in evaluating the paper.
Ensure the review is clear and concise, and the JSON is in the correct format.
Do not make things overly complicated.
In the next attempt, try and refine and improve your review.
Stick to the spirit of the original review unless there are glaring issues.

Respond in the same format as before:
THOUGHT:
<THOUGHT>

REVIEW JSON:
```json
<JSON>
```

If there is nothing to improve, simply repeat the previous JSON EXACTLY after the thought and include "I am done" at the end of the thoughts but before the JSON.
ONLY INCLUDE "I am done" IF YOU ARE MAKING NO MORE CHANGES."""


def load_paper(pdf_path, num_pages=None, min_size=100):
    try:
        if num_pages is None:
            text = pymupdf4llm.to_markdown(pdf_path)
        else:
            reader = PdfReader(pdf_path)
            min_pages = min(len(reader.pages), num_pages)
            text = pymupdf4llm.to_markdown(pdf_path, pages=list(range(min_pages)))
        if len(text) < min_size:
            raise Exception("Text too short")
    except Exception as e:
        print(f"Error with pymupdf4llm, falling back to pymupdf: {e}")
        try:
            doc = pymupdf.open(pdf_path)  # open a document
            if num_pages:
                doc = doc[:num_pages]
            text = ""
            for page in doc:  # iterate the document pages
                text = text + page.get_text()  # get plain text encoded as UTF-8
            if len(text) < min_size:
                raise Exception("Text too short")
        except Exception as e:
            print(f"Error with pymupdf, falling back to pypdf: {e}")
            reader = PdfReader(pdf_path)
            if num_pages is None:
                text = "".join(page.extract_text() for page in reader.pages)
            else:
                text = "".join(page.extract_text() for page in reader.pages[:num_pages])
            if len(text) < min_size:
                raise Exception("Text too short")

    return text


# get directory of this file
dir_path = os.path.dirname(os.path.realpath(__file__))

meta_reviewer_system_prompt = """You are a senior editor at eLife.
You are in charge of meta-reviewing a paper that was reviewed by {reviewer_count} reviewers.
Your job is to aggregate the reviews into a single meta-review in the same format.
Be critical and cautious in your decision, find consensus, and respect the opinion of all the reviewers."""


def get_meta_review(model, client, temperature, reviews):
    # Write a meta-review from a set of individual reviews
    review_text = ""
    for i, r in enumerate(reviews):
        review_text += f"""
Review {i + 1}/{len(reviews)}:
```
{json.dumps(r)}
```
"""
    base_prompt = elife_form + review_text

    llm_review, msg_history = get_response_from_llm(
        base_prompt,
        model=model,
        client=client,
        system_message=meta_reviewer_system_prompt.format(reviewer_count=len(reviews)),
        print_debug=False,
        msg_history=None,
        temperature=temperature,
    )
    meta_review = extract_json_between_markers(llm_review)
    return meta_review


def perform_improvement(review, coder):
    improvement_prompt = '''The following review has been created for your research paper:
"""
{review}
"""

Improve the text using the review.'''.format(
        review=json.dumps(review)
    )
    coder_out = coder.run(improvement_prompt)
