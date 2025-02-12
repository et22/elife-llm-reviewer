"""
python review_elife.py --num_reviews 1  --batch_size 1 \
    --num_fs_examples 0 --num_reflections 1 --temperature 0.1 \
    --num_reviews_ensemble 1

python review_elife.py --num_reviews 1  --batch_size 1 \
    --num_fs_examples 0 --num_reflections 5 --temperature 0.1 \
    --num_reviews_ensemble 1

python review_elife.py --num_reviews 1  --batch_size 1 \
    --num_fs_examples 0 --num_reflections 5 --temperature 0.1 \
    --num_reviews_ensemble 5

python review_elife.py --num_reviews 1  --batch_size 1 \
    --num_fs_examples 1 --num_reflections 5 --temperature 0.1 \
    --num_reviews_ensemble 5

"""

import pathlib
import pandas as pd
import numpy as np
import requests
import argparse
import os
import time
import multiprocessing as mp
from sklearn.utils import shuffle
from sklearn.metrics import f1_score, roc_auc_score, confusion_matrix

from ai_scientist.perform_review import (
    perform_review,
    reviewer_system_prompt_base,
    elife_form,
)


def parse_arguments():
    parser = argparse.ArgumentParser(description="Run AI reviewer experiments")
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o-mini-2024-07-18",
        choices=[
            "gpt-4o-mini-2024-07-18",
            "gpt-4o-2024-05-13",
            "gpt-4o-2024-08-06",
            "llama-3-1-405b-instruct",
            "deepseek-coder-v2-0724",
            "claude-3-5-sonnet-20240620",
        ],
        help="Model to use for AI Scientist.",
    )

    parser.add_argument(
        "--num_reviews",
        type=int,
        default=20,
        help="Number of reviews to generate.",
    )

    parser.add_argument(
        "--num_reflections",
        type=int,
        default=3,
        help="Number of reviews to generate.",
    )

    # add argument for few shot prompting
    parser.add_argument(
        "--num_fs_examples",
        type=int,
        default=2,
        help="Number of model reviews",
    )

    # add review ensembling
    parser.add_argument(
        "--num_reviews_ensemble",
        type=int,
        default=1,
        help="Number of model reviews to ensemble.",
    )

    # batch size for evals with mp
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Review batchsize.",
    )
    # pages to extract from pdf
    parser.add_argument(
        "--num_paper_pages",
        type=int,
        default=0,
        help="Paper pages to extract from pdf (0 - all).",
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=0.75,
        help="GPT temperature.",
    )
    return parser.parse_args()


# Create a new dataframe that stores the LLM reviews
llm_cols = [
    "paper_id",
    "Assessment",
    "Significance",
    "Strength of Evidence",
    "Summary",
    "Strengths",
    "Weaknesses",
    "Achievement",
    "Impact",
    "Context",
]


def prep_elife_data(
    ratings_path="ratings.tsv",
    data_seed=1,
    balanced_val=False,
    num_reviews=-1,
):
    ratings = pd.read_csv(ratings_path, sep="\t", index_col=0)
    ratings["paper_id"] = ratings.index

    ratings["simplified_decision"] = np.logical_and(
        ratings["significance"]
        .isin(["landmark", "fundamental", "important"])
        .to_numpy(),
        ratings["strength"]
        .isin(["exceptional", "compelling", "convincing"])
        .to_numpy(),
    )

    ratings = shuffle(ratings, random_state=data_seed)
    ratings.drop_duplicates(inplace=True)
    # Select 50% accept and 50% reject from all papers - only for meta evo
    # Weird indexing since some papers/discussions don't seem to be available
    # on OpenReview any more
    if balanced_val:
        ratings = ratings.groupby("simplified_decision").apply(
            lambda x: x.sample(n=int(num_reviews / 2), random_state=data_seed)
        )
        ratings = shuffle(ratings, random_state=data_seed)
        ratings = ratings.set_index("paper_id")
    return ratings


def get_perf_metrics(llm_ratings, ore_ratings):
    try:
        llm_ratings = llm_ratings.set_index("paper_id")
    except Exception:
        pass

    llm_ratings["Decision"] = np.logical_and(
        llm_ratings["Significance"]
        .isin(["landmark", "fundamental", "important"])
        .to_numpy(),
        llm_ratings["Strength of Evidence"]
        .isin(["exceptional", "compelling", "convincing"])
        .to_numpy(),
    )

    num_llm_reviews = llm_ratings.shape[0]
    # Get overall accuracy of decisions made by gpt
    correct = 0
    y_pred = []
    y_true = []
    for i in range(num_llm_reviews):
        name = llm_ratings.iloc[i].name
        if (
            llm_ratings["Decision"].loc[name]
            == ore_ratings["simplified_decision"].loc[name]
        ):
            correct += 1

        y_pred.append(llm_ratings["Decision"].loc[name])
        y_true.append(ore_ratings["simplified_decision"].loc[name])

    accuracy = correct / num_llm_reviews
    accuracy = round(accuracy, 2)
    f1 = round(f1_score(y_true, y_pred), 2)
    try:
        roc = round(roc_auc_score(y_true, y_pred), 2)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        fpr = fp / (fp + tn)
        fnr = fn / (fn + tp)

    except Exception:
        roc = 0
        fpr, fnr = 0, 0

    return accuracy, f1, roc, fpr, fnr


def download_paper_pdf(url, paper_id, verbose=True):
    # make path for paper pdf
    if not os.path.exists("iclr_papers"):
        os.makedirs("iclr_papers", exist_ok=True)
    # Download pdf and write to file
    paper_pdf = os.path.join("iclr_papers", f"{paper_id}.pdf")
    if not os.path.exists(paper_pdf):
        response = requests.get(url)
        with open(paper_pdf, "wb") as f:
            f.write(response.content)
        if verbose:
            print(f"Downloaded {paper_pdf}")
    else:
        if verbose:
            print(f"File {paper_pdf} already exists")
    return paper_pdf


def review_single_paper(
    idx,
    model,
    ore_ratings,
    llm_ratings,
    num_reflections,
    num_fs_examples,
    num_reviews_ensemble,
    temperature,
    reviewer_system_prompt,
    review_instruction_form,
    num_paper_pages,
):
    # Setup client for LLM model
    if model == "claude-3-5-sonnet-20240620":
        import anthropic

        client = anthropic.Anthropic()
    elif model.startswith("bedrock") and "claude" in model:
        import anthropic

        model = model.split("/")[-1]
        client = anthropic.AnthropicBedrock()
    elif model.startswith("vertex_ai") and "claude" in model:
        import anthropic

        # Expects: vertex_ai/<MODEL_ID>
        model = model.split("/")[-1]
        client = anthropic.AnthropicVertex()
    elif model in [
        "gpt-4o-2024-05-13",
        "gpt-4o-mini-2024-07-18",
        "gpt-4o-2024-08-06",
    ]:
        import openai

        client = openai.OpenAI()
    elif model == "deepseek-coder-v2-0724":
        import openai

        client = openai.OpenAI(
            api_key=os.environ["DEEPSEEK_API_KEY"],
            base_url="https://api.deepseek.com"
        )
    elif model == "llama-3-1-405b-instruct":
        import openai

        client = openai.OpenAI(
            api_key=os.environ["OPENROUTER_API_KEY"],
            base_url="https://openrouter.ai/api/v1",
        )
    else:
        raise ValueError(f"Model {model} not supported.")

    rating = ore_ratings.iloc[idx]
    if rating.paper_id in llm_ratings.index:
        print(f"{idx}: Review for {rating.paper_id} already exists")
        return {"idx": idx, "review": None}
    try:
        txt_path = f"elife_parsed/{rating.paper_id}.txt"
        if not os.path.exists(txt_path):
            raise ValueError("paper should already have been parsed!")
        else:
            with open(txt_path, "r") as f:
                text = f.read()
    except Exception as e:
        print(f"Error loading PDF: {e}")
        return {"idx": idx, "review": None}

    try:
        llm_review = perform_review(
            text,
            model,
            client,
            num_reflections,
            num_fs_examples,
            num_reviews_ensemble,
            temperature,
            reviewer_system_prompt=reviewer_system_prompt,
            review_instruction_form=review_instruction_form,
        )
    except Exception as e:
        print(f"Error in worker: {e}")
        return {"idx": idx, "review": None}

    return {"idx": idx, "review": llm_review}


def worker(
    input_queue,
    output_queue,
):
    while True:
        inputs = input_queue.get()
        if inputs is None:
            break
        result = review_single_paper(*inputs)
        output_queue.put(result)


def elife_review_validate(
    num_reviews,
    model,
    rating_fname,
    batch_size,
    num_reflections,
    num_fs_examples,
    num_reviews_ensemble,
    temperature,
    reviewer_system_prompt,
    review_instruction_form,
    num_paper_pages=None,
    data_seed=1,
    balanced_val=False,
):
    ore_ratings = prep_elife_data(
        data_seed=data_seed,
        balanced_val=balanced_val,
        num_reviews=num_reviews,
    )
    # Try loading llm ratings file otherwise create new one
    try:
        llm_ratings = pd.read_csv(rating_fname, index_col="paper_id")
        print(f"Loaded existing LLM reviews dataframe: {rating_fname}")
    except FileNotFoundError:
        # Set index name of a pandas dataframe
        llm_ratings = pd.DataFrame(columns=llm_cols)
        llm_ratings.set_index("paper_id", inplace=True)
        print(f"Created new LLM reviews dataframe: {rating_fname}")

    num_review_batches = num_reviews // batch_size
    paper_id = 0
    for i in range(num_review_batches):
        print(f"Start batch: {i + 1} / {num_review_batches}")
        # Track time used for each review - Collect evals for batch of papers
        start_time = time.time()
        batch_idx = np.arange(paper_id, paper_id + batch_size)

        # Set up queues for multiprocessing
        input_queue = mp.Queue()
        output_queue = mp.Queue()
        processes = []
        for _ in range(batch_size):
            p = mp.Process(target=worker, args=(input_queue, output_queue))
            p.start()
            processes.append(p)

        for idx in batch_idx:
            input_queue.put(
                [
                    idx,
                    model,
                    ore_ratings,
                    llm_ratings,
                    num_reflections,
                    num_fs_examples,
                    num_reviews_ensemble,
                    temperature,
                    reviewer_system_prompt,
                    review_instruction_form,
                    num_paper_pages,
                ]
            )
        for _ in range(batch_size):
            input_queue.put(None)

        # Collect results from the output queue
        llm_reviews = []
        for _ in range(batch_size):
            llm_reviews.append(output_queue.get())

        # Ensure all processes have finished
        for p in processes:
            p.join()

        # Check if all llm_cols are in the llm generated review
        for i_x in range(batch_size):
            idx = llm_reviews[i_x]["idx"]
            review = llm_reviews[i_x]["review"]
            if review is not None:
                correct_review = (sum([k in review for k in llm_cols[1:]])
                                  == len(llm_cols[1:]))
                if correct_review:
                    rating = ore_ratings.iloc[idx]
                    # Add the reviews to the rankings dataframe as a new row
                    llm_ratings.loc[rating.paper_id] = review
                    llm_ratings.to_csv(rating_fname)
                    
                else:
                    print(f"{i + 1}/{batch_size}: Review is incomplete.")
                    continue
            else:
                continue

        # Format string so that only two decimals are printed
        print(
            f"End batch: {i + 1} / {num_review_batches}" + 
            f" : Time used: {(time.time() - start_time):.2f}s"
        )
        print(75 * "=")
        paper_id += batch_size

    return llm_ratings, ore_ratings


if __name__ == "__main__":
    args = parse_arguments()

    temperature = str(args.temperature).replace(".", "_")
    rating_fname = f"llm_reviews/{args.model}_temp_{temperature}"
    pathlib.Path("llm_reviews/").mkdir(parents=True, exist_ok=True)

    if args.num_fs_examples > 0:
        rating_fname += f"_fewshot_{args.num_fs_examples}"

    if args.num_reflections > 1:
        rating_fname += f"_reflect_{args.num_reflections}"

    if args.num_reviews_ensemble > 1:
        rating_fname += f"_ensemble_{args.num_reviews_ensemble}"

    num_paper_pages = (None if args.num_paper_pages == 0
                       else args.num_paper_pages)
    if num_paper_pages is not None:
        rating_fname += f"_pages_{num_paper_pages}"
    else:
        rating_fname += "_pages_all"

    # Settings for reviewer prompt
    reviewer_system_prompt = reviewer_system_prompt_base
    reviewer_form_prompt = elife_form
    rating_fname += ".csv"

    elife_review_validate(
        args.num_reviews,
        args.model,
        rating_fname,
        args.batch_size,
        args.num_reflections,
        args.num_fs_examples,
        args.num_reviews_ensemble,
        args.temperature,
        reviewer_system_prompt,
        reviewer_form_prompt,
        num_paper_pages,
        balanced_val=False,
    )
