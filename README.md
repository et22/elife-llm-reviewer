# eLife LLM Reviewer

### Overview
This repo contains an adaptation of the ICLR LLM reviewer from the AI Scientist paper to the biology journal eLife. The original code for the ICLR reviewer can be found here: [*The AI Scientist: Towards Fully Automated
Open-Ended Scientific Discovery*](https://github.com/SakanaAI/AI-Scientist). 

### Results
I modified the AI Scientist reviewer to review and score papers submitted to the biology journal eLife. To validate the eLife LLM reviewer, I scraped hundreds of eLife papers, ran the eLife LLM reviewer on a subset of these papers, and compared the true review scores with the LLM predicted scores. Following the AI Scientist paper, I explored the effects of adding reflection steps, reviewer ensembles, and few-shot prompting on the performance of the reviewer. 

To minimize API usage cost, I used gpt-4o-mini and ran experiments on a small 50 paper dataset. I ran four experiments with different review conditions: gpt-4o-mini base, gpt-4o-mini +5 reflections, gpt-4o-mini + 5 reflections + 5 ensembles, gpt-4o-mini + 5 reflections + 1 shot prompting. The total cost for this set of experiments was ~$2. 

Under eLife’s [current review model](https://elifesciences.org/about/elife-assessments), papers are assessed on the strength of the evidence (inadequate, incomplete, solid, convincing, compelling, exceptional) and the significance of the results (useful, valuable, important, fundamental, landmark), but do not receive a binary accept/reject decision. Thus, the LLM reviewer performance is measured using the correlation between true and predicted strength scores and significance scores. 

The gpt-4o-mini + 5 reflections model performed best with a strength correlation of r = 0.28, p = 0.0496 and significance correlation of r = 0.43, p = 0.0021 (**Figure 1**). Unfortunately, eLife reports only an aggregated score rather than scores from each reviewer, so this correlation cannot be benchmarked against the inter-human reliability. 

<p align="center">
<img src="https://github.com/et22/elife-llm-reviewer/blob/main/figures/figure1.png" alt="Figure 1" width="900"/>
</p>

*Figure 1.* Pearson correlation between strength and significance scores from human reviewers vs scores from different LLM reviewers. The correlation for 4o-mini + 5 refl. + 1 shot is undefined because this model only predicted a single score for all papers.


To better understand why the addition of the one shot prompt did not further improve performance, I examined the distribution of true scores vs the distribution predicted by the LLM reviewers (Figure 2). The one shot model predicts ‘convincing’ and ‘important’, the true scores of the one-shot example, at a higher rate than the other reviewers, suggesting the model may be mimicking the example scores and confabulating a reason for them.  


<p align="center">
<img src="https://github.com/et22/paper-implementations/blob/main/rajan2006_eigenvalue_spectra/figure2.png" alt="Eigenvalue density with different variance" width="900"/>
</p>

*Figure 2.* Distribution of strength and significance scores from human eLife reviews and from the different LLM reviewers. 


Although the positive correlation between true and predicted strength scores obtained from the gpt-4o-mini + 5 reflections suggests the LLM reviewer may be capturing some signal, there is ample room for additional optimization. Given additional time and resources, it would be interesting to explore how other hyperparameter choices such as prompt phrasing, model temperature, and number of few shot examples impact the performance of the LLM reviewer. In particular, model temperature should be increased to better capture the diversity of human scores, and the base prompt should be modified to correct for the positivity bias present in the LLM reviews. 


### Reproducing the results 
Results can be reproduced by installing dependencies, setting the `OPENAI_API_KEY` environment variable to your API key, then running the following scripts from the `review_elife_bench/` directory: 

```
python parse_elife.py --num_articles 100

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
```

Plots can be generated with the jupyter notebook `plot_elife_reviews.ipynb`.
### Acknowledgements
AI Scientist code is copied/adapted from [here](https://github.com/SakanaAI/AI-Scientist). ChatGPT was used to generate a large chunk of the code for fetching and parsing eLife XML. 