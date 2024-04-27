# llmamd - LLM assisted model development

This project was created to test the hypothesis that an LLM can be used to enhance the performance of simpler text classifiers.  This is done by using ChatGPT 3.5 Turbo as the LLM to programatically generate text that augments the original training data.

The LLM is used to generate the number of samples that are originally present in the training data effectively doubling the training samples.  The LLM is prompted to generate text samples in two specific ways.  The first way directs the LLM to generate samples from the positive class.  The second way directs the LLM to generate samples from the LLM to generate samples from the negative class.

The iterations of the prompts used for each of these generation scenarios are found in the file: `data/prompt_log.csv`

The table below describes the main files of the analysis:

| File | Description |
|-------|------------|
| docs/Final_paper.pdf | report summarizing all the work done in this project |
| docs/Research_Paper.pdf | report describing the details of approach and methodology used for this project |
| docs/project_proposal.pdf | initial proposal for the project (note: ChatGPT was used instead of Hugging Face model) |
| docs/data_quality_profiling.pdf | initial exploratory data analysis (EDA) of the kaggle disaster data |
| projtool.py | module of functions written for this analysis |
| preproc_pipeline.ipynb | development of the pre-processing pipeline |
| preproc_disaster.ipynb | implementation of the full pre-processing pipeline |
| chatgpt_tweet_gen.ipynb | generates the augmented data samples |
| logistic_regressions.ipynb | implementation of the logistic regressions model |
| neural_net.ipynb | implementation of the neural network model |
| vectorize_tweets.ipynb | converts the processed tweets to vector |
| data/data_readme.txt | describes the files in the data directory |
| data/test.csv | original (unprocessed) kaggle test data |
| data/train.csv | original (unprocessed) kaggle training data |
| data/vocab.txt | vocabulary established for this analysis |
|  |  |
|  |  |
|  |  |
|  |  |
