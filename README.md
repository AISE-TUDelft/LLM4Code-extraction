# Traces of Memorisation in Large Language Models for Code
Replication package for the paper titled "Traces of Memorisation in Large Language Models for Code"

For questions about the content of this repo, please use the issues board. If you have any questions about the paper, please email the first author.

## Requirements
The requirements can be installed using Conda by running:

> conda create --name LLMExtract --file requirements.txt

Make sure that you have added conda-forge to your conda channels:

> conda config --append channels conda-forge

The code is intended to run on an Nvidia A40 with 48GB Vram, 32GB of RAM and 8 CPU cores. Smaller models can be run with lesser hardware as well, but be sure to change the batchsizes in the scripts.

## Generation
To run an experiment simply run the following commands. The $model_name should be set to the path of a model on HuggingFace, such as Salesforce/codegen2-1B. 

For text models:

> python3 generate_text.py $model_name

For code models:

> python3 generate.py $model_name

## Ethical use
Use the code and concepts shared here ethically. The authors have shared this code to improve the security and safety of LLMs. Do not use this code for malicious purposes. When disclosing data leakage do not needlessly use put the privacy of people at risk. 
