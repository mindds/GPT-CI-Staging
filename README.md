# GPT-CI-Staging

This repository provides code for the study titled "A GPT-4o-powered framework for identifying cognitive impairment stages in electronic health records" which uses GPT-4o to automate the extraction and interpretation of cognitive information from electronic health records (EHRs). The framework was evaluated across two key clinical tasks: **Cognitive Impairment (CI) stage classification** and **Clinical Dementia Rating (CDR) scoring**.

## Study Overview

In this study, we introduce a **GPT-4o-powered framework** for automating cognitive assessment from unstructured clinical notes. Our evaluation used two real-world datasets:

1. **CI Stage Classification**  
   We applied the framework to classify patients as **Cognitively Unimpaired (CU)**, **Mild Cognitive Impairment (MCI)**, or **Dementia** using a dataset of 1,002 Medicare fee-for-service patients from the **Mass General Brigham (MGB) Healthcare Accountable Care Organization (ACO)**.  
   GPT-4o’s performance was compared with several other frameworks to assess its language understanding capabilities and potential in clinical settings.

2. **CDR Scoring**  
   We further evaluated GPT-4o on the task of assigning **global Clinical Dementia Rating (CDR)** scores using specialist notes from patients who visited the MGB memory clinic.

Beyond performance evaluation, we explored the design of an **interactive AI agent** that integrates the GPT-4o-powered framework to enable real-time interaction and decision support for cognitive diagnoses.

## Repository Structure

- `ci_staging/`  
  Contains the full pipeline for **CI stage classification**, including GPT inference, evaluation and comparison across different frameworks. 

- `cdr_scoring/`  
  Contains the pipeline for **CDR score assignment**, from preprocessing to prompting of GPT and downstream results analysis.


## ⚠️ Notes

- No protected health information (PHI) is included in this repository. All code is shared for reproducibility and academic use.


## Citation

Please cite this repository if you use the codes or models in your research:
```bibtex
[@article{leng2025gptci,
  title     = {A GPT-4o-powered framework for identifying cognitive impairment stages in electronic health records},
  author    = {Leng, Yu and Magdamo, Colin G. and Sheu, Yi-han and Mohite, Prathamesh and Noori, Ayush and Ye, Elissa M. and Ge, Wendong and Sun, Haoqi and Brenner, Laura and Robbins, Gregory and Mukerji, Shibani and Zafar, Sahar and Benson, Nicole and Moura, Lidia and Hsu, John and Hyman, Bradley T. and Westover, Michael B. and Blacker, Deborah and Das, Sudeshna},
  journal   = {npj Digital Medicine},
  volume    = {8},
  number    = {1},
  pages     = {401},
  year      = {2025},
  publisher = {Nature Publishing Group},
  doi       = {10.1038/s41746-025-01834-5},
  pmid      = {40610683},
  pmcid     = {PMC12229571}
}]
