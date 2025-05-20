# GPT-CI-Staging

This repository provides code for our study on using GPT-4o to automate the extraction and interpretation of cognitive information from electronic health records (EHRs). The framework was evaluated across two key clinical tasks: **Cognitive Impairment (CI) stage classification** and **Clinical Dementia Rating (CDR) scoring**.

## Study Overview

In this study, we introduce a **GPT-4o-powered framework** for automating cognitive assessment from unstructured clinical notes. Our evaluation used two real-world datasets:

1. **CI Stage Classification**  
   We applied the framework to classify patients as **Cognitively Unimpaired (CU)**, **Mild Cognitive Impairment (MCI)**, or **Dementia** using a dataset of 1,002 Medicare fee-for-service patients from the **Mass General Brigham (MGB) Healthcare Accountable Care Organization (ACO)**.  
   GPT-4o’s performance was compared with several deep learning models to assess its language understanding capabilities and potential in clinical settings.

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
