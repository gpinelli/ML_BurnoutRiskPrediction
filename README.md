# Burnout Risk Level Prediction 
 
This is my machine learning project, a 3-class classification of university students' burnout risk (`Low` / `Medium` / `High`) from pre-semester information about their use of Generative AI tools.
 
**Author:** Pinelli Gabriele
 
## Contents
 
- [Overview](#overview)
- [Dataset](#dataset)
- [Repository structure](#repository-structure)
- [How to run](#how-to-run)
## Overview
 
The goal was to build a supervised learning pipeline that predicts `Burnout_Risk_Level` using only information that would be available *before* the end of the semester. The project covers exploratory analysis, leakage prevention, preprocessing, model selection with nested cross-validation, hyperparameter tuning, and an analysis of the final model through learning curve, validation curve and confusion matrices.
 
The metric used throughout is **F1 Macro**, which gives equal weight to the three classes. This matters because the minority class (`High` risk) is the most relevant one in practice.
 
## Dataset
 
Source: [AI Impact on Students (Kaggle)](https://www.kaggle.com/datasets/laveshjadon/ai-impact-on-students/data)
 
The dataset describes the use of Generative AI tools by university students and its effects. It has 50,000 instances, 15 features of mixed type and the target.
 
| Class | Count |
|---|---|
| Low | 16,369 |
| Medium | 21,144 |
| High | 12,487 |
 
**Features used by the models** (pre-semester):
 
| Type | Features |
|---|---|
| Numerical | `Pre_Semester_GPA`, `Weekly_GenAI_Hours`, `Traditional_Study_Hours` |
| Ordinal | `Year_of_Study`, `Prompt_Engineering_Skill`, `Tool_Diversity`, `Institutional_Policy` |
| Nominal | `Major_Category`, `Primary_Use_Case` |
| Boolean | `Paid_Subscription` |
 
**Excluded features:**
 
- `Post_Semester_GPA`, `Perceived_AI_Dependency`, `Anxiety_Level_During_Exams`, `Skill_Retention_Score`: they describe the post-semester situation, so they would not be available in a real prediction setting, and some of them are correlated with the target (data-leakage risk).
- `Student_ID`: uninformative identifier.
 
## Repository structure
 
```
.
├── MLProject_PinelliGabriele502175_githubVersion.ipynb   # full project notebook
├── requirements.txt                                      # Python dependencies
├── data/
│   └── ai_student_impact_dataset.csv                     # dataset (see Dataset section)
└── README.md
```
 
## How to run
 
1. Clone the repository and install the dependencies:
```bash
   pip install -r requirements.txt
```
 
2. Download the dataset from Kaggle (or from the backup folder linked above) and place it at `data/ai_student_impact_dataset.csv`.
3. Open the notebook and run all cells in order:
```bash
   jupyter notebook MLProject_BurnoutRiskPrediction.ipynb
```
 
Main libraries: `pandas`, `numpy`, `scipy`, `matplotlib`, `seaborn`, `scikit-learn`, `imbalanced-learn`.
 
## Notes
 
The dataset belongs to its original authors: please refer to the Kaggle page for its license and terms of use.