
# Overview
This repository contains source code of KUnit, benchmark used for empirical evaluation, datasets used for user study, task description, user study participant's responses and results of RQ2.

```
.
├── Benchmark_Empirical_Evaluation         # Contains 50 programs obtained from Stack Overflow and GitHub
├── Datasets_UserStudy                     # Contains 5 datasets used for user study
├── KUnit                                  # Source code of KUnit
    ├── KUnit_data                         # Contains source code for data preparation stage
    ├── KUnit_model                        # Contains source code for model design stage
    ├── Instructions.txt                   # Intructions for using KUnit 
    └── requirements.txt                   # Dependency and Python virutal environment information
├── UserStudy_Tasks                        # Details of tasks provided to the participants
├── Participants_Response.pdf              # Contains qualitative response from user study participants
└── RQ2_Results.xlsx                       # Contains results of comparing KUnit with DeepDiagnosis
```
# Benchmark for Empirical Evaluation
The 50 buggy programs used for empicial evaluation are stored under the directory [Benchmark_Empirical_Evaluation](Benchmark_Empirical_Evaluation). Each buggy program is stored in folder named after the StackOverflow post handle and GitHub repository name corresponding to it. Each folder also contains the data and model stages separted in different files used for evaluating KUnit.

# Datasets used for User Study
The directory [Datasets_UserStudy](Datasets_UserStudy) contains the 5 datasets used in user study.

# KUnit
To run KUnit, one needs to create a virtual environment. The instructions for creating virtual environment and how to use KUnit for mock testing are provided in [Instructions.txt](Instructions.txt). 


# User Study Task Description
The detail description of the task for each stage (data preparation and model design) provided to the participants during user study are in directory [UserStudy_Tasks](UserStudy_Tasks)

# Participants Response
The qualitative response highlighting the advantages and disadvantages obtained from 36 participants in a post-study survey are provided in [Participants_Response.pdf](Participants_Response.pdf)

# Results
The results of comparing KUnit with the state-of-the-art approach DeepDiagnosis on 50 programs in our benchmark are provided in [RQ2_Results.xlsx](RQ2_Results.xlsx))
