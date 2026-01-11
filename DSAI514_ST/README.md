Project: Statistical Inference on Personal Health Data (DSAI 514)
Student: Yiğit Ateş - 2025776009
Date: December 2025

-------------------------------------------------------------------------
1. PROJECT OVERVIEW
-------------------------------------------------------------------------
This project analyzes the relationship between daily physical activity 
(steps taken) and REM sleep duration using my personal data collected from
Samsung Health.

The analysis includes:
- Data cleaning and date alignment (merging steps from Day T with sleep on Day T+1)
- EDA
- Parameter Estimation (MLE, MoM, Bayesian)
- Goodness-of-Fit testing (KS Test)
- Hypothesis Testing (One sample t-test against 1.5h benchmark)
- Linear Regression Analysis

-------------------------------------------------------------------------
2. FILE STRUCTURE
-------------------------------------------------------------------------
- inference.ipynb    : Main Python script containing all analysis and plotting logic.
- data/              : Folder containing raw datasets.
  - sleep12.csv      : Raw sleep logs from Samsung Health.
  - steps12.csv      : Raw step count logs from Samsung Health.
- report.pdf         : The final project report.
- README.txt         : This file.

-------------------------------------------------------------------------
3. REQUIREMENTS & DEPENDENCIES
-------------------------------------------------------------------------
The code requires Python 3.x and the following libraries:
- pandas
- numpy
- matplotlib
- seaborn
- statsmodels
- scipy

-------------------------------------------------------------------------
4. HOW TO RUN
-------------------------------------------------------------------------
1. Ensure the 'data' folder contains 'sleep12.csv' and 'steps12.csv'.
2. Press run all in the notebook:
3. The notebook will:
   - Print statistical summaries and test results to the console.
   - Generate and save the following figures in the current directory