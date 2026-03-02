Medical Appointment No-Show Prediction
📖 Project Overview

This project aims to predict whether a patient will miss a scheduled medical appointment (“No-show”) using demographic, medical, and appointment-related features.

Missed medical appointments lead to resource wastage, increased healthcare costs, and reduced efficiency. By identifying patients likely to miss appointments, healthcare providers can take preventive actions such as reminders or follow-ups.

🎯 Problem Statement

The objective of this project is to build a machine learning model that predicts whether a patient will not show up for a scheduled appointment based on historical appointment data.

The task is formulated as a binary classification problem:

0 → Showed up

1 → No-show

The goal is to develop a balanced model that:

Achieves accuracy ≥ 75%

Improves recall and F1-score for the minority class (No-show)

Handles class imbalance effectively

📂 Dataset Description

Source: Kaggle – Medical Appointment No Shows

Total Records: 110,527

Total Features: 14 original features

Important Features:
Feature	Description
Gender	Patient gender
Age	Patient age
Scholarship	Government financial assistance
Hipertension	Hypertension condition
Diabetes	Diabetes condition
Alcoholism	Alcohol dependency
Handicap	Disability level
SMS_received	Whether reminder SMS was sent
ScheduledDay	Appointment booking date
AppointmentDay	Actual appointment date
No-show	Target variable
🧹 Data Preprocessing

The following preprocessing steps were performed:

Removed invalid ages (< 0 or > 100)

Converted No-show to binary (0/1)

Converted date columns to datetime format

Created new feature:

AwaitingTime = Days between scheduling and appointment

Created historical feature:

Num_App_Missed = Number of previous missed appointments

One-hot encoded categorical variables

Dropped irrelevant columns (PatientId, AppointmentID, Neighbourhood)

📊 Exploratory Data Analysis (EDA)

Key insights from EDA:

Approximately 20% of patients do not show up

Longer waiting time increases probability of no-show

Younger age groups show higher no-show rate

Previous missed appointments strongly correlate with future no-shows

SMS reminders do not significantly reduce no-show rate

🤖 Models Implemented
1️⃣ Logistic Regression (Balanced)

Used class_weight='balanced'

Applied feature scaling (StandardScaler)

Adjusted decision threshold

Evaluated using Accuracy, Precision, Recall, F1-score, ROC-AUC

2️⃣ Random Forest (Balanced)

Used class balancing

Tuned max_depth and min_samples_split

No scaling required

Provided improved recall while maintaining accuracy ≥ 75%

📈 Model Evaluation Metrics

Since the dataset is imbalanced (~80% show-ups), relying only on accuracy is misleading.

The following metrics were used:

Accuracy

Precision

Recall

F1-Score

ROC-AUC

Final Balanced Model Performance (Approximate)
Metric	Logistic	Random Forest
Accuracy	~0.75–0.78	~0.76–0.81
Recall	~0.25–0.35	~0.25–0.35
F1 Score	~0.30+	~0.30–0.40

The Random Forest model achieved the best balance between accuracy and recall.

⚖️ Class Imbalance Handling

The dataset contains approximately:

80% Show

20% No-show

To address imbalance:

Used class_weight='balanced'

Adjusted probability threshold

Evaluated using F1-score instead of accuracy alone

📌 Key Features Influencing Prediction

Feature importance analysis revealed:

AwaitingTime

Age

Number of previous missed appointments

These features were the strongest predictors of no-show behavior.
