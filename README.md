# Introduction
The diagnosis of heart disease in most cases depends on a complex combination of clinical and pathological data. Because of this complexity, there exists a significant amount of interest among clinical professionals and researchers regarding the efficient and accurate prediction of heart disease. In this paper, we develop a heart disease prediction system that can assist medical professionals in predicting heart disease status based on clinical data of patients. Our approaches include three steps. Firstly, we select important features based on feature selection technique and according to clinical studies. Secondly, we develop artificial neural network algorithms for classifying heart disease based on these extracted clinical features. The accuracy of prediction is near 87%. Finally, we develop a user-friendly heart disease prediction system (HDPS). The HDPS system will consist of multiple features, including input clinical data section and prediction performance display section with execution time, accuracy and predicted result. Our approaches are effective in predicting the heart disease of patients. The HDPS system developed in this study is a novel approach that can be used in the classification of heart disease.
  
# Dataset
The dataset used in this project is the Cleveland Heart Disease dataset taken from the UCI repository.
  
The dataset contains 76 attributes, but all published experiments refer to using a subset of 14 of them. In particular, the Cleveland database is the only one that has been used by ML researchers to this date. The “target” field refers to the presence of heart disease in the patient. It is integer value from 0 (no presence) to 4 (presence).
  
The names and personal profiles (social security number) of the patients were removed from the database in order to maintain privacy of the profiles.
  
## Attribute Information
According to clinical trials and studies only 14 factors plays a major role in detecting heart disease, which are:
  
  - Age: Displays the age of an individual
  - Sex: Displays the gender of an individual using the following format:
    - 1 = Male
    - 0 = Female
  - Chest Pain Type: Displays the type of chest pain experienced by the individual using the following format:
    - 1 = Typical angina
    - 2 = Atypical angina
    - 3 = Non-anginal pain
    - 4 = Asymtotic
  - Resting Blood Pressure: Displays the resting blood pressure value of an individual in mmHg (unit)
  - Serum Cholesterol: Displays the serum cholesterol in mg/dl (unit)
  - Fasting Blood Sugar: Compares the fasting blood sugar value of an individual with 120mg/dl.
    - 1 = If fasting blood sugar > 120mg/dl
    - 0 = If fasting blood sugar < 120mg/dl
  - Resting ECG: Displays resting electrocardiographic results:
    - 0 = Normal
    - 1 = Having ST-T wave abnormality
    - 2 = Left ventricular hypertrophy
  - Maximum Heart Rate Achieved: Displays the max heart rate achieved by an individual
  - Exercise Induced Angina:
    - 1 = Yes
    - 0 = No
  - ST depression induced by exercise relative to rest: Displays the value which is an integer or float.
  - Peak exercise ST segment:
    - 1 = Upsloping
    - 2 = Flat
    - 3 = Downsloping
  - Number of major vessels colored by fluoroscopy: Displays the value as integer or float.
  - Thalassemia: Displays the thalassemia:
    - 3 = Normal
    - 6 = Fixed defect
    - 7 = Reversible defect
  - Diagnosis of heart disease: Displays whether the individual is suffering from heart disease or not:
    - 0 = Absence
    - 1 = Present
