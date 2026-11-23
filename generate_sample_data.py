"""Create a SYNTHETIC stand-in for student-por.csv (for testing only).

The real dataset is the UCI 'Student Performance' data (student-por.csv):
https://archive.ics.uci.edu/dataset/320/student+performance
Download it and place it in  data/student-por.csv  to use real data.
"""
import numpy as np
import pandas as pd

from src.config import DATA_FILE

# Create a random number generator with a fixed seed.
# Set the number of student records to 649.
rng = np.random.default_rng(42)
n = 649

# Generate random student ages between 15 and 20.
# Generate the gender of each student (Use F for female and M for male.)
# Generate the weekly study time for each student(Use values from 1 to 4 to represent different study times.)
age = rng.choice([15, 16, 17, 18, 19, 20], n, p=[.17, .27, .26, .24, .05, .01])
sex = rng.choice(["F", "M"], n, p=[.59, .41])
studytime = rng.choice([1, 2, 3, 4], n, p=[.27, .50, .16, .07])
# Generate the number of previous school failures.(Use values from 0 to 3 for each student.)
# Generate random absence records for students.(Keep the absence values between 0 and 32)
failures = rng.choice([0, 1, 2, 3], n, p=[.85, .09, .03, .03])
absences = np.clip(rng.poisson(3.5, n), 0, 32)
Medu = rng.choice([0, 1, 2, 3, 4], n, p=[.01, .18, .26, .21, .34])# Generate the education levels of students' mothers.
Fedu = rng.choice([0, 1, 2, 3, 4], n, p=[.01, .22, .30, .21, .26])# Generate the education levels of students' fathers.
# Generate information about family educational support.
# Generate information about school educational support.
# Generate information about internet access.
famsup = rng.choice(["yes", "no"], n, p=[.61, .39])
schoolsup = rng.choice(["yes", "no"], n, p=[.10, .90])
internet = rng.choice(["yes", "no"], n, p=[.77, .23])
# Calculate a simulated ability score using student information.
# Consider study time, failures, absences and parents' education.
ability = (rng.normal(11.5, 2.8, n) + 0.6 * (studytime - 2) - 1.4 * failures
           - 0.08 * absences + 0.25 * ((Medu + Fedu) / 2 - 2))

# Generate the first-period grades for all students.
# Generate the second-period grades using the first grades.
# Generate the third-period grades using the second grades.
G1 = np.clip(np.round(ability + rng.normal(0, 1.2, n)), 0, 19).astype(int)
G2 = np.clip(np.round(0.9 * G1 + 0.1 * ability + rng.normal(0, 1.0, n)), 0, 19).astype(int)
G3 = np.clip(np.round(0.95 * G2 + 0.05 * ability + rng.normal(0, 1.0, n)), 0, 20).astype(int)

# Combine all the generated student information into one table.
# Store the table in a DataFrame called df.
df = pd.DataFrame(dict(age=age, sex=sex, studytime=studytime, failures=failures,
                       absences=absences, Medu=Medu, Fedu=Fedu, famsup=famsup,
                       schoolsup=schoolsup, internet=internet, G1=G1, G2=G2, G3=G3))

# Create the folder where the dataset will be stored if needed.
# Save the student data as a CSV file separated by semicolons.
DATA_FILE.parent.mkdir(exist_ok=True)
df.to_csv(DATA_FILE, sep=";", index=False)
print(f"Synthetic dataset written to {DATA_FILE} ({len(df)} rows)")
