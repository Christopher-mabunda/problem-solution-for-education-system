import unittest

import pandas as pd

# Create a test class to check if Pandas functions work correctly.
# Use Python's unittest tools to run the tests and check the results.
class TestPandas(unittest.TestCase):    
# Test whether Pandas can successfully read a CSV file.
# Check whether the dataset contains exactly 100 rows.
    def test_read_csv(self):
        data = pd.read_csv("/input/tests/data/train.csv")

# Check if the CSV file has the correct number of rows.
# The test passes if there are 100 rows.
        self.assertEqual(100, len(data.index))

    
# Test whether Pandas can open and read a Feather file. 
# Check if the file contains exactly 10 data values.    
    def test_read_feather(self):
        data = pd.read_feather("/input/tests/data/feather-0_3_1.feather")

# Check if the total number of values is equal to 10.
# The test passes if the Feather file contains 10 values.        
        self.assertEqual(10, data.size)
