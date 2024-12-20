import unittest
from unittest.mock import patch
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification

# Assuming the CellPhenoX class is defined in a file called 'cellpheno.py'
from pyCellPhenoX.CellPhenoX import CellPhenoX

class TestCellPhenoX(unittest.TestCase):

    def setUp(self):
        # Create a simple synthetic dataset for testing
        X, y = make_classification(n_samples=100, n_features=10, n_classes=2, random_state=42)
        self.X = pd.DataFrame(X)
        self.y = pd.Series(y)
        self.CV_repeats = 3
        self.outer_num_splits = 2
        self.inner_num_splits = 2

        # Initialize the class
        self.cellpheno = CellPhenoX(
            X=self.X,
            y=self.y,
            CV_repeats=self.CV_repeats,
            outer_num_splits=self.outer_num_splits,
            inner_num_splits=self.inner_num_splits,
        )

    def test_initialization(self):
        """Test that the class initializes without errors."""
        self.assertEqual(self.cellpheno.X.shape, (100, 10))
        self.assertEqual(len(self.cellpheno.y), 100)
        self.assertEqual(self.cellpheno.CV_repeats, 3)
        self.assertEqual(self.cellpheno.outer_num_splits, 2)
        self.assertEqual(self.cellpheno.inner_num_splits, 2)

    def test_split_data(self):
        """Test that data splitting works without errors."""
        train_outer_ix, test_outer_ix = np.arange(0, 50), np.arange(50, 100)
        result = self.cellpheno.split_data(train_outer_ix, test_outer_ix)

        self.assertEqual(len(result), 8)
        self.assertEqual(result[0].shape[0], 50)
        self.assertEqual(result[1].shape[0], 50)

    def test_attributes_existence(self):
        """Test that important attributes exist."""
        self.assertTrue(hasattr(self.cellpheno, 'best_model'))
        self.assertTrue(hasattr(self.cellpheno, 'best_score'))
        self.assertTrue(hasattr(self.cellpheno, 'shap_values_per_cv'))

