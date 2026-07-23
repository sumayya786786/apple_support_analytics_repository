from __future__ import annotations

import json
import unittest
from pathlib import Path

import pandas as pd


class PipelineOutputTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = Path(__file__).resolve().parents[1]
        cls.tables = cls.root / "outputs" / "tables"
        cls.figures = cls.root / "outputs" / "figures"

    def test_core_audit(self):
        audit = json.loads((self.tables / "data_quality_audit.json").read_text())
        self.assertEqual(audit["retained"], 80)
        self.assertEqual(audit["turnaround_applicable"], 68)
        self.assertEqual(audit["resolution_model_n"], 79)
        self.assertEqual(audit["primary_dissatisfied_model_eligible"], 22)
        self.assertTrue(audit["checksum_matches_expected"])

    def test_primary_ols_reconciliation(self):
        coefficients = pd.read_csv(self.tables / "ols_hc3_coefficients.csv").set_index("predictor")
        self.assertAlmostEqual(coefficients.loc["communication_quality", "B"], 0.639802576, places=6)
        self.assertAlmostEqual(coefficients.loc["waiting_acceptability", "B"], 0.416282669, places=6)
        self.assertAlmostEqual(coefficients.loc["resolution_unresolved", "B"], -2.188384275, places=6)
        diagnostics = json.loads((self.tables / "ols_diagnostics.json").read_text())
        self.assertAlmostEqual(diagnostics["adjusted_r_squared"], 0.834131049, places=6)

    def test_reported_text_counts(self):
        counts = pd.read_csv(self.tables / "text_theme_counts_all_comments.csv")
        lookup = counts.set_index(["source_question", "theme"])["count"]
        self.assertEqual(lookup.loc[("negative_comment", "communication_updates")], 23)
        self.assertEqual(lookup.loc[("negative_comment", "explanation_clarity")], 20)
        self.assertEqual(lookup.loc[("improvement_comment", "process_ease_hand_offs")], 28)
        self.assertEqual(lookup.loc[("improvement_comment", "turnaround")], 18)

    def test_expected_figures_exist(self):
        names = {
            "figure_2_satisfaction_distribution.png",
            "figure_3_service_quality_profile.png",
            "figure_4_predictor_correlation_heatmap.png",
            "figure_5_primary_ols_coefficients.png",
            "figure_6_classification_performance.png",
            "figure_7_classification_calibration.png",
            "figure_8_biggest_factor.png",
        }
        self.assertTrue(names.issubset({path.name for path in self.figures.glob("*.png")}))


if __name__ == "__main__":
    unittest.main()
