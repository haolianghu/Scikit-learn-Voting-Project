import unittest
from pathlib import Path

from voting_baseline import analyze, load_data


ROOT = Path(__file__).resolve().parents[1]


class VotingTests(unittest.TestCase):
    def test_existing_county_data(self):
        frame = load_data(ROOT / "clinton1.dat")
        result = analyze(frame)
        self.assertEqual(len(frame), 2704)
        self.assertLessEqual(len(result["regression"]["selected_features_by_training_r2"]), 9)
        self.assertGreaterEqual(result["classification"]["best_neighbors"], 1)
        self.assertTrue(0 <= result["classification"]["test_accuracy"] <= 1)

    def test_invalid_file_schema(self):
        import tempfile
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.dat"
            path.write_text("one two\n")
            with self.assertRaises(ValueError):
                load_data(path)


if __name__ == "__main__":
    unittest.main()
