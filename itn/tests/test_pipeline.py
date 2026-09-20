# -*- coding: utf-8 -*-
"""Unit tests: python -m unittest discover -s tests (from the itn/ directory)."""

import argparse
import csv
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from itn import LABELS  # noqa: E402
from itn.cli import cmd_validate  # noqa: E402
from itn.data import read_sentences, write_submission  # noqa: E402
from itn.features import sentence_features  # noqa: E402
from itn.lexicons import is_number, is_ordinal, tags  # noqa: E402
from itn.spans import from_spans, repair, score, to_spans  # noqa: E402


class TestSpans(unittest.TestCase):
    def test_labels_are_the_17_contest_labels(self):
        self.assertEqual(len(LABELS), 17)
        self.assertIn("O", LABELS)
        self.assertIn("B-WHITELIST", LABELS)

    def test_basic_decode(self):
        self.assertEqual(to_spans(["O", "B-TIME", "I-TIME", "O"]), [(1, 2, "TIME")])

    def test_orphan_i_starts_a_span(self):
        # The statement: an I- that does not continue a span of the same class
        # reads as the start of a new span.
        self.assertEqual(to_spans(["I-TIME", "I-TIME"]), [(0, 1, "TIME")])
        self.assertEqual(
            to_spans(["B-TIME", "I-CARDINAL"]), [(0, 0, "TIME"), (1, 1, "CARDINAL")]
        )

    def test_adjacent_same_class_spans_are_separate(self):
        self.assertEqual(
            to_spans(["B-CARDINAL", "B-CARDINAL"]),
            [(0, 0, "CARDINAL"), (1, 1, "CARDINAL")],
        )

    def test_roundtrip(self):
        labels = ["O", "B-MEASURE", "I-MEASURE", "O", "B-DATE"]
        self.assertEqual(from_spans(to_spans(labels), len(labels)), labels)

    def test_repair_is_decode_preserving(self):
        labels = ["I-TIME", "I-TIME", "I-DATE"]
        self.assertEqual(to_spans(repair(labels)), to_spans(labels))

    def test_exact_boundaries_only(self):
        gold = [["B-MEASURE", "I-MEASURE", "I-MEASURE"]]
        partial = [["B-MEASURE", "I-MEASURE", "O"]]
        self.assertEqual(score(gold, partial)["tp"], 0)
        self.assertEqual(score(gold, partial)["f1"], 0.0)

    def test_perfect_and_empty(self):
        gold = [["O", "B-TIME", "I-TIME"]]
        self.assertEqual(score(gold, gold)["score"], 100.00)
        all_o = [["O", "O", "O"]]
        self.assertEqual(score(gold, all_o)["score"], 0.00)

    def test_f1_arithmetic(self):
        gold = [["B-CARDINAL", "O", "B-DATE"]]
        pred = [["B-CARDINAL", "O", "B-TIME"]]
        result = score(gold, pred)
        self.assertEqual((result["tp"], result["fp"], result["fn"]), (1, 1, 1))
        self.assertAlmostEqual(result["f1"], 0.5)


class TestData(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()

    def _write(self, name, rows, header):
        path = os.path.join(self.dir, name)
        with open(path, "w", encoding="utf-8", newline="") as fh:
            writer = csv.writer(fh, lineterminator="\n")
            writer.writerow(header)
            writer.writerows(rows)
        return path

    def test_grouping_and_out_of_order_token_ids(self):
        path = self._write(
            "train.csv",
            [
                ["s1", 1, "три", "I-TIME"],
                ["s1", 0, "в", "O"],
                ["s2", 0, "бес", "B-CARDINAL"],
            ],
            ["sent_id", "token_id", "token", "label"],
        )
        sentences = read_sentences(path)
        self.assertEqual(len(sentences), 2)
        self.assertEqual(sentences[0].tokens, ["в", "три"])
        self.assertEqual(sentences[0].labels, ["O", "I-TIME"])

    def test_unlabelled_file_has_no_labels(self):
        path = self._write(
            "test.csv", [["s1", 0, "бес"]], ["sent_id", "token_id", "token"]
        )
        self.assertIsNone(read_sentences(path)[0].labels)

    def test_submission_is_one_row_per_token(self):
        path = self._write(
            "test.csv",
            [["s1", 0, "в"], ["s1", 1, "три"]],
            ["sent_id", "token_id", "token"],
        )
        sentences = read_sentences(path)
        out = os.path.join(self.dir, "solution.csv")
        rows = write_submission(out, sentences, [["O", "B-TIME"]])
        self.assertEqual(rows, 2)
        with open(out, encoding="utf-8") as fh:
            lines = fh.read().strip().split("\n")
        self.assertEqual(lines[0], "sent_id,token_id,label")
        self.assertEqual(lines[2], "s1,1,B-TIME")

    def test_reads_gzipped_csv(self):
        import gzip

        path = os.path.join(self.dir, "train.csv.gz")
        with gzip.open(path, "wt", encoding="utf-8", newline="") as fh:
            writer = csv.writer(fh, lineterminator="\n")
            writer.writerow(["sent_id", "token_id", "token", "label"])
            writer.writerow(["s1", 0, "бес", "B-CARDINAL"])
        sentences = read_sentences(path)
        self.assertEqual(sentences[0].tokens, ["бес"])
        self.assertEqual(sentences[0].labels, ["B-CARDINAL"])

    def test_submission_rejects_length_mismatch(self):
        path = self._write("t.csv", [["s1", 0, "в"]], ["sent_id", "token_id", "token"])
        sentences = read_sentences(path)
        with self.assertRaises(ValueError):
            write_submission(os.path.join(self.dir, "x.csv"), sentences, [["O", "O"]])


class TestValidate(unittest.TestCase):
    """The judge rejects a submission outright for a missing row, a duplicate row
    or an unknown label, so the validator is the last check before submitting."""

    def setUp(self):
        self.dir = tempfile.mkdtemp()
        # sample_submission.csv has sent_id/token_id/label, so it works as the
        # index to validate against just like test.csv does.
        self.index = os.path.join(self.dir, "sample_submission.csv")
        with open(self.index, "w", encoding="utf-8", newline="") as fh:
            writer = csv.writer(fh, lineterminator="\n")
            writer.writerow(["sent_id", "token_id", "label"])
            for sid in ("s000000", "s000123"):
                for tid in range(3):
                    writer.writerow([sid, tid, "O"])

    def _validate(self, rows):
        pred = os.path.join(self.dir, "solution.csv")
        with open(pred, "w", encoding="utf-8", newline="") as fh:
            writer = csv.writer(fh, lineterminator="\n")
            writer.writerow(["sent_id", "token_id", "label"])
            writer.writerows(rows)
        return cmd_validate(argparse.Namespace(test=self.index, pred=pred))

    def _all_rows(self):
        return [[sid, tid, "O"] for sid in ("s000000", "s000123") for tid in range(3)]

    def test_accepts_a_complete_submission(self):
        self.assertEqual(self._validate(self._all_rows()), 0)

    def test_rejects_missing_row(self):
        self.assertEqual(self._validate(self._all_rows()[:-1]), 1)

    def test_rejects_duplicate_row(self):
        rows = self._all_rows()
        self.assertEqual(self._validate(rows + [rows[0]]), 1)

    def test_rejects_unknown_label(self):
        rows = self._all_rows()
        rows[0][2] = "B-NUMBER"
        self.assertEqual(self._validate(rows), 1)

    def test_rejects_row_absent_from_the_index(self):
        self.assertEqual(self._validate(self._all_rows() + [["s999999", 0, "O"]]), 1)

    def test_row_order_does_not_matter(self):
        rows = self._all_rows()
        rows.reverse()
        self.assertEqual(self._validate(rows), 0)


class TestFeatures(unittest.TestCase):
    def test_one_feature_list_per_token(self):
        tokens = "выйдем через три минут".split()
        feats = sentence_features(tokens)
        self.assertEqual(len(feats), 4)
        self.assertTrue(all(isinstance(f, str) for f in feats[0]))

    def test_context_distinguishes_identical_words(self):
        # "три" in three different contexts must not get identical features,
        # otherwise the three classes in the statement are indistinguishable.
        a = set(sentence_features("говорил три раза".split())[1])
        b = set(sentence_features("в три тридцать".split())[1])
        c = set(sentence_features("через три минут".split())[1])
        self.assertNotEqual(a, b)
        self.assertNotEqual(b, c)
        self.assertNotEqual(a, c)

    def test_single_token_sentence(self):
        self.assertEqual(len(sentence_features(["бес"])), 1)

    def test_empty_sentence(self):
        self.assertEqual(sentence_features([]), [])


class TestLexicons(unittest.TestCase):
    def test_numerals_both_languages(self):
        for token in ("три", "двадцать", "бес", "жиырма", "мың"):
            self.assertTrue(is_number(token), token)
        for token in ("дом", "жұмыс", "көрінеді"):
            self.assertFalse(is_number(token), token)

    def test_kazakh_case_suffix_is_stripped(self):
        self.assertTrue(is_number("мыңға"))

    def test_ordinals(self):
        for token in ("первого", "бірінші", "двадцатый"):
            self.assertTrue(is_ordinal(token), token)

    def test_email_and_unit_tags(self):
        self.assertIn("MAIL", tags("собачка"))
        self.assertIn("UNIT", tags("процентов"))
        self.assertIn("CUR", tags("теңге"))


if __name__ == "__main__":
    unittest.main()
