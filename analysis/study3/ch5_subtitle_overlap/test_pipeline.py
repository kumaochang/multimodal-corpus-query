"""Synthetic edge fixtures only; no research inputs or network."""
import math
import unittest
import numpy as np
from pipeline import normalize,dice,route_average,paired_valid,validate_segment,validate_pair

class Semantics(unittest.TestCase):
    def test_unicode_whitespace_and_case(self):
        self.assertEqual(normalize('  A\tB\n e\u0301 '),'A B é')
        self.assertNotEqual(normalize('A'),normalize('a'))

    def test_punctuation_single_alternative(self):
        self.assertEqual(normalize('甲，乙！ + C-D',True),'甲乙 + CD')
        self.assertEqual(normalize('甲，乙！'),'甲，乙！')

    def test_multiset_not_set(self):
        self.assertEqual(dice('aaaa','aa'),.5)
        self.assertEqual(dice('abc','abc'),1)
        self.assertEqual(dice('ab','cd'),0)

    def test_short_empty_not_zero(self):
        self.assertTrue(math.isnan(dice('a','ab')))
        self.assertTrue(math.isnan(dice('','ab')))
        self.assertEqual(dice('ab','bc'),0)

    def test_top3_and_insufficient_candidates(self):
        self.assertEqual(route_average(np.array([1.,0.,.5]),3),.5)
        self.assertEqual(route_average(np.array([1.,0.,np.nan]),2),.5)
        self.assertTrue(math.isnan(route_average(np.array([1.,np.nan,.5]),3)))

    def test_shared_denominator(self):
        self.assertEqual(paired_valid(np.array([[.2,.5],[.2,np.nan],[0.,0.]])).tolist(),[True,False,True])

    def test_id_and_bank_pairing(self):
        validate_pair([2,4],[2,4],[2,4],[2,4])
        for args in [([2,4],[2,4],[4,2],[2,4]),([2,4],[2,4],[2,4],[4,2])]:
            with self.assertRaises(ValueError):validate_pair(*args)

    def test_candidate_exclusion_uniqueness(self):
        for rows in [[2,2],[2,7]]:
            with self.assertRaises(ValueError):validate_segment(np.array(rows),np.array([.8,.5]),np.array([2,4]),np.array([2,4,7]))

    def test_equal_scores_follow_bank_not_row_number(self):
        bank=np.array([9,3,7]);scores=np.array([.8,.8,.2],dtype=np.float32)
        self.assertEqual(validate_segment(np.array([9,3,7]),scores,bank,bank),1)
        with self.assertRaises(ValueError):validate_segment(np.array([3,9,7]),scores,bank,bank)

if __name__=='__main__':unittest.main()
