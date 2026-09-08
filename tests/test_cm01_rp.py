import unittest
import numpy as np
from cm01_rp.metrics import bits, diagnose, distances, distributions


class CM01RPTests(unittest.TestCase):
    def logits(self, mapping):
        answers = (np.arange(16)[:, None] + np.asarray(mapping)[None, :]) % 16
        return np.tile(((bits()[answers] * 2 - 1) * 20.)[None, ...], (2,1,1,1))

    def test_joint_probabilities_and_distance_extremes(self):
        np.testing.assert_allclose(distributions(np.full((4,),.5)), np.full(16,1/16))
        p = np.zeros((2,16,16,4)); p[:,:,1,:] = 1
        joint,tv,js = distances(p)
        np.testing.assert_allclose(joint.sum(-1),1)
        self.assertEqual(tv[0,0,0,1],1.)
        self.assertEqual(js[0,0,0,1],1.)
        self.assertEqual(tv[0,0,0,2],0.)
        np.testing.assert_allclose(tv,tv.swapaxes(-1,-2))
        np.testing.assert_allclose(js,js.swapaxes(-1,-2))

    def test_sufficient_protocol_covers_all_classes(self):
        result,_ = diagnose(self.logits(range(16)),np.ones((2,16),int),[list(range(16))]*2)
        self.assertFalse(result['positive'])
        for ctx in result['contexts']:
            self.assertEqual(ctx['missing_v'],[])
            self.assertEqual(ctx['near_pairs'],[])
            self.assertEqual(ctx['best_correct_r_by_v'],[16]*16)

    def test_used_duplicate_plus_missing_class_is_positive(self):
        mapping = list(range(15)) + [0]
        result,_ = diagnose(self.logits(mapping),np.ones((2,16),int),[list(range(16))]*2)
        self.assertTrue(result['positive'])
        self.assertEqual(result['contexts'][0]['occupied_near_pairs'],[[0,15]])
        self.assertEqual(result['contexts'][0]['missing_v'],[15])

    def test_unused_duplicate_does_not_trigger_positive(self):
        usage=np.ones((2,16),int);usage[:,15]=0
        result,_=diagnose(self.logits(list(range(15))+[0]),usage,[list(range(16))]*2)
        self.assertFalse(result['positive'])
        self.assertEqual(result['contexts'][0]['near_pairs'],[[0,15]])

    def test_missing_unnecessary_class_does_not_trigger_positive(self):
        result,_=diagnose(self.logits(list(range(15))+[0]),np.ones((2,16),int),[list(range(15))]*2)
        self.assertFalse(result['positive'])

    def test_one_wrong_context_fails_full_coverage(self):
        logits=self.logits(range(16)); logits[0,0,0] = logits[0,0,1]
        result,_=diagnose(logits,np.ones((2,16),int),[list(range(16))]*2)
        self.assertEqual(result['contexts'][0]['best_correct_r_by_v'][0],15)
        self.assertIn(0,result['contexts'][0]['missing_v'])


if __name__=='__main__':
    unittest.main()
