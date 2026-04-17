import unittest

import numpy as np
import torch

from evaluation_framework.metrics.open_set.oscr_curve import OSCRCurve


class TestOSCR(unittest.TestCase):

    def test_OSCR_thresholding(self):
        logits = [[9.8, 0.1, 0.1],
                  [1, 8, 1],
                  [1, 2, 7],
                  [3, 5, 2],
                  [1 / 3, 1 / 3 + 0.01, 1 / 3],

                  [8, 1, 1],
                  [7, 2, 1],
                  [6, 3, 1],
                  [5, 4, 1],
                  [1 / 3 + 0.01, 1 / 3, 1 / 3]
                  ]

        targets = [0, 1, 0, 1, 1, 3, 3, 3, 3, 3]

        logits = torch.log(torch.Tensor(logits))
        targets = torch.Tensor(targets)

        oscr = OSCRCurve(precision=1, unknown_label=3).compute(logits=logits, targets=targets)

        real_oscr = torch.tensor([[0, 0, 0.5],
                                  [0.0, 0.2, 0.6],
                                  [0.2, 0.2, 0.5],
                                  [0.4, 0.4, 0.5],
                                  [0.6, 0.4, 0.4],
                                  [0.8, 0.4, 0.3],
                                  [0.8, 0.6, 0.4],
                                  [1, 0.8, 0.4],
                                  [1, 0.8, 0.4],
                                  [1, 0.8, 0.4],
                                  [1, 0.8, 0.4]
                                  ])

        self.assertTrue(torch.allclose(oscr, real_oscr, atol=1e-6))

        self.assertEqual(OSCRCurve(precision=3, unknown_label=3).get_name(), 'OSCR Curve')

        oscr = OSCRCurve(precision=0, unknown_label=3).compute(logits=logits, targets=targets)

        real_oscr_short = torch.tensor([[0, 0, 0.5],
                                        [1, 0.8, 0.4]
                                        ])

        self.assertTrue(torch.allclose(oscr, real_oscr_short, atol=1e-6))

    def test_OSCR_empty_positives(self):
        logits = [[9.8, 0.1, 0.1],
                  [1, 8, 1],
                  [1, 2, 7],
                  [3, 5, 2],
                  [1 / 3, 1 / 3 + 0.01, 1 / 3],

                  [8, 1, 1],
                  [7, 2, 1],
                  [6, 3, 1],
                  [5, 4, 1],
                  [1 / 3 + 0.01, 1 / 3, 1 / 3]
                  ]

        targets = [3, 3, 3, 3, 3, 3, 3, 3, 3, 3]

        logits = torch.log(torch.Tensor(logits))
        targets = torch.Tensor(targets)

        oscr = OSCRCurve(precision=1, unknown_label=3).compute(logits=logits, targets=targets)

        real_oscr = torch.tensor([0.0, 0.0, 0.0])

        self.assertTrue(torch.allclose(oscr, real_oscr, atol=1e-6))

    def test_OSCR_empty_negatives(self):
        logits = [[9.8, 0.1, 0.1],
                  [1, 8, 1],
                  [1, 2, 7],
                  [3, 5, 2],
                  [1 / 3, 1 / 3 + 0.01, 1 / 3],

                  [8, 1, 1],
                  [7, 2, 1],
                  [6, 3, 1],
                  [5, 4, 1],
                  [1 / 3 + 0.01, 1 / 3, 1 / 3]
                  ]

        targets = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

        logits = torch.Tensor(logits).apply_(lambda x: np.log(x))
        targets = torch.Tensor(targets)

        oscr = OSCRCurve(precision=1, unknown_label=3).compute(logits=logits, targets=targets)

        real_oscr = torch.tensor([0.0, 0.0, 0.0])

        self.assertTrue(torch.allclose(oscr, real_oscr, atol=1e-6))

    def test_OSCR_garbage(self):
        logits = [[90 + 0.01, 10, 0],
                  [20, 80, 0],
                  [30 + 0.01, 20, 50],

                  [80, 10, 10],
                  [20, 70 + 0.01, 10],
                  [40, 30, 30],

                  [30, 40, 30],
                  [30, 30, 40],
                  [10, 10, 80],
                  [80, 10, 10]

                  ]

        targets = [0, 0, 0, 1, 1, 1, 2, 2, 2, 2]

        logits = torch.log(torch.Tensor(logits))
        targets = torch.Tensor(targets)

        oscr = OSCRCurve(precision=1, unknown_label=2).compute(logits=logits, targets=targets)

        real_oscr = torch.tensor([[0, 0, 0.4],
                                  [0.0, 1 / 6, 0.5],
                                  [0.25, 1 / 6, 0.4],
                                  [0.25, 2 / 6, 0.5],
                                  [0.25, 2 / 6, 0.5],
                                  [0.25, 2 / 6, 0.5],
                                  [0.50, 2 / 6, 0.4],
                                  [0.75, 3 / 6, 0.4],
                                  [0.75, 3 / 6, 0.4],
                                  [1.0, 3 / 6, 0.3],
                                  [1.0, 3 / 6, 0.3]
                                  ])

        self.assertTrue(torch.allclose(oscr, real_oscr, atol=1e-3))


