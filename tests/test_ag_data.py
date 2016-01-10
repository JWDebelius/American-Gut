import pandas as pd
import numpy as np

import numpy.testing as npt
import pandas.util.testing as pdt

from testfixtures import TempDirectory
from unittest import TestCase, main

from americangut.ag_data import *


class AgDataTest(TestCase):
    def setUp(self):
        d = TempDirectory()

    def test__init__(self):
        bodysite = 'fecal'
        trim = '100nt'
        depth = '1k'

        test = AgData(
            bodysite=bodysite,
            trim=trim,
            depth=depth
            )
        self.assertEqual(test.data_set['bodysite'], bodysite)
        self.assertEqual(test.data_set['sequence_trim'], trim)
        self.assertEqual(test.data_set['rarefaction_depth'], depth)
        self.assertEqual(test.data_set['samples_per_participants'],
                         'all_samples')
        self.assertEqual(test.data_set['participant_set'], 'all')


if __name__ == '__main__':
    main()
