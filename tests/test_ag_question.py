from unittest import TestCase, main

import numpy as np
import pandas as pd
import numpy.testing as npt
import pandas.util.testing as pdt

from americangut.question.ag_question import AgQuestion


class AgQuestionTest(TestCase):
    def setUp(self):
        self.map_ = pd.DataFrame([[1, 2], [3, 4]],
                                 columns=['A', 'B'],
                                 index=['X', 'Y'])

        self.name = 'TEST_COLUMN'
        self.description = ('"Say something, anything."\n'
                            '"Testing... 1, 2, 3"\n'
                            '"Anything but that!"')
        self.dtype = str

        self.fun = lambda x: 'foo'

        self.ag_question = AgQuestion(
            name=self.name,
            description=self.description,
            dtype=self.dtype
            )

    def test_init(self):
        test = AgQuestion(self.name, self.description, self.dtype)

        self.assertEqual(test.name, self.name)
        self.assertEqual(test.description, self.description)
        self.assertEqual(test.dtype, self.dtype)
        self.assertEqual(test.clean_name, 'Test Column')
        self.assertEqual(test.free_response, False)
        self.assertEqual(test.mimarks, False)
        self.assertEqual(test.ontology, None)
        self.assertEqual(test.remap_, None)

    def test_init_kwargs(self):
        test = AgQuestion(self.name, self.description, self.dtype,
                          clean_name='This is a test.',
                          free_response=True,
                          mimarks=True,
                          ontology='GAZ',
                          remap=self.fun,
                          )

        self.assertEqual(test.name, self.name)
        self.assertEqual(test.description, self.description)
        self.assertEqual(test.dtype, self.dtype)
        self.assertEqual(test.clean_name, 'This is a test.')
        self.assertEqual(test.free_response, True)
        self.assertEqual(test.mimarks, True)
        self.assertEqual(test.ontology, 'GAZ')
        self.assertEqual(test.remap_(1), 'foo')

    def test_init_name_error(self):
        with self.assertRaises(TypeError):
            AgQuestion(1, self.description, self.dtype)

    def test_init_description_error(self):
        with self.assertRaises(TypeError):
            AgQuestion(self.name, 3, self.dtype)

    def test_init_class_error(self):
        with self.assertRaises(TypeError):
            AgQuestion(self.name, self.description, 'bool')

    def test_init_clean_name_error(self):
        with self.assertRaises(TypeError):
            AgQuestion(self.name, self.description, self.dtype, ['Cats'])

    def test_check_map(self):
        self.ag_question.name = 'A'
        self.ag_question.check_map(self.map_)

    def test_check_map_error(self):
        with self.assertRaises(ValueError):
            self.ag_question.check_map(self.map_)

if __name__ == '__main__':
    main()
