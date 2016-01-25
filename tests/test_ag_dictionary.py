from unittest import TestCase, main

import numpy as np
import pandas as pd

from americangut.ag_dictionary import (ag_dictionary,
                                       _remap_abx,
                                       _remap_bowel_frequency,
                                       _remap_fecal_quality,
                                       _remap_contraceptive,
                                       _remap_gluten,
                                       _remap_last_travel,
                                       _remap_sleep,
                                       _remap_diet,
                                       _remap_fed_as_infant,
                                       _remap_weight,
                                      )

from americangut.question.ag_categorical import AgCategorical


class AgDictionary(TestCase):

    def test_remap_abx(self):
        self.assertEqual(_remap_abx('I have not taken antibiotics in the '
                                    'past year.'), 'More than a year')
        self.assertEqual(_remap_abx('I have not taken antibiotics in the '
                                    'past year'), 'More than a year')
        self.assertEqual(_remap_abx('foo'), 'foo')

    def test_remap_bowel_frequency(self):
        self.assertEqual(_remap_bowel_frequency('Four'), 'Four or more')
        self.assertEqual(_remap_bowel_frequency('Five or more'), 
                         'Four or more')
        self.assertEqual(_remap_bowel_frequency('foo'), 'foo')

    def test_remap_fecal_quality(self):
        self.assertEqual(_remap_fecal_quality('I tend to be constipated '
                                              '(have difficulty passing '
                                              'stool)'), 'Constipated')
        self.assertEqual(
            _remap_fecal_quality('I tend to have diarrhea (watery stool)'),
            'Diarrhea'
            )
        self.assertEqual(
            _remap_fecal_quality('I tend to have normal formed stool'),
            'Normal'
            )
        self.assertEqual(_remap_fecal_quality('foo'), 'foo')

    def test_remap_contraceptive(self):
        self.assertEqual(_remap_contraceptive('foo,bar'), 'foo')
        self.assertEqual(_remap_contraceptive('foo-bar'), 'foo-bar')
        self.assertEqual(_remap_contraceptive(3.5), 3.5)

    def test_remap_gluten(self):
        self.assertEqual('Celiac Disease',
                         _remap_gluten('I was diagnosed with celiac disease'))
        self.assertEqual('Non Medical',
                         _remap_gluten('I do not eat gluten because it makes '
                                        'me feel bad'))
        self.assertEqual("Anti-Gluten Allergy",
                         _remap_gluten('I was diagnosed with gluten allergy '
                                       '(anti-gluten IgG), but not celiac '
                                       'disease'))
        self.assertEqual('foo', _remap_gluten('foo'))

    def test_remap_last_travel(self):
        self.assertEqual('More than a year',
                         _remap_last_travel('I have not been outside of my '
                                            'country of residence in'
                                            ' the past year.'))
        self.assertEqual(_remap_last_travel('foo'), 'foo')

    def test_remap_sleep(self):
        self.assertEqual(_remap_sleep('Less than 5 hours'), 
                         'Less than 6 hours')
        self.assertEqual(_remap_sleep('5-6 hours'), 'Less than 6 hours')
        self.assertEqual(_remap_sleep('foo'), 'foo')

    def test_remap_diet(self):
        self.assertEqual(_remap_diet('Vegetarian but eat seafood'),
                         'Pescetarian')
        self.assertEqual(_remap_diet('Omnivore but do not eat red meat'),
                        'No red meat')
        self.assertEqual(_remap_diet('foo'), 'foo')

    def test_remap_fed_as_infant(self):
        self.assertEqual(_remap_fed_as_infant('Primarily breast milk'),
                         'Breast milk')
        self.assertEqual(_remap_fed_as_infant('Primarily infant formula'),
                         'Formula')
        self.assertEqual(_remap_fed_as_infant('A mixture of breast milk and '
                                              'formula'), 'Mix')
        self.assertEqual(_remap_fed_as_infant('foo'), 'foo')

    def test_remap_weight(self):
        self.assertEqual(_remap_weight('Increased more than 10 pounds'),
                         'Weight gain')
        self.assertEqual(_remap_weight('Decreased more than 10 pounds'),
                         'Weight loss')
        self.assertEqual(_remap_weight('foo'), 'foo')

    def test_ag_dictionary(self):
        name = 'BOWEL_MOVEMENT_QUALITY'
        description = ('Does the participant tend toward constipation or '
                       'diarrhea?')
        dtype = str
        order = ['I tend to have normal formed stool',
                 "I don't know, I do not have a point of reference",
                 'I tend to be constipated (have difficulty passing stool)',
                 'I tend to have diarrhea (watery stool)',
                 ]
        extremes = ["I tend to have normal formed stool",
                    "I tend to have diarrhea (watery stool)"]

        test = ag_dictionary(name)
        self.assertEqual(test.name, name)
        self.assertEqual(test.description, description)
        self.assertEqual(test.order, order)
        self.assertEqual(test.dtype, dtype)
        self.assertEqual(test.extremes, extremes)
        self.assertEqual(test.free_response, False)
        self.assertEqual(test.mimmarks, False)
        self.assertEqual(test.ontology, None)
        self.assertEqual('Normal'3,
                         test.remap_("I tend to have normal formed stool"))
        self.assertEqual('Constipated',
                         test.remap_("I tend to be constipated (have "
                                     "difficulty passing stool)"))
        self.assertEqual('Diarrhea',
                         test.remap_("I tend to have diarrhea (watery stool)")
                         )
        self.assertEqual('foo', test.remap_('foo'))

    def test_ag_dictionary_error(self):
        with self.assertRaises(ValueError):
            ag_dictionary('foo')

if __name__ == '__main__':
    main()