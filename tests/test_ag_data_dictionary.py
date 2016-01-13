from unittest import TestCase, main

import inspect

import pandas as pd
import numpy as np

import numpy.testing as npt
import pandas.util.testing as pdt

from americangut.ag_data_dictionary import (AgQuestion,
                                            AgCategorical,
                                            AgFrequency,
                                            AgClinical,
                                            AgMultiple,
                                            AgBool,
                                            AgContinous,
                                            _remap_abx,
                                            _remap_bowel_frequency,
                                            _remap_fecal_quality,
                                            _remap_contraceptive,
                                            _remap_gluten,
                                            _remap_last_travel,
                                            _remap_pool_frequency,
                                            _remap_sleep,
                                            _remap_diet)

amgut_sub = np.array([
    ["10317.000006668", "71.0", "male", "I do not have this condition",
     "false", "false", "false", "true", "", "", "07/09/2013 08:30", "Never"],
    ["10317.000030344", "40.0", "female", "I do not have this condition",
     "false", "true", "false", "false", "One", "I tend to have normal formed "
     "stool", "09/08/2015 09:10", "Regularly (3-5 times/week)"],
    ["10317.000031833", "76.0", "female", "I do not have this condition",
     "true", "true", "false", "false", "One", "I tend to have normal formed "
     "stool", "08/17/2015 08:33", "Daily"],
    ["10317.000013134", "45.0", "male", "", "false", "false", "false", "true",
     "", "", "01/12/2014 00:35", "Regularly (3-5 times/week)"],
    ["10317.000022634", "40.0", "male", "I do not have this condition",
     "false", "false", "false", "true", "Less than one", "I tend to be "
     "constipated (have difficulty passing stool)", "02/02/2015 09:00",
     "Never"],
    ["10317.000020683", "34.0", "male", "I do not have this condition",
     "true", "true", "false", "false", "Less than one", "I don't know, "
     "I do not have a point of reference", "11/08/2014 10:15",
     "Regularly (3-5 times/week)"],
    ["10317.000022916", "77.0", "male", "I do not have this condition", "true",
     "false", "false", "false", "One", "I don't know, I do not have a point of"
     " reference", "01/13/2015 19:14", "Rarely (a few times/month)"],
    ["10317.000031363", "45.0", "male", "I do not have this condition", "true",
     "false", "false", "false", "One", "I tend to be constipated (have "
     "difficulty passing stool)", "08/07/2015 08:45",
     "Occasionally (1-2 times/week)"],
    ["10317.000006952", "33.0", "female", "I do not have this condition",
     "false", "false", "false", "true", "", "", "07/03/2013 12:45",
     "Rarely (a few times/month)"],
    ["10317.000027363", "52.0", "male", "I do not have this condition",
     "true", "false", "false", "false", "One", "", "07/05/2015 11:35",
     "Rarely (a few times/month)"],
    ["10317.000002337", "57.0", "female", "I do not have this condition",
     "false", "false", "false", "true", "", "", "05/08/2013 09:10", "Daily"],
    ["10317.000027574", "6.0", "female", "I do not have this condition",
     "false", "false", "false", "true", "One", "I tend to have normal formed "
     "stool", "06/17/2015 19:15", "Never"],
    ["10317.000003365", "26.0", "female", "I do not have this condition",
     "false", "false", "false", "true", "", "", "04/02/2013 10:30", "Regularly"
     " (3-5 times/week)"],
    ["10317.000002347", "45.0", "female", "I do not have this condition",
     "false", "false", "false", "true", "", "", "05/04/2013 14:40",
     "Occasionally (1-2 times/week)"],
    ["10317.000011322", "", "female", "I do not have this condition", "false",
     "false", "false", "true", "", "", "01/14/2014 09:08", "Regularly "
     "(3-5 times/week)"],
    ["10317.000020053", "35.0", "male", "Self-diagnosed", "true", "true",
     "false", "false", "Three", "I tend to have normal formed stool",
     "01/03/2015 17:15", "Occasionally (1-2 times/week)"],
    ["10317.000029572", "65.0", "male", "Diagnosed by a medical professional "
     "(doctor, physician assistant)", "false", "false", "false", "false",
     "Less than one", "I tend to be constipated (have difficulty passing "
     "stool)", "08/28/2015 14:00", "Daily"],
    ["10317.000023046", "14.0", "female", "Diagnosed by a medical professional"
     " (doctor, physician assistant)", "false", "false", "false", "true",
     "Three", "I tend to have normal formed stool", "01/27/2015 10:15",
     "Never"],
    ["10317.000028154", "54.0", "female", "Diagnosed by a medical professional"
     " (doctor, physician assistant)", "true", "true", "false", "false",
     "Five or more", "I tend to have diarrhea (watery stool)",
     "06/25/2015 21:25", "Occasionally (1-2 times/week)"],
    ["10317.000020499", "33.0", "female", "Diagnosed by a medical professional"
     " (doctor, physician assistant)", "true", "false", "false", "false",
     "Five or more", "I tend to have diarrhea (watery stool)",
     "12/02/2014 09:00", "Rarely (a few times/month)"],
    ])


class AgQuestionTest(TestCase):
    def setUp(self):
        self.map_ = pd.DataFrame(
            amgut_sub,
            columns=["#SampleID", "AGE_YEARS", "SEX", "IBD",
                     "ALCOHOL_TYPES_BEERCIDER", "ALCOHOL_TYPES_RED_WINE",
                     "ALCOHOL_TYPES_SOUR_BEERS", "ALCOHOL_TYPES_UNSPECIFIED",
                     "BOWEL_MOVEMENT_FREQUENCY", "BOWEL_MOVEMENT_QUALITY",
                     "COLLECTION_TIMESTAMP", "ALCOHOL_FREQUENCY"],
            ).set_index('#SampleID')
        self.map_.replace('', np.nan, inplace=True)

        self.name = 'TEST_COLUMN'
        self.description = ('"Say something, anything."\n'
                            '"Testing... 1, 2, 3"\n'
                            '"Anything but that!"')
        self.dtype = str

        self.fun = lambda x: 'foo'

        self.ag_question = AgQuestion(
            name='ALCOHOL_TYPES_UNSPECIFIED',
            description=('Has the participant described their alcohol use'),
            dtype=bool
            )

    def test_init(self):
        test = AgQuestion(self.name, self.description, self.dtype)

        self.assertEqual(test.name, self.name)
        self.assertEqual(test.description, self.description)
        self.assertEqual(test.dtype, self.dtype)
        self.assertEqual(test.clean_name, 'Test Column')
        self.assertEqual(test.free_response, False)
        self.assertEqual(test.mimmarks, False)
        self.assertEqual(test.ontology, None)
        self.assertEqual(test.subset, True)
        self.assertEqual(test.remap_, None)

    def test_init_kwargs(self):
        test = AgQuestion(self.name, self.description, self.dtype,
                          clean_name='This is a test.',
                          free_response=True,
                          mimmarks=True,
                          ontology='GAZ',
                          remap=self.fun,
                          subset=False
                          )

        self.assertEqual(test.name, self.name)
        self.assertEqual(test.description, self.description)
        self.assertEqual(test.dtype, self.dtype)
        self.assertEqual(test.clean_name, 'This is a test.')
        self.assertEqual(test.free_response, True)
        self.assertEqual(test.mimmarks, True)
        self.assertEqual(test.ontology, 'GAZ')
        self.assertEqual(test.subset, False)
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

    def test_remap_data_type(self):
        self.ag_question.remap_data_type(self.map_)
        self.assertEquals(set(self.map_[self.ag_question.name]) - {np.nan},
                          {True, False})
        self.ag_question.dtype = int
        self.ag_question.remap_data_type(self.map_)
        self.assertEqual(set(self.map_[self.ag_question.name]) - {np.nan},
                         {0, 1})


class AgCategoricalTest(TestCase):

    def setUp(self):
        self.map_ = pd.DataFrame(
            amgut_sub,
            columns=["#SampleID", "AGE_YEARS", "SEX", "IBD",
                     "ALCOHOL_TYPES_BEERCIDER", "ALCOHOL_TYPES_RED_WINE",
                     "ALCOHOL_TYPES_SOUR_BEERS",
                     "ALCOHOL_TYPES_UNSPECIFIED",
                     "BOWEL_MOVEMENT_FREQUENCY", "BOWEL_MOVEMENT_QUALITY",
                     "COLLECTION_TIMESTAMP", "ALCOHOL_FREQUENCY"],
            ).set_index('#SampleID')
        self.map_.replace('', np.nan, inplace=True)

        self.name = 'BOWEL_MOVEMENT_QUALITY'
        self.description = ('Whether the participant is constipated, has '
                            'diarrhea or is regular.')
        self.dtype = str
        self.order = ["I don't know, I do not have a point of reference",
                      "I tend to have normal formed stool",
                      "I tend to be constipated (have difficulty passing "
                      "stool)",
                      "I tend to have diarrhea (watery stool)",
                      ]
        self.extremes = ["I tend to have normal formed stool",
                         "I tend to have diarrhea (watery stool)"]

        def remap_poo_quality(x):
            if x in {"I don't know, I do not have a point of reference"}:
                return 'No Reference'
            elif x == 'I tend to have normal formed stool':
                return 'Well formed'
            elif x == ('I tend to be constipated (have difficulty passing'
                       ' stool)'):
                return 'Constipated'
            elif x == 'I tend to have diarrhea (watery stool)':
                return 'Diarrhea'
            else:
                return x

        self.remap_ = remap_poo_quality

        self.ag_categorical = AgCategorical(
            name=self.name,
            description=self.description,
            dtype=self.dtype,
            order=self.order,
            extremes=self.extremes,
            remap=remap_poo_quality
            )

        self.ag_clinical = AgClinical(
            name='IBD',
            description=('Has the participant been diagnosed with IBD?'),
            )

        self.ag_frequency = AgFrequency(
            name='ALCOHOL_FREQUENCY',
            description='How often the participant consumes alcohol.',
            )

    def test_ag_categorical_init(self):
        self.assertEqual('No Reference', self.remap_(
            "I don't know, I do not have a point of reference"
            ))
        self.assertEqual(3, self.remap_(3))

        test = AgCategorical(self.name,
                             self.description,
                             self.dtype,
                             self.order,
                             remap=self.remap_)

        self.assertEqual(self.order, test.order)
        self.assertEqual([], test.earlier_order)
        self.assertEqual(test.extremes,
                         ["I don't know, I do not have a point of reference",
                          "I tend to have diarrhea (watery stool)"
                          ])
        self.assertEqual('No Reference', test.remap_(
            "I don't know, I do not have a point of reference"
            ))
        self.assertEqual(3, test.remap_(3))
        self.assertEqual(test.type, 'Categorical')

    def test_ag_init_error(self):
        with self.assertRaises(ValueError):
            AgCategorical(self.name, self.description, list, self.order)

    def test_ag_clinical_init(self):
        clinical_groups = ['Diagnosed by a medical professional '
                           '(doctor, physician assistant)',
                           'Diagnosed by an alternative medicine practitioner',
                           'Self-diagnosed',
                           'I do not have this condition']
        test_class = AgClinical(
            name='TEST_COLUMN',
            description='Testing. 1, 2, 3.\nAnything but that!',
            strict=True,
            )
        self.assertEqual(test_class.dtype, str)
        self.assertEqual(test_class.order, clinical_groups)
        self.assertEqual(test_class.type, 'Clinical')

    def test_ag_frequency_init(self):
        order = ['Never', 'Rarely (a few times/month)',
                 'Occasionally (1-2 times/week)', 'Regularly (3-5 times/week)',
                 'Daily']
        test = AgFrequency(
            name='ALCOHOL_FREQUENCY',
            description=('How often the participant consumes alcohol.'),
            )

        self.assertEqual(test.order, order)
        self.assertEqual(test.dtype, str)
        self.assertEqual(test.type, 'Frequency')

    def test_ag_categorical_update_groups(self):
        self.ag_categorical._update_order(self.ag_categorical.remap_)
        self.assertEqual(self.ag_categorical.order,
                         ['No Reference', 'Well formed', 'Constipated',
                          'Diarrhea'])
        self.assertEqual(self.ag_categorical.extremes,
                         ['Well formed', 'Diarrhea'])

        self.assertEqual(self.ag_categorical.earlier_order, [self.order])

    def test_ag_categorical_remove_ambigious(self):
        self.ag_categorical.drop_ambiguous = True
        self.ag_categorical.remove_ambiguity(self.map_)

        self.assertEqual(set(self.map_[self.ag_categorical.name]) - {np.nan}, {
            "I tend to be constipated (have difficulty passing stool)",
            "I tend to have diarrhea (watery stool)",
            "I tend to have normal formed stool",
            })

    def test_ag_categorical_remap_groups(self):
        self.ag_categorical.remap_groups(self.map_)

        self.assertEqual(set(self.map_[self.ag_categorical.name]) - {np.nan},
                         {'No Reference', 'Well formed', 'Constipated',
                          'Diarrhea'})

        self.assertEqual(self.ag_categorical.earlier_order, [self.order])

    def test_ag_categorical_drop_infrequent(self):
        self.ag_categorical.frequency_cutoff = 4
        self.ag_categorical.drop_infrequent(self.map_)

        self.assertEqual(set(self.map_[self.ag_categorical.name]) - {np.nan}, {
            "I tend to have normal formed stool"
            })

    def test_ag_clinical_remap_strict(self):
        self.ag_clinical.remap_clinical(self.map_)

        count = pd.Series([14, 5],
                          index=pd.Index(['No', 'Yes'], name='IBD'),
                          dtype=np.int64)
        pdt.assert_series_equal(count, self.map_.groupby('IBD').count().max(1))

    def test_ag_clincial_remap(self):
        self.ag_clinical.strict = False
        self.ag_clinical.remap_clinical(self.map_)

        count = pd.Series([14, 4],
                          index=pd.Index(['No', 'Yes'], name='IBD'),
                          dtype=np.int64)
        pdt.assert_series_equal(count, self.map_.groupby('IBD').count().max(1))

    def test_ag_frequency_clean(self):
        self.ag_frequency.clean(self.map_)
        self.assertEqual(set(self.map_[self.ag_frequency.name]) - {np.nan}, {
            'Never', 'A few times/month', '1-2 times/week', '3-5 times/week',
            'Daily'
            })


class AgBoolTest(TestCase):
    def setUp(self):
        self.map_ = pd.DataFrame(
            amgut_sub,
            columns=["#SampleID", "AGE_YEARS", "SEX", "IBD",
                     "ALCOHOL_TYPES_BEERCIDER", "ALCOHOL_TYPES_RED_WINE",
                     "ALCOHOL_TYPES_SOUR_BEERS",
                     "ALCOHOL_TYPES_UNSPECIFIED",
                     "BOWEL_MOVEMENT_FREQUENCY", "BOWEL_MOVEMENT_QUALITY",
                     "COLLECTION_TIMESTAMP", "ALCOHOL_FREQUENCY"],
            ).set_index('#SampleID')
        self.map_.replace('', np.nan, inplace=True)

        self.name = 'ALCOHOL_TYPES_BEERCIDER'
        self.description = ('The participant drinks beer or cider.')
        self.dtype = bool
        self.order = ['true', 'false']
        self.extremes = ['true', 'false']
        self.unspecified = 'ALCOHOL_TYPES_UNSPECIFIED'

        self.ag_bool = AgBool(
            name='ALCOHOL_TYPES_UNSPECIFIED',
            description=('Has the participant described their alcohol use'),
            )
        self.ag_multiple = AgMultiple(
            name=self.name,
            description=self.description,
            unspecified=self.unspecified,
            )

    def test_ag_bool_init(self):
        name = 'TEST_COLUMN'
        description = 'Testing. 1, 2, 3.\nAnything but that!'

        test_class = AgBool(name, description)

        self.assertEqual(test_class.dtype, bool)
        self.assertEqual(test_class.order, self.order)
        self.assertEqual(test_class.type, 'Bool')

    def test_ag_multiple_init(self):
        test = AgMultiple(
            name=self.name,
            description=self.description,
            unspecified=self.unspecified)
        self.assertEqual(test.unspecified, self.unspecified)

    def test_ag_multiple_correct_unspecified(self):
        self.ag_multiple.remap_data_type(self.map_)
        self.assertEqual(np.sum(pd.isnull(self.map_[self.ag_multiple.name])),
                         0)
        self.ag_multiple.correct_unspecified(self.map_)
        self.assertEqual(np.sum(pd.isnull(self.map_[self.ag_multiple.name])),
                         10)

    def test_ag_multiple_unspecified_error(self):
        self.ag_multiple.unspecified = 'TEST_COLUMN'
        with self.assertRaises(ValueError):
            self.ag_multiple.correct_unspecified(self.map_)


class AgContinousTest(TestCase):

    def setUp(self):
        self.map_ = pd.DataFrame(
            amgut_sub,
            columns=["#SampleID", "AGE_YEARS", "SEX", "IBD",
                     "ALCOHOL_TYPES_BEERCIDER", "ALCOHOL_TYPES_RED_WINE",
                     "ALCOHOL_TYPES_SOUR_BEERS",
                     "ALCOHOL_TYPES_UNSPECIFIED",
                     "BOWEL_MOVEMENT_FREQUENCY", "BOWEL_MOVEMENT_QUALITY",
                     "COLLECTION_TIMESTAMP", "ALCOHOL_FREQUENCY"],
            ).set_index('#SampleID')
        self.map_.replace('', np.nan, inplace=True)

        self.name = 'AGE_YEARS'
        self.description = 'Participant age in years'
        self.unit = 'years'

        self.ag_continous = AgContinous(self.name, self.description, self.unit,
                                        range=[20, 50])

    def test_ag_continous_init(self):
        test = AgContinous(
            name=self.name,
            description=self.description,
            unit=self.unit
            )
        self.assertEqual(self.unit, test.unit)
        self.assertEqual('Continous', test.type)
        self.assertEqual(test.dtype, float)

    def test_ag_continous_init_error(self):
        with self.assertRaises(ValueError):
            AgContinous('TEST_COLUMN',
                        'Testing. 1, 2, 3.\nAnything but that!',
                        range=[12, 5])

    def test_ag_continous_drop_outliers(self):
        self.assertEqual(self.map_[
            self.ag_continous.name].dropna().astype(float).min(),
                         6.0)
        self.assertEqual(self.map_[
            self.ag_continous.name].dropna().astype(float).max(),
                         77.0)

        self.ag_continous.drop_outliers(self.map_)
        self.assertEqual(self.map_[self.ag_continous.name].dropna().min(),
                         26.0)
        self.assertEqual(self.map_[self.ag_continous.name].dropna().max(),
                         45.0)
        self.assertEqual(np.sum(pd.isnull(self.map_[self.ag_continous.name])),
                         10)


class AgAuxillaryDictTest(TestCase):
    def test_remap_abx(self):
        self.assertEqual(_remap_abx('I have not taken antibiotics in the past '
                                    'year'), 'More than a year')
        self.assertEqual(_remap_abx(3), 3)
        self.assertEqual(_remap_abx('cat'), 'cat')

    def test_remap_bowel_frequency(self):
        self.assertEqual(_remap_bowel_frequency('Four'), 'Four or more')
        self.assertEqual(_remap_bowel_frequency('Five or more'),
                         'Four or more')
        self.assertEqual(_remap_bowel_frequency('foo'), 'foo')

    def test_remap_fecal_quality(self):
        self.assertEqual('Diarrhea',
                         _remap_fecal_quality('I tend to have diarrhea '
                                              '(watery stool)'))
        self.assertEqual('Constipated',
                         _remap_fecal_quality('I tend to be constipated '
                                              '(have difficulty passing '
                                              'stool)'))
        self.assertEqual('Normal',
                         _remap_fecal_quality('I tend to have normal formed'
                                              ' stool'))
        self.assertEqual('foo',
                         _remap_fecal_quality('foo'))

    def test_remap_contraception(self):
        self.assertEqual('Yes', _remap_contraceptive('Yes, I like Supergirl'))
        self.assertEqual('I like Supergirl',
                         _remap_contraceptive('I like Supergirl'))

    def test_remap_gluten(self):
        self.assertEqual('Celiac Disease',
                         _remap_gluten('I was diagnosed with celiac disease'))
        self.assertEqual('Non Medical',
                         _remap_gluten('I do not eat gluten because it makes'
                                       ' me feel bad'))
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
        self.assertEqual('foo', _remap_last_travel('foo'))

    def test_remap_pool_frequency(self):
        self.assertEqual('More than once a week',
                         _remap_pool_frequency('Daily'))
        self.assertEqual('More than once a week',
                         _remap_pool_frequency('Occasionally '
                                               '(1-2 times/week)')),
        self.assertEqual('More than once a week',
                         _remap_pool_frequency('Regularly (3-5 times/week)'))
        self.assertEqual('More than once a week',
                         _remap_pool_frequency('1-2 times/week')),
        self.assertEqual('More than once a week',
                         _remap_pool_frequency('3-5 times/week'))
        self.assertEqual('foo', _remap_pool_frequency('foo'))

    def test_remap_sleep(self):
        self.assertEqual('Less than 6 hours',
                         _remap_sleep('Less than 5 hours'))
        self.assertEqual('Less than 6 hours',
                         _remap_sleep('5-6 hours'))
        self.assertEqual('foo', _remap_sleep('foo'))


if __name__ == '__main__':
    main()
