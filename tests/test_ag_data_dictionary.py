from unittest import TestCase, main

import inspect

import pandas as pd
import numpy as np

import numpy.testing as npt
import pandas.util.testing as pdt

from americangut.ag_data_dictionary import *

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


class AgDictionaryTest(TestCase):
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

        self.remap_poo_quality = remap_poo_quality

        self.ag_question = AgQuestion(
            name='ALCOHOL_TYPES_UNSPECIFIED',
            description=('Has the participant described their alcohol use'),
            dtype=bool
            )
        self.ag_categorical = AgCategorical(
            name='BOWEL_MOVEMENT_QUALITY',
            description=('Whether the participant is constipated, has '
                         'diarrhea or is regular.'),
            dtype=str,
            groups={"I don't know, I do not have a point of reference",
                    "I tend to be constipated (have difficulty passing stool)",
                    "I tend to have diarrhea (watery stool)",
                    "I tend to have normal formed stool"
                    },
            remap=remap_poo_quality
            )

        self.ag_clinical = AgCategorical(
            name='IBD',
            description=('Has the participant been diagnosed with IBD?'),
            dtype=str,
            groups={
                ('Diagnosed by a medical professional (doctor, physician '
                 'assistant)'),
                'Diagnosed by an alternative medicine practitioner',
                'Self-diagnosed',
                'I do not have this condition'
                },
            clinical=True
            )

        self.ag_ordinal = AgOrdinal(
            name='ALCOHOL_FREQUENCY',
            description='How often the participant consumes alcohol.',
            dtype=str,
            order=['Never', 'Rarely (a few times/month)',
                   'Occasionally (1-2 times/week)',
                   'Regularly (3-5 times/week)',
                   'Daily']
            )

        self.ag_continous = AgContinous(
            name='AGE_YEARS',
            description='Participant age in years',
            dtype=float,
            unit='years'
            )

        self.ag_multiple = AgMultiple(
            name="ALCOHOL_TYPES_BEERCIDER",
            description='Does the participant drink beer or cider',
            dtype=bool,
            unspecified="ALCOHOL_TYPES_UNSPECIFIED"
            )

    def test_ag_question_init(self):
        name = 'TEST_COLUMN'
        description = 'Testing. 1, 2, 3.\nAnything but that!'
        dtype = str
        clean_name = 'Test Column'

        test_class = AgQuestion(name, description, dtype, clean_name)
        self.assertEqual(name, test_class.name)
        self.assertEqual(description, test_class.description)
        self.assertEqual(dtype, test_class.dtype)
        self.assertEqual(clean_name, test_class.clean_name)

    def test_ag_question_init_name_error(self):
        name = ['TEST_COLUMN']
        description = 'Testing. 1, 2, 3.\nAnything but that!'
        dtype = str

        with self.assertRaises(TypeError):
            AgQuestion(name, description, dtype)

    def test_ag_question_init_description_error(self):
        name = 'TEST_COLUMN'
        description = {'Joann': 'Testing. 1, 2, 3.',
                       'Mark': 'Anything but that!'}
        dtype = str

        with self.assertRaises(TypeError):
            AgQuestion(name, description, dtype)

    def test_ag_question_init_dtype_error(self):
        name = 'TEST_COLUMN'
        description = 'Testing. 1, 2, 3.\nAnything but that!'
        dtype = 'str'

        with self.assertRaises(TypeError):
            AgQuestion(name, description, dtype)

    def test_ag_question_init_clean_name_error(self):
        name = 'TEST_COLUMN'
        description = 'Testing. 1, 2, 3.\nAnything but that!'
        dtype = str
        clean_name = ['Test Column']

        with self.assertRaises(TypeError):
            AgQuestion(name, description, dtype, clean_name)

    def test_ag_question_remap_data_type(self):
        self.assertTrue(isinstance(self.map_.loc['10317.000006668',
                                   self.ag_question.name], str))

        self.ag_question.remap_data_type(self.map_)
        self.assertTrue(isinstance(self.map_.loc['10317.000006668',
                                   self.ag_question.name], np.bool_))
        self.assertEqual(self.map_[self.ag_question.name].dtype, np.bool_)
        self.assertEqual(np.sum(self.map_[self.ag_question.name]), 10)

        self.ag_question.dtype = int
        self.ag_question.remap_data_type(self.map_)
        self.assertTrue(isinstance(self.map_.loc['10317.000006668',
                                   self.ag_question.name], int))

    def test_ag_question_check_map_error(self):
        test_class = AgQuestion(
            name='TEST_COLUMN',
            description='Testing. 1, 2, 3.\nAnything but that!',
            dtype=str
        )
        with self.assertRaises(ValueError):
            test_class.check_map(self.map_)

    def test_ag_categorical_init(self):
        name = 'BOWEL_MOVEMENT_QUALITY'
        description = ('Whether the participant is constipated, has '
                       'diarrhea or is regular.')
        dtype = str
        groups = {"I don't know, I do not have a point of reference",
                  "I tend to be constipated (have difficulty passing stool)",
                  "I tend to have diarrhea (watery stool)",
                  "I tend to have normal formed stool"
                  }

        self.assertEqual('No Reference', self.remap_poo_quality(
            "I don't know, I do not have a point of reference"
            ))
        self.assertEqual(3, self.remap_poo_quality(3))

        test = AgCategorical(name,
                             description,
                             dtype,
                             groups,
                             remap=self.remap_poo_quality)

        self.assertEqual(groups, test.groups)
        self.assertEqual([], test.earlier_groups)
        self.assertEqual('No Reference', test.remap_(
            "I don't know, I do not have a point of reference"
            ))
        self.assertEqual(3, test.remap_(3))
        self.assertEqual(test.type, 'Categorical')

    def test_ag_categorical_remove_ambigious(self):
        self.ag_categorical.remove_ambigious(self.map_)

        self.assertEqual(set(self.map_[self.ag_categorical.name]) - {np.nan}, {
            "I tend to be constipated (have difficulty passing stool)",
            "I tend to have diarrhea (watery stool)",
            "I tend to have normal formed stool",
            })

        self.assertEqual(self.ag_categorical.groups, {
            "I tend to be constipated (have difficulty passing stool)",
            "I tend to have diarrhea (watery stool)",
            "I tend to have normal formed stool"
            })

        self.assertEqual(self.ag_categorical.earlier_groups, [{
            "I don't know, I do not have a point of reference",
            "I tend to be constipated (have difficulty passing stool)",
            "I tend to have diarrhea (watery stool)",
            "I tend to have normal formed stool"
            }])

    def test_ag_categorical_remap_groups(self):
        self.ag_categorical.remap_groups(self.map_)

        self.assertEqual(set(self.map_[self.ag_categorical.name]) - {np.nan},
                         {'No Reference', 'Well formed', 'Constipated',
                          'Diarrhea'})

        self.assertEqual(self.ag_categorical.earlier_groups, [{
            "I don't know, I do not have a point of reference",
            "I tend to be constipated (have difficulty passing stool)",
            "I tend to have diarrhea (watery stool)",
            "I tend to have normal formed stool"
            }])

    def test_ag_categorical_drop_infrequent(self):
        self.ag_categorical.drop_infrequency(self.map_, cutoff=4)

        self.assertEqual(set(self.map_[self.ag_categorical.name]) - {np.nan}, {
            "I tend to have normal formed stool"
            })
        self.assertEqual({"I tend to have normal formed stool"},
                         self.ag_categorical.groups)
        self.assertEqual(self.ag_categorical.earlier_groups, [{
            "I don't know, I do not have a point of reference",
            "I tend to be constipated (have difficulty passing stool)",
            "I tend to have diarrhea (watery stool)",
            "I tend to have normal formed stool"
            }])

    def test_ag_categorical_clincal_error(self):
        with self.assertRaises(ValueError):
            self.ag_categorical.remap_clinical(self.map_)

    def test_ag_categorical_remap_clinical_strict(self):
        self.ag_clinical.remap_clinical(self.map_)

        count = pd.Series([14, 5],
                          index=pd.Index(['No', 'Yes'], name='IBD'),
                          dtype=np.int64)
        pdt.assert_series_equal(count, self.map_.groupby('IBD').count().max(1))

        self.assertEqual(self.ag_clinical.groups, {'Yes', 'No'})
        self.assertEqual(self.ag_clinical.earlier_groups, [{
            ('Diagnosed by a medical professional (doctor, physician '
             'assistant)'),
            'Diagnosed by an alternative medicine practitioner',
            'Self-diagnosed',
            'I do not have this condition'
            }])

    def test_ag_ordinal_init(self):
        order = ['Never', 'Rarely (a few times/month)',
                 'Occasionally (1-2 times/week)', 'Regularly (3-5 times/week)',
                 'Daily']

        test = AgOrdinal(
            name='ALCOHOL_FREQUENCY',
            description=('How often the participant consumes alcohol.'),
            dtype=str,
            order=order
            )

        self.assertEqual(test.order, order)
        self.assertEqual(test.type, 'Ordinal')

    def test_ag_ordinal_clean_frequency(self):
        self.ag_ordinal.clean_frequency(self.map_)
        self.assertEqual(set(self.map_[self.ag_ordinal.name]) - {np.nan}, {
            'Never', 'A few times/month', '1-2 times/week', '3-5 times/week',
            'Daily'
            })
        self.assertEqual(self.ag_ordinal.earlier_groups, [[
            'Never', 'Rarely (a few times/month)',
            'Occasionally (1-2 times/week)', 'Regularly (3-5 times/week)',
            'Daily'
            ]])
        npt.assert_array_equal(self.ag_ordinal.order, [
            'Never', 'A few times/month', '1-2 times/week', '3-5 times/week',
            'Daily'
            ])

    def test_ag_ordinal_convert_to_numeric(self):
        self.ag_ordinal.convert_to_numeric(self.map_)
        self.assertEqual(set(self.map_[self.ag_ordinal.name]) - {np.nan},
                         {0, 1, 2, 3, 4})
        self.assertEqual(self.ag_ordinal.groups, {0, 1, 2, 3, 4})
        self.assertEqual(self.ag_ordinal.earlier_groups, [[
            'Never', 'Rarely (a few times/month)',
            'Occasionally (1-2 times/week)', 'Regularly (3-5 times/week)',
            'Daily'
            ]])
        npt.assert_array_equal(self.ag_ordinal.order,
                               np.array([0, 1, 2, 3, 4]))

    def test_ag_ordinal_label_order(self):
        self.ag_ordinal.label_order(self.map_)
        self.assertEqual(set(self.map_[self.ag_ordinal.name]) - {np.nan}, {
            '(0) Never',
            '(1) Rarely (a few times/month)',
            '(2) Occasionally (1-2 times/week)',
            '(3) Regularly (3-5 times/week)',
            '(4) Daily'
            })
        self.assertEqual(self.ag_ordinal.earlier_groups, [[
            'Never', 'Rarely (a few times/month)',
            'Occasionally (1-2 times/week)', 'Regularly (3-5 times/week)',
            'Daily'
            ]])
        self.assertEqual(self.ag_ordinal.order, [
            '(0) Never',
            '(1) Rarely (a few times/month)',
            '(2) Occasionally (1-2 times/week)',
            '(3) Regularly (3-5 times/week)',
            '(4) Daily'
            ])

    def test_ag_continous_init(self):
        unit = 'years'

        test = AgContinous(
            name='AGE_YEARS',
            description='Participant age in years',
            dtype=int,
            unit=unit
            )

        self.assertEqual(unit, test.unit)
        self.assertEqual('Continous', test.type)

    def test_ag_continous_drop_outliers_limit_error(self):
        with self.assertRaises(ValueError):
            self.ag_continous.drop_outliers(self.map_, 12, 5)

    def test_ag_continous_drop_outliers(self):
        self.assertEqual(self.map_[
            self.ag_continous.name].dropna().astype(float).min(),
                         6.0)
        self.assertEqual(self.map_[
            self.ag_continous.name].dropna().astype(float).max(),
                         77.0)
        # self.assertEqual(self.map_.AGE_YEAR.dropna().min(), 2
        self.ag_continous.drop_outliers(self.map_, 20, 50)
        self.assertEqual(self.map_[self.ag_continous.name].dropna().min(),
                         26.0)
        self.assertEqual(self.map_[self.ag_continous.name].dropna().max(),
                         45.0)
        self.assertEqual(np.sum(pd.isnull(self.map_[self.ag_continous.name])),
                         10)

    def test_ag_multiple_init(self):
        unspecified = "ALCOHOL_TYPES_UNSPECIFIED"
        test = AgMultiple(
            name="ALCOHOL_TYPES_BEERCIDER",
            description='Does the participant drink beer or cider',
            dtype=bool,
            unspecified=unspecified)
        self.assertEqual(test.unspecified, unspecified)

    def test_ag_multiple_correct_unspecified(self):
        self.ag_multiple.remap_data_type(self.map_)
        self.assertEqual(np.sum(pd.isnull(self.map_[self.ag_multiple.name])),
                         0)
        self.ag_multiple.correct_unspecified(self.map_)
        self.assertEqual(np.sum(pd.isnull(self.map_[self.ag_multiple.name])),
                         10)

if __name__ == '__main__':
    main()
