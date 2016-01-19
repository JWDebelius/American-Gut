import inspect
import os
from unittest import TestCase, main

import biom
import numpy as np
import pandas as pd
import numpy.testing as npt
import pandas.util.testing as pdt
import skbio

import americangut as ag
from americangut.ag_data import (AgData,
                                 AgQuestion,
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


class AgDataTest(TestCase):
    def setUp(self):
        self.repo_dir = ag.WORKING_DIR.split('American-Gut')[0]
        self.base_dir = os.path.join(self.repo_dir,
                                     'American-Gut/tests/data/11-packaged')
        self.bodysite = 'fecal'
        self.trim = '100nt'
        self.depth = '10k'

        self.ag_data = AgData(
            bodysite=self.bodysite,
            trim=self.trim,
            depth=self.depth,
            base_dir=self.base_dir
            )

        # Sets up the known map data for the 10k data
        map_data = np.array(
            [['60', 'Daily', 'false', 'true', '25.13', 'Overweight',
              'I do not have this condition', '2.866733', '2.970151',
              '2.49309705457', '2.37419183588', '9.15', '9.7', '7.6', '8.0'],
             ['57.0', 'Occasionally (1-2 times/week)', 'false', 'true',
              '26.73', 'Overweight', 'I do not have this condition',
              '2.198198', '2.446593', '1.83213159668', '1.91370771085', '5.65',
              '6.3', '5.1', '5.9'],
             ['34.0', 'Rarely (a few times/month)', 'false', 'true', '19.27',
              'Normal', 'None', '1.932658', '2.012048', '0.970334630884',
              '1.0890945893', '4.05', '4.2', '3.9', '4.2']],
            dtype=object)
        columns = pd.Index([
            'AGE_YEARS', 'ALCOHOL_FREQUENCY', 'ALCOHOL_TYPES_BEERCIDER',
            'ALCOHOL_TYPES_UNSPECIFIED', 'BMI', 'BMI_CAT', 'IBD',
            'PD_whole_tree_1k', 'PD_whole_tree_10k', 'shannon_1k',
            'shannon_10k', 'chao1_1k', 'chao1_10k', 'observed_otus_1k',
            'observed_otus_10k'
            ], dtype='object')
        index = pd.Index(['10317.000007108', '10317.000005844',
                          '10317.000002035'], dtype='object', name='#SampleID')
        alpha_cols = ['PD_whole_tree_1k', 'PD_whole_tree_10k', 'shannon_1k',
                      'shannon_10k', 'chao1_1k', 'chao1_10k',
                      'observed_otus_1k', 'observed_otus_10k']
        self.map_ = pd.DataFrame(map_data, columns=columns, index=index)
        self.map_.replace('None', np.nan, inplace=True)
        self.map_[alpha_cols] = self.map_[alpha_cols].astype(float)

        # Sets up the otu data for the 10k depth
        otu_data = np.array([[14., 11., 31.],
                             [00.,  6.,  2.],
                             [80., 39., 41.],
                             [00.,  1.,  8.],
                             [00., 18.,  1.],
                             [03.,  3.,  1.],
                             [03., 22., 16.]])
        otu_ids = np.array([u'4447072', u'4352657', u'4468234', u'1000986',
                            u'4481131', u'4479989', u'2532173'], dtype=object)
        sample = np.array([u'10317.000002035', u'10317.000007108',
                           u'10317.000005844'], dtype=object)
        self.otu_ = biom.Table(otu_data, otu_ids, sample)

        # Sets up known distance matrices
        unweighted = np.array([[0.00000000,  0.29989178,  0.29989178],
                               [0.29989178,  0.00000000,  0.00000000],
                               [0.29989178,  0.00000000,  0.00000000]])

        weighted = np.array([[0.0000000,  0.5983488,  0.3448625],
                             [0.5983488,  0.0000000,  0.3317153],
                             [0.3448625,  0.3317153,  0.0000000]])
        self.beta = {
            'unweighted_unifrac':
            skbio.DistanceMatrix(data=unweighted, ids=('10317.000002035',
                                                       '10317.000007108',
                                                       '10317.000005844')),
            'weighted_unifrac':
            skbio.DistanceMatrix(data=weighted, ids=('10317.000002035',
                                                     '10317.000007108',
                                                     '10317.000005844'))
            }

    def test__init__(self):
        ag_data = AgData(
            bodysite=self.bodysite,
            trim=self.trim,
            depth=self.depth,
            base_dir=self.base_dir
            )
        # Checks dataset
        self.assertEqual(ag_data.base_dir, self.base_dir)
        self.assertEqual(ag_data.bodysite, self.bodysite)
        self.assertEqual(ag_data.trim, self.trim)
        self.assertEqual(ag_data.depth, self.depth)
        self.assertFalse(ag_data.sub_participants)
        self.assertFalse(ag_data.one_sample)

    def test__init__bodysite(self):
        with self.assertRaises(ValueError):
            AgData(bodysite='Vagina',
                   trim=self.trim,
                   depth=self.depth)

    def test__init__trim(self):
        with self.assertRaises(ValueError):
            AgData(bodysite=self.bodysite,
                   trim='75nt',
                   depth=self.depth)

    def test__init__depth(self):
        with self.assertRaises(ValueError):
            AgData(bodysite=self.bodysite,
                   trim=self.trim,
                   depth='5k')

    def test__init__dir_error(self):
        no_repo = os.path.join(self.repo_dir, 'LukeCageIsABadass')
        with self.assertRaises(ValueError):
            AgData(
                bodysite=self.bodysite,
                trim=self.trim,
                depth=self.depth,
                base_dir=no_repo,
                )

    def test_check_dataset(self):
        first_data_set = {'bodysite': self.bodysite,
                          'sequence_trim': self.trim,
                          'rarefaction_depth': self.depth,
                          'samples_per_participants': 'all_samples',
                          'participant_set': 'all'
                          }
        first_data_dir = os.path.join(self.base_dir,
                                      'fecal/100nt/all_participants/'
                                      'all_samples/10k')

        after_data_set = {'bodysite': self.bodysite,
                          'sequence_trim': self.trim,
                          'rarefaction_depth': '1k',
                          'samples_per_participants': 'all_samples',
                          'participant_set': 'all'
                          }
        after_data_dir = os.path.join(self.base_dir,
                                      'fecal/100nt/all_participants/'
                                      'all_samples/1k')
        self.assertEqual(first_data_set, self.ag_data.data_set)
        self.assertEqual(first_data_dir, self.ag_data.data_dir)

        self.ag_data.depth = '1k'
        self.ag_data._check_dataset()
        self.assertEqual(after_data_set, self.ag_data.data_set)
        self.assertEqual(after_data_dir, self.ag_data.data_dir)

    def test_check_dataset_loc_error(self):
        if os.path.exists(os.path.join(self.base_dir, 'skin')):
            os.path.rmdir(os.path.join(self.base_dir, 'skin'))
        self.ag_data.bodysite = 'skin'
        with self.assertRaises(ValueError):
            self.ag_data._check_dataset()

    def test_reload_files(self):
        # Sets up the known map data
        map_data = np.array([
            ['60', 'Daily', 'false', 'true', '25.13', 'Overweight',
             'I do not have this condition', '2.866733', '2.49309705457',
             '9.15', '7.6'],
            ['57.0', 'Occasionally (1-2 times/week)', 'false', 'true', '26.73',
             'Overweight', 'I do not have this condition', '2.198198',
             '1.83213159668', '5.65', '5.1'],
            ['34.0', 'Rarely (a few times/month)', 'false', 'true', '19.27',
             'Normal', 'None', '1.932658', '0.970334630884', '4.05',
             '3.9']
            ], dtype=object)
        columns = pd.Index([
            'AGE_YEARS', 'ALCOHOL_FREQUENCY', 'ALCOHOL_TYPES_BEERCIDER',
            'ALCOHOL_TYPES_UNSPECIFIED', 'BMI', 'BMI_CAT', 'IBD',
            'PD_whole_tree_1k', 'shannon_1k', 'chao1_1k', 'observed_otus_1k'
            ], dtype='object')
        index = pd.Index(['10317.000007108', '10317.000005844',
                          '10317.000002035'], dtype='object', name='#SampleID')
        alpha_cols = ['PD_whole_tree_1k', 'shannon_1k', 'chao1_1k',
                      'observed_otus_1k']
        known_map = pd.DataFrame(map_data, columns=columns, index=index)
        known_map.replace('None', np.nan, inplace=True)
        known_map[alpha_cols] = known_map[alpha_cols].astype(float)

        # Sets up known biom table
        otu_data = np.array([[05.,  5., 16.],
                             [01.,  0.,  0.],
                             [00.,  5.,  1.],
                             [00.,  1.,  1.],
                             [43., 11., 21.],
                             [00.,  2.,  3.],
                             [00., 14.,  0.],
                             [00.,  1.,  0.],
                             [01., 11.,  8.]])
        otu_ids = np.array([u'4447072', u'4477696', u'4352657', u'4446058',
                            u'4468234', u'1000986', u'4481131', u'4479989',
                            u'2532173'], dtype=object)
        sample = np.array([u'10317.000002035', u'10317.000007108',
                           u'10317.000005844'], dtype=object)
        known_otu = biom.Table(otu_data, otu_ids, index)

        # Sets up known distance matrices
        unweighted = np.array([[0.00000000,  0.53135624,  0.42956480],
                               [0.53135624,  0.00000000,  0.20605718],
                               [0.42956480,  0.20605718,  0.00000000]])
        weighted = np.array([[0.0000000,  0.9016606,  0.3588304],
                             [0.9016606,  0.0000000,  0.5709402],
                             [0.3588304,  0.5709402,  0.0000000]])
        beta = {
            'unweighted_unifrac':
            skbio.DistanceMatrix(data=unweighted, ids=('10317.000002035',
                                                       '10317.000007108',
                                                       '10317.000005844')),
            'weighted_unifrac':
            skbio.DistanceMatrix(data=weighted, ids=('10317.000002035',
                                                     '10317.000007108',
                                                     '10317.000005844'))
            }

        # Checks the 10k data
        pdt.assert_frame_equal(self.map_, self.ag_data.map_)

        npt.assert_array_equal(self.otu_.ids('observation'),
                               self.ag_data.otu_.ids('observation'))
        npt.assert_array_equal(self.otu_.ids('sample'),
                               self.ag_data.otu_.ids('sample'))
        for otu_id in self.otu_.ids('observation'):
            npt.assert_array_equal(self.otu_.data(otu_id, axis='observation'),
                                   self.ag_data.otu_.data(otu_id,
                                                          axis='observation'))

        self.assertEqual(self.beta.keys(), self.ag_data.beta.keys())
        for metric in self.ag_data.beta.keys():
            npt.assert_almost_equal(self.beta[metric].data,
                                    self.ag_data.beta[metric].data, 5)
            self.assertEqual(self.beta[metric].ids,
                             self.ag_data.beta[metric].ids)

        # Reloads the data at 1k
        self.ag_data.depth = '1k'
        self.ag_data.reload_files()

        # Checks the 1k data
        pdt.assert_frame_equal(known_map, self.ag_data.map_)

        npt.assert_array_equal(otu_ids, self.ag_data.otu_.ids('observation'))
        npt.assert_array_equal(sample, self.ag_data.otu_.ids('sample'))
        for otu_id in otu_ids:
            npt.assert_array_equal(known_otu.data(otu_id, axis='observation'),
                                   self.ag_data.otu_.data(otu_id,
                                                          axis='observation'))

        self.assertEqual(beta.keys(), self.ag_data.beta.keys())
        for metric in self.ag_data.beta.keys():
            npt.assert_almost_equal(beta[metric].data,
                                    self.ag_data.beta[metric].data, 5)
            self.assertEqual(beta[metric].ids, self.ag_data.beta[metric].ids)

    def test_drop_alpha_outliers(self):
        outliers = np.array(['10317.000007108'], dtype='object')
        keep = ['10317.000005844', '10317.000002035']
        keep_map = self.map_.loc[keep]
        keep_otu = self.otu_.filter(keep)
        keep_beta = {k: self.beta[k].filter(keep) for k in self.beta.keys()}

        self.ag_data.drop_alpha_outliers(limit=2.5)

        # Checks the data
        pdt.assert_frame_equal(keep_map, self.ag_data.map_)

        npt.assert_array_equal(keep_otu.ids('observation'),
                               self.ag_data.otu_.ids('observation'))
        npt.assert_array_equal(keep_otu.ids('sample'),
                               self.ag_data.otu_.ids('sample'))
        for otu_id in keep_otu.ids('observation'):
            npt.assert_array_equal(keep_otu.data(otu_id, axis='observation'),
                                   self.ag_data.otu_.data(otu_id,
                                                          axis='observation'))

        self.assertEqual(self.beta.keys(), self.ag_data.beta.keys())
        for metric in self.ag_data.beta.keys():
            npt.assert_almost_equal(keep_beta[metric].data,
                                    self.ag_data.beta[metric].data, 5)
            self.assertEqual(keep_beta[metric].ids,
                             self.ag_data.beta[metric].ids)

    def test_drop_alpha_outliers_error(self):
        with self.assertRaises(ValueError):
            self.ag_data.drop_alpha_outliers('Killgrave')

    def test_drop_bmi_outlier(self):
        keep = ['10317.000007108', '10317.000002035']
        discard = ['10317.000005844']
        self.map_['BMI'] = self.map_['BMI'].astype(float)
        self.map_['BMI_CORRECTED'] = self.map_.loc[keep, 'BMI']
        self.map_.loc[discard, 'BMI_CAT'] = np.nan

        self.ag_data.drop_bmi_outliers(bmi_limits=[5, 26])

        # Checks the data
        pdt.assert_frame_equal(self.map_, self.ag_data.map_)

    def test_return_dataset(self):
        keep = ['10317.000007108', '10317.000005844']
        keep_map = self.map_.loc[keep]
        keep_otu = self.otu_.filter(keep)
        keep_beta = {k: self.beta[k].filter(keep) for k in self.beta.keys()}

        group = AgClinical(
            name='IBD',
            description=('Has the participant been diagnosed with IBD?'),
            )
        map_, otu_, beta = self.ag_data.return_dataset(group)

        # Checks the data
        pdt.assert_frame_equal(keep_map, map_)

        npt.assert_array_equal(keep_otu.ids('observation'),
                               otu_.ids('observation'))
        npt.assert_array_equal(keep_otu.ids('sample'),
                               otu_.ids('sample'))
        for otu_id in keep_otu.ids('observation'):
            npt.assert_array_equal(keep_otu.data(otu_id, axis='observation'),
                                   otu_.data(otu_id, axis='observation'))

        self.assertEqual(self.beta.keys(), beta.keys())
        for metric in self.ag_data.beta.keys():
            npt.assert_almost_equal(keep_beta[metric].data,
                                    beta[metric].data, 5)
            self.assertEqual(keep_beta[metric].ids, beta[metric].ids)


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
        self.assertEqual(test.remap_, None)

    def test_init_kwargs(self):
        test = AgQuestion(self.name, self.description, self.dtype,
                          clean_name='This is a test.',
                          free_response=True,
                          mimmarks=True,
                          ontology='GAZ',
                          remap=self.fun,
                          )

        self.assertEqual(test.name, self.name)
        self.assertEqual(test.description, self.description)
        self.assertEqual(test.dtype, self.dtype)
        self.assertEqual(test.clean_name, 'This is a test.')
        self.assertEqual(test.free_response, True)
        self.assertEqual(test.mimmarks, True)
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
        self.map_.replace([''], np.nan, inplace=True)
        self.map_.replace(['None'], np.nan, inplace=True)
        self.map_.replace(['nan'], np.nan, inplace=True)

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

        self.assertEqual(set(self.map_[self.ag_categorical.name]) - {np.nan},
                         {"I tend to be constipated (have difficulty "
                          "passing stool)", "I tend to have diarrhea "
                          "(watery stool)", "I tend to have normal formed "
                          "stool"})

    def test_ag_categorical_remap_groups(self):
        self.ag_categorical.remap_groups(self.map_)

        self.assertEqual(set(self.map_[self.ag_categorical.name]) - {np.nan},
                         {'No Reference', 'Well formed', 'Constipated',
                          'Diarrhea'})

        self.assertEqual(self.ag_categorical.earlier_order,
                         [self.order, self.order])

    def test_ag_categorical_drop_infrequent(self):
        self.ag_categorical.frequency_cutoff = 4
        self.ag_categorical.drop_infrequent(self.map_)

        self.assertEqual(set(self.map_[self.ag_categorical.name]) - {np.nan}, {
            "I tend to have normal formed stool"
            })

    def test_ag_clinical_remap_strict(self):
        self.ag_clinical.remap_clinical(self.map_)

        count = pd.Series([14, 4],
                          index=pd.Index(['No', 'Yes'], name='IBD'),
                          dtype=np.int64)
        pdt.assert_series_equal(count, self.map_.groupby('IBD').count().max(1))

    def test_ag_clincial_remap(self):
        self.ag_clinical.strict = False
        self.ag_clinical.remap_clinical(self.map_)

        count = pd.Series([14, 5],
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
