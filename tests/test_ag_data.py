import os
from unittest import TestCase, main

import biom
import numpy as np
import pandas as pd
import numpy.testing as npt
import pandas.util.testing as pdt
import skbio

import americangut as ag
from americangut.ag_data import AgData
from americangut.question.ag_bool import (AgBool,
                                          AgMultiple,
                                          )
from americangut.question.ag_categorical import (AgCategorical,
                                                 AgClinical,
                                                 AgFrequency,
                                                 )
from americangut.question.ag_continous import AgContinous

amgut_sub = np.array([
    ['10317.000006668', '71.0', 'Never', 'false', 'true', '24.61', 'Normal',
     'I do not have this condition', '14.340194', '28.045005', '5.02856010656',
     '5.23963694715', '251.873888011', '531.783308576', '133.1', '366.2',
     'false', 'true', 'true', 'true', 'true', ''],
    ['10317.000030344', '40.0', 'Regularly (3-5 times/week)', 'false', 'false',
     '21.15', 'Normal', 'I do not have this condition', '17.52718',
     '35.567157', '5.68192485378', '5.88986104793', '315.738586263',
     '767.278538734', '155.4', '459.8', 'true', 'false', 'true', 'true',
     'true', 'I tend to have normal formed stool'],
    ['10317.000031833', '76.0', 'Daily', 'true', 'false', '28.35',
     'Overweight', 'I do not have this condition', '23.138257', '44.269159',
     '6.45463332504', '6.7301856679', '362.525618183', '927.222520687',
     '200.6', '581.8', 'false', 'true', 'true', 'true', 'true',
     'I tend to have normal formed stool'],
    ['10317.000013134', '45.0', 'Regularly (3-5 times/week)', 'false', 'true',
     '23.51', 'Normal', '', '19.004077', '34.987892', '5.8177687212',
     '6.04825725462', '283.688480584', '621.790214285', '153.5',
     '419.1', 'true', 'false', 'true', 'true', 'false', ''],
    ['10317.000022634', '40.0', 'Never', 'false', 'true', '22.24', 'Normal',
     'I do not have this condition', '12.757604', '24.35038', '4.18196692357',
     '4.34957998607', '195.708672633', '481.450659304', '109.5', '306.7',
     'true', 'true', 'true', 'true', 'true', 'I tend to be constipated '
     '(have difficulty passing stool)'],
    ['10317.000020683', '34.0', 'Regularly (3-5 times/week)', 'true', 'false',
     '23.36', 'Normal', 'I do not have this condition', '15.938636',
     '30.445418', '4.69301334505', '4.8805030057', '256.001215824',
     '470.162645729', '130.8', '349.5', 'true', 'false', 'true', 'true',
     'true', "I don't know, I do not have a point of reference"],
    ['10317.000022916', '77.0', 'Rarely (a few times/month)', 'true', 'false',
     '30.41', 'Obese', 'I do not have this condition', '16.846201',
     '34.052185', '5.76200535747', '5.98830200759', '317.03496711',
     '775.942491501', '161.4', '481.6', 'false', 'false', 'false', 'true',
     'true', "I don't know, I do not have a point of reference"],
    ['10317.000031363', '45.0', 'Occasionally (1-2 times/week)', 'true',
     'false', '20.16', 'Normal', 'I do not have this condition', '15.329904',
     '28.674869', '5.56385250945', '5.76521128359', '224.719925507',
     '689.432707101', '134.1', '374.1', 'true', 'true', 'true', 'true',
     'true', 'I tend to be constipated (have difficulty passing stool)'],
    ['10317.000006952', '33.0', 'Rarely (a few times/month)', 'false', 'true',
     '21.8', 'Normal', 'I do not have this condition', '16.478963',
     '29.894423', '5.56200604556', '5.7662879992', '248.312830767',
     '510.61022798', '140.8', '370.0', 'true', 'false', 'true', 'true', 'true',
     ''],
    ['10317.000027363', '52.0', 'Rarely (a few times/month)', 'true', 'false',
     '21.93', 'Normal', 'I do not have this condition', '17.552066',
     '35.073292', '5.51344459713', '5.73262282623', '283.392113931',
     '818.052897634', '152.0', '452.3', 'true', 'true', 'true', 'true',
     'true', ''],
    ['10317.000002337', '57.0', 'Daily', 'false', 'true', '22.72', 'Normal',
     'I do not have this condition', '11.320356', '22.440616',
     '4.67832361812', '4.84110706669', '168.043870321', '426.056128298',
     '100.2', '279.5', 'true', 'false', 'true', 'false', 'true', ''],
    ['10317.000027574', '6.0', 'Never', 'false', 'true', '15.48',
     'Underweight', 'I do not have this condition', '12.825713', '22.999707',
     '4.77128567758', '4.89638844086', '166.111886447', '420.433247276',
     '105.0', '268.6', 'false', 'true', 'false', 'true', 'true',
     'I tend to have normal formed stool'],
    ['10317.000003365', '26.0', 'Regularly (3-5 times/week)', 'false', 'true',
     '20.86', 'Normal', 'I do not have this condition', '11.413715',
     '21.602869', '4.34580753681', '4.49314030837', '190.522332339',
     '392.257705499', '110.2', '285.9', 'true', 'true', 'true', 'true',
     'true', ''],
    ['10317.000002347', '45.0', 'Occasionally (1-2 times/week)', 'false',
     'true', '25.71', 'Overweight', 'I do not have this condition',
     '13.129761', '25.323507', '4.42378118684', '4.58755771612',
     '200.751420094', '454.103603971', '105.0', '300.9', 'true', 'true',
     'true', 'true', 'true', ''],
    ['10317.000011322', '', 'Regularly (3-5 times/week)', 'false', 'true',
     '34.16', 'Obese', 'I do not have this condition', '13.600805',
     '25.763357', '4.35932913512', '4.47382610643', '215.597203752',
     '512.739207256', '118.2', '330.8', 'false', 'false', 'false', 'true',
     'true', ''],
    ['10317.000020053', '35.0', 'Occasionally (1-2 times/week)', 'true',
     'false', '22.12', 'Normal', 'Self-diagnosed', '11.637385', '25.926528',
     '3.30051047459', '3.44538552585', '193.070246388', '540.825339832',
     '92.5', '314.7', 'true', 'true', 'true', 'true', 'false',
     'I tend to have normal ' 'formed stool'],
    ['10317.000029572', '65.0', 'Daily', 'false', 'false', '26.31',
     'Overweight', 'Diagnosed by a medical professional (doctor, physician '
     'assistant)', '25.204442', '48.174941', '6.71316811398', '7.04213735924',
     '381.901710119', '983.028749379', '222.8', '614.7', 'true', 'false',
     'true', 'true', 'false', 'I tend to be constipated (have difficulty '
     'passing stool)'],
    ['10317.000023046', '14.0', 'Never', 'false', 'true', '18.6', 'Normal',
     'Diagnosed by a medical professional (doctor, physician assistant)',
     '12.123028', '24.986377', '4.82881317767', '5.00842790392',
     '212.767995747', '601.601496282', '114.2', '351.0', 'false', 'true',
     'true', 'true', 'false', 'I tend to have normal formed stool'],
    ['10317.000028154', '54.0', 'Occasionally (1-2 times/week)', 'true',
     'false', '21.72', 'Normal', 'Diagnosed by a medical professional '
     '(doctor, physician assistant)', '17.004322', '31.447394',
     '5.75849162154', '5.95677948173', '250.100161474', '589.906183066',
     '149.6', '377.1', 'true', 'false', 'true', 'true', 'false',
     'I tend to have diarrhea (watery stool)'],
    ['10317.000020499', '33.0', 'Rarely (a few times/month)', 'true', 'false',
     '23.48', 'Normal', 'Diagnosed by a medical professional (doctor, '
     'physician ' 'assistant)', '11.597293', '24.977299', '3.70846915327',
     '3.84415524862', '214.889997618', '381.499716075', '99.9', '293.9',
     'true', 'true', 'true', 'true', 'false', 'I tend to have diarrhea '
     '(watery stool)'],
    ])

columns = pd.Index([
    '#SampleID',  'AGE_YEARS', 'ALCOHOL_FREQUENCY', 'ALCOHOL_TYPES_BEERCIDER',
    'ALCOHOL_TYPES_UNSPECIFIED', 'BMI', 'BMI_CAT', 'IBD',
    'PD_whole_tree_1k', 'PD_whole_tree_10k', 'shannon_1k',
    'shannon_10k', 'chao1_1k', 'chao1_10k', 'observed_otus_1k',
    'observed_otus_10k', 'SUBSET_AGE', 'SUBSET_ANTIBIOTIC_HISTORY',
    'SUBSET_BMI', 'SUBSET_DIABETES', 'SUBSET_IBD', 'BOWEL_MOVEMENT_QUALITY'],
    dtype='object')


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
              '2.49309705457', '2.37419183588', '9.15', '9.7', '7.6', '8.0',
              'true', 'false', 'true', 'true', 'true'],
             ['57.0', 'Occasionally (1-2 times/week)', 'false', 'true',
              '26.73', 'Overweight', 'I do not have this condition',
              '2.198198', '2.446593', '1.83213159668', '1.91370771085', '5.65',
              '6.3', '5.1', '5.9', 'true', 'true', 'true', 'true', 'true'],
             ['34.0', 'Rarely (a few times/month)', 'false', 'true', '19.27',
              'Normal', 'None', '1.932658', '2.012048', '0.970334630884',
              '1.0890945893', '4.05', '4.2', '3.9', '4.2', 'false', 'true',
              'false', 'true', 'true']],
            dtype=object)
        columns = pd.Index([
            'AGE_YEARS', 'ALCOHOL_FREQUENCY', 'ALCOHOL_TYPES_BEERCIDER',
            'ALCOHOL_TYPES_UNSPECIFIED', 'BMI', 'BMI_CAT', 'IBD',
            'PD_whole_tree_1k', 'PD_whole_tree_10k', 'shannon_1k',
            'shannon_10k', 'chao1_1k', 'chao1_10k', 'observed_otus_1k',
            'observed_otus_10k', 'SUBSET_AGE', 'SUBSET_ANTIBIOTIC_HISTORY',
            'SUBSET_BMI', 'SUBSET_DIABETES', 'SUBSET_IBD',
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
        self.otu_ids = np.array([u'4447072', u'4352657', u'4468234',
                                 u'1000986', u'4481131', u'4479989',
                                 u'2532173'], dtype=object)
        sample = np.array([u'10317.000002035', u'10317.000007108',
                           u'10317.000005844'], dtype=object)
        self.otu_ = biom.Table(otu_data, self.otu_ids, sample)

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
        otu_ids = np.array([u'4447072', u'4477696', u'4352657',
                            u'4446058', u'4468234', u'1000986',
                            u'4481131', u'4479989', u'2532173'],
                            dtype=object)
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

    def test_filter_inplace(self):
        ids = ['10317.000002035']
        self.ag_data.filter(ids)
        pdt.assert_frame_equal(self.map_.loc[ids], self.ag_data.map_)

        npt.assert_array_equal(self.otu_ids,
                               self.ag_data.otu_.ids('observation'))
        npt.assert_array_equal(ids, self.ag_data.otu_.ids('sample'))
        for otu_id in self.otu_ids:
            npt.assert_array_equal(
                self.otu_.filter(ids).data(otu_id, axis='observation'),
                self.ag_data.otu_.data(otu_id, axis='observation')
                )

        self.assertEqual(self.beta.keys(), self.ag_data.beta.keys())
        for metric in self.ag_data.beta.keys():
            npt.assert_almost_equal(self.beta[metric].filter(ids).data,
                                    self.ag_data.beta[metric].data, 5)
            self.assertEqual(self.beta[metric].filter(ids).ids,
                             self.ag_data.beta[metric].ids)

    def test_filter_return(self):
        ids = ['10317.000002035']
        map_, otu_, beta = self.ag_data.filter(ids, inplace=False)
        pdt.assert_frame_equal(self.map_.loc[ids], map_)

        npt.assert_array_equal(self.otu_ids,
                               otu_.ids('observation'))
        npt.assert_array_equal(ids, otu_.ids('sample'))
        for otu_id in self.otu_ids:
            npt.assert_array_equal(
                self.otu_.filter(ids).data(otu_id, axis='observation'),
                otu_.data(otu_id, axis='observation')
                )

        self.assertEqual(self.beta.keys(), beta.keys())
        for metric in beta.keys():
            npt.assert_almost_equal(self.beta[metric].filter(ids).data,
                                    beta[metric].data, 5)
            self.assertEqual(self.beta[metric].filter(ids).ids,
                             beta[metric].ids)

    def test_drop_alpha_outliers(self):
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

    def test_filter_by_question(self):
        keep = ['10317.000007108', '10317.000005844']
        keep_map = self.map_.loc[keep]
        keep_otu = self.otu_.filter(keep)
        keep_beta = {k: self.beta[k].filter(keep) for k in self.beta.keys()}

        group = AgClinical(
            name='IBD',
            description=('Has the participant been diagnosed with IBD?'),
            )
        map_, otu_, beta = self.ag_data.filter_by_question(group,
                                                           inplace=False)

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

    def test_filter_subset(self):
        # Sets up the known index values after filtering
        known = pd.Index(['10317.000005844'],
                         dtype='object',
                         name=u'#SampleID')
        self.ag_data.filter_subset(inplace=True)
        self.assertEqual(self.ag_data.map_.index, known)

    def test_clean_group_continous(self):
        map_ = pd.DataFrame(
            amgut_sub,
            columns=columns,
            ).set_index('#SampleID')
        map_.replace('', np.nan, inplace=True)
        self.ag_data.map_ = map_

        question = AgContinous(name='AGE_YEARS',
                               description='Participant age in years',
                               unit='years',
                               range=[20, 50])

        self.assertEqual(
            self.ag_data.map_[question.name].dropna().astype(float).min(),
            6.0)
        self.assertEqual(
            self.ag_data.map_[question.name].dropna().astype(float).max(),
            77.0)

        self.ag_data.clean_group(question)

        self.assertEqual(self.ag_data.map_[question.name].dropna().min(),
                         26.0)
        self.assertEqual(self.ag_data.map_[question.name].dropna().max(),
                         45.0)
        self.assertEqual(np.sum(pd.isnull(self.ag_data.map_[question.name])),
                         10)

    def test_clean_group_bool(self):
        map_ = pd.DataFrame(
            amgut_sub,
            columns=columns,
            ).set_index('#SampleID')
        map_.replace('', np.nan, inplace=True)
        self.ag_data.map_ = map_

        question = AgBool(
            name='ALCOHOL_TYPES_UNSPECIFIED',
            description=('Has the participant described their alcohol use'),
            )

        known_order = [['true', 'false'], [True, False]]

        self.ag_data.clean_group(question)
        self.assertEqual(set(self.ag_data.map_[question.name]) - {np.nan},
                         {'yes', 'no'})
        self.assertEqual(question.earlier_order, known_order)

    def test_clean_group_categorical(self):
        map_ = pd.DataFrame(
            amgut_sub,
            columns=columns,
            ).set_index('#SampleID')
        map_.replace('', np.nan, inplace=True)
        self.ag_data.map_ = map_

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

        question = AgCategorical(
            name='BOWEL_MOVEMENT_QUALITY',
            description=('Whether the participant is constipated, has '
                         'diarrhea or is regular.'),
            dtype=str,
            order=["I don't know, I do not have a point of reference",
                   "I tend to have normal formed stool",
                   "I tend to be constipated (have difficulty passing stool)",
                   "I tend to have diarrhea (watery stool)",
                   ],
            extremes=["I tend to have normal formed stool",
                      "I tend to have diarrhea (watery stool)"],
            remap=remap_poo_quality,
            drop_ambigious=True,
            frequency_cutoff=4,
            )

        known_order = [["I don't know, I do not have a point of reference",
                        "I tend to have normal formed stool",
                        "I tend to be constipated (have difficulty passing "
                        "stool)",
                        "I tend to have diarrhea (watery stool)",
                        ],
                       ["I tend to have normal formed stool",
                        "I tend to be constipated (have difficulty passing "
                        "stool)",
                        "I tend to have diarrhea (watery stool)",
                        ],
                       ["Well formed", "Constipated", "Diarrhea"],
                       ]

        self.ag_data.clean_group(question)

        self.assertEqual(set(self.ag_data.map_[question.name]) - {np.nan},
                         {"Well formed"})

        self.assertEqual(question.earlier_order, known_order)

    def test_clean_group_multiple(self):
        map_ = pd.DataFrame(
            amgut_sub,
            columns=columns,
            ).set_index('#SampleID')
        map_.replace('', np.nan, inplace=True)
        self.ag_data.map_ = map_

        question = AgMultiple(
            name='ALCOHOL_TYPES_BEERCIDER',
            description=('The participant drinks beer or cider.'),
            unspecified='ALCOHOL_TYPES_UNSPECIFIED',
            )

        known_order = [['true', 'false'], [True, False]]
        self.assertEqual(np.sum(pd.isnull(self.ag_data.map_[question.name])),
                         0)
        self.ag_data.clean_group(question)

        self.assertEqual(np.sum(pd.isnull(self.ag_data.map_[question.name])),
                         10)

        self.assertEqual(question.earlier_order, known_order)

    def test_clean_group_clinical(self):
        map_ = pd.DataFrame(
            amgut_sub,
            columns=columns,
            ).set_index('#SampleID')
        map_.replace('', np.nan, inplace=True)
        self.ag_data.map_ = map_

        question = AgClinical(
            name='IBD',
            description=('Has the participant been diagnosed with IBD?'),
            strict=True
            )

        known_order = [
            ['Diagnosed by a medical professional (doctor, physician '
             'assistant)',
             'Diagnosed by an alternative medicine practitioner',
             'Self-diagnosed',
             'I do not have this condition'
             ],
            ]

        self.ag_data.clean_group(question)
        count = pd.Series([14, 4],
                          index=pd.Index(['No', 'Yes'], name='IBD'),
                          dtype=np.int64)
        pdt.assert_series_equal(
            count,
            self.ag_data.map_.groupby('IBD').count().max(1)
            )
        self.assertEqual(question.earlier_order, known_order)

    def test_clean_group_frequency(self):
        map_ = pd.DataFrame(
            amgut_sub,
            columns=columns,
            ).set_index('#SampleID')
        map_.replace('', np.nan, inplace=True)
        self.ag_data.map_ = map_

        known_order = [['Never', 'Rarely (a few times/month)',
                        'Occasionally (1-2 times/week)',
                        'Regularly (3-5 times/week)', 'Daily'],
                       ['Never', 'A few times/month', '1-2 times/week',
                        '3-5 times/week', 'Daily']
                       ]

        question = AgFrequency(
            name='ALCOHOL_FREQUENCY',
            description=('How often the participant consumes alcohol.'),
            combine='weekly',
            )

        self.ag_data.clean_group(question)

        self.assertEqual(set(self.ag_data.map_[question.name]) - {np.nan},
                         {'Never', 'A few times/month', "More than once/week"})
        self.assertEqual(question.earlier_order, known_order)

if __name__ == '__main__':
    main()
