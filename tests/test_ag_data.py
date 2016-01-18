import os
from unittest import TestCase, main

import biom
import numpy as np
import pandas as pd
import numpy.testing as npt
import pandas.util.testing as pdt

import americangut as ag
from americangut.ag_data import *


class AgDataTest(TestCase):
    def setUp(self):
        self.repo_dir = ag.WORKING_DIR.split('American-Gut')[0]
        self.base_dir = os.path.join(self.repo_dir,
                                     'American-Gut/tests/data/11-packaged')
        self.bodysite = 'fecal'
        self.trim = '100nt'
        self.depth = '1k'

        self.ag_data = AgData(
            bodysite=self.bodysite,
            trim=self.trim,
            depth=self.depth,
            base_dir=self.base_dir
            )

        data_dir = os.path.join(self.base_dir,
                                'fecal/100nt/all_participants/all_samples/1k/')
        known_map = pd.read_csv(os.path.join(data_dir, 'ag_1k_fecal.txt'),
                                sep='\t',
                                dtype=str,
                                na_values=['NA', 'unknown', '', 'no_data',
                                           'None', 'Unknown'],
                                ).set_index('#SampleID')
        alpha_cols = ['PD_whole_tree_1k', 'shannon_1k', 'chao1_1k',
                      'observed_otus_1k']
        known_map[alpha_cols] = known_map[alpha_cols].astype(float)
        self.map_ = known_map
        self.otu_ = biom.load_table(os.path.join(data_dir, 'ag_1k_fecal.biom'))
        self.beta = {
            'unweighted_unifrac': skbio.DistanceMatrix.read(
                os.path.join(data_dir, 'unweighted_unifrac_ag_1k_fecal.txt')
                ),
            'weighted_unifrac': skbio.DistanceMatrix.read(
                os.path.join(data_dir, 'weighted_unifrac_ag_1k_fecal.txt')
            )}

    def test__init__(self):
        known_data_set = {'bodysite': self.bodysite,
                          'sequence_trim': self.trim,
                          'rarefaction_depth': self.depth,
                          'samples_per_participants': 'all_samples',
                          'participant_set': 'all'
                          }

        test = AgData(
            bodysite=self.bodysite,
            trim=self.trim,
            depth=self.depth,
            base_dir=self.base_dir
            )
        # Checks dataset
        self.assertEqual(test.data_set, known_data_set)
        self.assertEqual(test.beta, self.beta)
        pdt.assert_frame_equal(test.map_, self.map_)
        npt.assert_array_equal(test.otu_.ids('sample'),
                               self.otu_.ids('sample'))
        npt.assert_array_equal(test.otu_.ids('observation'),
                               self.otu_.ids('observation'))
        for sample in test.otu_.ids('sample'):
            npt.assert_array_equal(test.otu_.data(sample),
                                   self.otu_.data(sample))

    def test__init__subset(self):
        data_dir = os.path.join(self.base_dir,
                                'fecal/100nt/sub_participants/one_sample/10k/')
        known_map = pd.read_csv(os.path.join(data_dir, 'ag_10k_fecal.txt'),
                                sep='\t',
                                dtype=str,
                                na_values=['NA', 'unknown', '', 'no_data',
                                           'None', 'Unknown'],
                                ).set_index('#SampleID')
        alpha_cols = ['PD_whole_tree_1k', 'PD_whole_tree_10k',
                      'shannon_1k', 'shannon_10k',
                      'chao1_1k', 'chao1_10k',
                      'observed_otus_1k', 'observed_otus_10k']
        known_map[alpha_cols] = known_map[alpha_cols].astype(float)
        known_otu = biom.load_table(os.path.join(data_dir,
                                    'ag_10k_fecal.biom'))
        known_beta = {
            'unweighted_unifrac': skbio.DistanceMatrix.read(
                os.path.join(data_dir, 'unweighted_unifrac_ag_10k_fecal.txt')
                ),
            'weighted_unifrac': skbio.DistanceMatrix.read(
                os.path.join(data_dir, 'weighted_unifrac_ag_10k_fecal.txt')
            )}
        known_data_set = {'bodysite': self.bodysite,
                          'sequence_trim': self.trim,
                          'rarefaction_depth': '10k',
                          'samples_per_participants': 'one_sample',
                          'participant_set': 'sub'
                          }

        test = AgData(
            bodysite=self.bodysite,
            trim=self.trim,
            depth='10k',
            one_sample=True,
            sub_participants=True,
            base_dir=self.base_dir
            )
        # Checks dataset
        self.assertEqual(test.data_set, known_data_set)
        self.assertEqual(test.beta, known_beta)
        pdt.assert_frame_equal(test.map_, known_map)
        npt.assert_array_equal(test.otu_.ids('sample'),
                               known_otu.ids('sample'))
        npt.assert_array_equal(test.otu_.ids('observation'),
                               known_otu.ids('observation'))
        for sample in test.otu_.ids('sample'):
            npt.assert_array_equal(test.otu_.data(sample),
                                   known_otu.data(sample))

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

    def test_reload_files(self):
        # Loads the files for the final version
        data_dir = os.path.join(self.base_dir,
                                'skin/100nt/all_participants/all_samples/1k/')
        known_map = pd.read_csv(os.path.join(data_dir, 'ag_1k_skin.txt'),
                                sep='\t',
                                dtype=str,
                                na_values=['NA', 'unknown', '', 'no_data',
                                           'None', 'Unknown'],
                                ).set_index('#SampleID')
        alpha_cols = ['PD_whole_tree_1k', 'shannon_1k', 'chao1_1k',
                      'observed_otus_1k']
        known_map[alpha_cols] = known_map[alpha_cols].astype(float)
        known_otu = biom.load_table(os.path.join(data_dir,
                                    'ag_1k_skin.biom'))
        known_beta = {
            'unweighted_unifrac': skbio.DistanceMatrix.read(
                os.path.join(data_dir, 'unweighted_unifrac_ag_1k_skin.txt')
                ),
            'weighted_unifrac': skbio.DistanceMatrix.read(
                os.path.join(data_dir, 'weighted_unifrac_ag_1k_skin.txt')
            )}

        # Tests the fecal files are loaded
        self.assertEqual(self.ag_data.beta, self.beta)
        pdt.assert_frame_equal(self.ag_data.map_, self.map_)
        npt.assert_array_equal(self.ag_data.otu_.ids('sample'),
                               self.otu_.ids('sample'))
        npt.assert_array_equal(self.ag_data.otu_.ids('observation'),
                               self.otu_.ids('observation'))
        for sample in self.ag_data.otu_.ids('sample'):
            npt.assert_array_equal(self.ag_data.otu_.data(sample),
                                   self.otu_.data(sample))

        # Changes the bodysite
        self.ag_data.bodysite = 'skin'
        self.ag_data.reload_files()

        # Checks dataset
        self.assertEqual(self.ag_data.beta, known_beta)
        pdt.assert_frame_equal(self.ag_data.map_, known_map)
        npt.assert_array_equal(self.ag_data.otu_.ids('sample'),
                               known_otu.ids('sample'))
        npt.assert_array_equal(self.ag_data.otu_.ids('observation'),
                               known_otu.ids('observation'))
        for sample in self.ag_data.otu_.ids('sample'):
            npt.assert_array_equal(self.ag_data.otu_.data(sample),
                                   known_otu.data(sample))

if __name__ == '__main__':
    main()
