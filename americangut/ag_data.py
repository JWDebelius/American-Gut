import os

import biom
import numpy as np
import pandas as pd
import skbio


class AgData:
    na_values = ['NA', 'unknown', '', 'no_data', 'None', 'Unknown']
    alpha_columns = {'10k': ['PD_whole_tree_1k', 'PD_whole_tree_10k',
                             'shannon_1k', 'shannon_10k',
                             'chao1_1k', 'chao1_10k',
                             'observed_otus_1k', 'observed_otus_10k'],
                     '1k': ['PD_whole_tree_1k', 'shannon_1k', 'chao1_1k',
                            'observed_otus_1k']}
    bodysite_limits = {'fecal': 55,
                       }
    subset_columns = {'IBD': 'SUBSET_IBD',
                      'DIABETES': 'SUBSET_DIABETES',
                      'BMI_CAT': 'SUBSET_BMI',
                      'ANTIBIOTIC_HISTORY': 'SUBSET_ANTIBIOTIC_HISTORY',
                      'AGE_CAT': 'SUBSET_AGE'}

    def __init__(self, bodysite, trim, depth, sub_participants=False,
                 one_sample=False, **kwargs):

        # Checks the inputs
        if bodysite not in {'fecal', 'oral', 'skin'}:
            raise ValueError('%s is not a supported bodysite.' % bodysite)
        if trim not in {'notrim', '100nt'}:
            raise ValueError('%s is not a supported trim length.' % trim)
        if depth not in {'1k', '10k'}:
            raise ValueError('%s is not a supported rarefaction depth' % depth)

        self.bodysite = bodysite
        self.trim = trim
        self.depth = depth
        self.sub_participants = sub_participants
        self.one_sample = one_sample

        self._check_dataset()

        # Gets the base directory
        base_dir = kwargs.get('base_dir',
                              os.path.abspath('../primary-processing/'
                                              'agp_processing/'
                                              '11-packaged'))

        # Gets the data directory
        self.data_dir = os.path.join(base_dir,
                                     '%(bodysite)s/'
                                     '%(sequence_trim)s/'
                                     '%(participant_set)s_participants/'
                                     '%(samples_per_participants)s/'
                                     '%(rarefaction_depth)s'
                                     % self.data_set
                                     )
        if not os.path.exists(self.data_dir):
            raise ValueError('The identified dataset does not exist')

        self.reload_files()

    def _check_dataset(self):
        self.data_set = {'bodysite': self.bodysite,
                         'sequence_trim': self.trim,
                         'rarefaction_depth': self.depth,
                         }

        if self.one_sample:
            self.data_set['samples_per_participants'] = 'one_sample'
        else:
            self.data_set['samples_per_participants'] = 'all_samples'

        if self.sub_participants:
            self.data_set['participant_set'] = 'sub'
        else:
            self.data_set['participant_set'] = 'all'

    def reload_files(self):
        self._check_dataset()

        # Determines the filepaths
        map_fp = os.path.join(self.data_dir,
                              'ag_%(rarefaction_depth)s_%(bodysite)s.txt'
                              % self.data_set)
        otu_fp = os.path.join(self.data_dir,
                              'ag_%(rarefaction_depth)s_%(bodysite)s.biom'
                              % self.data_set)
        unweighted_fp = os.path.join(self.data_dir,
                                     'unweighted_unifrac_ag_'
                                     '%(rarefaction_depth)s_%(bodysite)s.txt'
                                     % self.data_set)
        weighted_fp = os.path.join(self.data_dir,
                                   'weighted_unifrac_ag_%(rarefaction_depth)s'
                                   '_%(bodysite)s.txt'
                                   % self.data_set)

        self.map_ = pd.read_csv(map_fp,
                                sep='\t',
                                dtype=str,
                                na_values=self.na_values)
        self.map_.set_index('#SampleID', inplace=True)
        columns = self.alpha_columns[self.data_set['rarefaction_depth']]
        self.map_[columns] = self.map_[columns].astype(float)

        self.otu_ = biom.load_table(otu_fp)

        self.beta = {'unweighted_unifrac':
                     skbio.DistanceMatrix.read(unweighted_fp),
                     'weighted_unifrac':
                     skbio.DistanceMatrix.read(weighted_fp)
                     }

    def drop_alpha_outliers(self, metric='PD_whole_tree_10k', limit=55):
        self.outliers = self.map_.loc[(self.map_[metric] >= limit)].index
        keep = self.map_.loc[(self.map_[metric] < limit)].index
        self.map_ = self.map_.loc[keep]
        self.otu_ = self.otu_.filter(keep)
        self.beta = {'unweighted_unifrac':
                     self.beta['unweighted_unifrac'].filter(keep),
                     'weighted_unifrac':
                     self.beta['weighted_unifrac'].filter(keep)
                     }

    def drop_bmi_outliers(self, bmi_limits=[5, 55]):
        self.map_['BMI'] = self.map_['BMI'].astype(float)
        keep = ((self.map_.BMI > bmi_limits[0]) &
                (self.map_.BMI < bmi_limits[1]))
        discard = ~ keep
        self.map_['BMI_CORRECTED'] = self.map_.loc[keep, 'BMI']
        self.map_.loc[discard, 'BMI_CAT'] = np.nan

    def cast_subset(self):
        pass

    # def return_dataset(self, group):
    #     """..."""
    #     defined = self.map_[~ pd.isnull(self.map_[group.name])].index
    #     map_ = self.map_.loc[defined]
    #     otu_ = self.otu_.filter(defined, axis='sample')
    #     beta = {k: v.filter(defined) for k, v in self.beta.iteritems()}
    #     return map_, otu_, beta

    def clean_up_column(self, group):
        """Cleans up the indicated mapping column using the group"""

        datatype = group.type
        group.remap_data_type(self.map_)

        if datatype in {'Continous'}:
            group.drop_outliers(self.map_)
        elif datatype in {'Categorical', 'Multiple', 'Clinical', 'Frequency'}:
            group.remove_ambiguity(self.map_)
            group.drop_infrequent(self.map_)
            group.remap_groups(self.map_)

        if datatype in {'Clincial'}:
            group.remap_clincial(self.map_)
        elif datatype == 'Multiple':
            group.correct_unspecified(self.map_)
        elif datatype == 'Frequency':
            group.clean(self.map_)
            
