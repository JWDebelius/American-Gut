import os

import biom
import numpy as np
import pandas as pd
import skbio


def _remap_bowel_movement_frequency(x):
    if x in {'Four', 'Five or more'}:
        return 'Four or more'
    else:
        return x


def _remap_breastmilk(x):
    if x in {'I eat both solid food and formula/breast milk', 'Yes'}:
        return 'Yes'
    else:
        return x


def _remap_contraceptive(x):
    if isinstance(x, str):
        return x.split(', ')[0]
    else:
        return x


def _remap_flu_date(x):
    if x in {'Week', 'Month'}:
        return 'Month'
    else:
        return x


def _remap_last_move(x):
    if x in {'Within the past month', 'Within the past 3 months'}:
        return 'Within the past 3 months'
    else:
        return x


def _remap_sex(x):
    if x in {'Other'}:
        return np.nan
    else:
        return x


def _remap_sleep(x):
    if x in {'Less than 5 hours', '5-6 hours'}:
        return 'Less than 6 hours'
    else:
        return x


class AgData:
    na_values = ['NA', 'unknown', '', 'no_data', 'None', 'Unknown']
    alpha_columns = {'PD_whole_tree_1k', 'PD_whole_tree_10k',
                     'shannon_1k', 'shannon_10k',
                     'chao1_1k', 'chao1_10k',
                     'observed_otus_1k', 'observed_otus_10k'}

    def __init__(self, bodysite, trim, depth, sub_participants=False,
                 one_sample=False):

        # Checks the inputs
        if bodysite not in {'fecal', 'oral', 'skin'}:
            raise ValueError('%s is not a supported bodysite.' % bodysite)
        if trim not in {'notrim', '100nt'}:
            raise ValueError('%s is not a supported trim length.' % trim)
        if depth not in {'1k', '10k'}:
            raise ValueError('%s is not a supported rarefaction depth' % depth)

        self.data_set = {'bodysite': bodysite,
                         'sequence_trim': trim,
                         'rarefaction_depth': depth,
                         }

        if one_sample:
            self.data_set['samples_per_participants'] = 'one_sample'
        else:
            self.data_set['samples_per_participants'] = 'all_samples'

        if sub_participants:
            self.data_set['participant_set'] = 'sub'
        else:
            self.data_set['participant_set'] = 'all'

        # Gets the base directory
        base_dir = os.path.abspath('../primary-processing/agp_processing/'
                                   '11-packaged')

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

    def reload_files(self):
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
        self.map_[list(self.alpha_columns)] = \
            self.map_[list(self.alpha_columns)].astype(float)

        self.otu = biom.load_table(otu_fp)

        self.unweighted = skbio.DistanceMatrix.read(unweighted_fp)
        self.weighted = skbio.DistanceMatrix.read(weighted_fp)

    def drop_alpha_outliers(self, metric='PD_whole_tree_10k', limit=55):
        self.outliers = self.map_.loc[(self.map_[metric] >= limit)].index
        keep = self.map_.loc[(self.map_[metric] < limit)].index
        self.map_ = self.map_.loc[keep]
        self.otu = self.otu.filter(keep)
        self.unweighted = self.unweighted.filter(keep)
        self.weighted = self.weighted.filter(keep)

    def clean_up_column(self, group):
        """Cleans up the indicated mapping column using the group"""

        datatype = group.type

        if datatype in {'Continous'}:
            group.drop_outliers(self.map_)
        elif datatype == 'Categorical':
            group.remove_ambigious(self.map_)
            group.drop_infrequent(self.map_)
            group.remap_clincial(self.map_)
        elif datatype == 'Multiple':
            group.correct_unspecified(self.map_)
        elif datatype == 'Ordinal':
            group.clean_frequency(self.map_)

        if datatype in {'Categorical', 'Multiple', 'Ordinal'}:
            group.remap_groups(self.map_)






