import copy
import os

import biom
import numpy as np
import pandas as pd
import skbio

import americangut as ag


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
                 one_sample=False, base_dir=None):
        """A pretty object to contain datasets for analysis

        Parameters
        ----------
        bodysite : {'fecal', 'oral', 'skin'}
            The bodysite samples for the data set
        trim : {'100nt', 'notrim'}
            The sequence length for the dataset
        depth : {'1k', '10k'}
            The rarefaction depth to analyze.
        sub_participants : bool
            Whether the healthy subset of adults should be used. This is
            defined as participants 20-69 with BMI between 18.5-30, and no
            reported history of diabetes, inflammatory bowel disease, or
            antibiotic use in the past year.
        one_sample : bool
            Whether one sample per participant should be used
        base_dir : str
            The path where datasets are currently located. The path should
            terminate with the 11-packaged folder if the data was generated
            with the American Gut primary-processing notebook set. Sub
            directories should be layed out according to the directory
            structure in those notebooks.
        """

        # Gets the base directory
        if base_dir is None:
            base_dir = os.path.join(
                ag.WORKING_DIR.split('American-Gut')[0],
                'American-Gut/ipynb/primary-processing/agp_processing/'
                '11-packaged'
                )
        # Checks the inputs
        if not os.path.exists(base_dir):
            raise ValueError('The base directory does not exist!')
        if bodysite not in {'fecal', 'oral', 'skin'}:
            raise ValueError('%s is not a supported bodysite.' % bodysite)
        if trim not in {'notrim', '100nt'}:
            raise ValueError('%s is not a supported trim length.' % trim)
        if depth not in {'1k', '10k'}:
            raise ValueError('%s is not a supported rarefaction depth' % depth)

        self.base_dir = base_dir
        self.bodysite = bodysite
        self.trim = trim
        self.depth = depth
        self.sub_participants = sub_participants
        self.one_sample = one_sample

        self.reload_files()

    def _check_dataset(self):
        """Makes sure the directory is pointing toward the correct directory"""

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

        # Gets the data directory
        self.data_dir = os.path.join(self.base_dir,
                                     '%(bodysite)s/'
                                     '%(sequence_trim)s/'
                                     '%(participant_set)s_participants/'
                                     '%(samples_per_participants)s/'
                                     '%(rarefaction_depth)s'
                                     % self.data_set
                                     )
        if not os.path.exists(self.data_dir):
            raise ValueError('The identified dataset does not exist')

    def reload_files(self):
        """Loads the map, otu table, and distance matrices from file"""
        self._check_dataset()

        # Determines the filepaths
        self.map_fp = os.path.join(self.data_dir,
                                   'ag_%(rarefaction_depth)s_%(bodysite)s.txt'
                                   % self.data_set)
        self.otu_fp = os.path.join(self.data_dir,
                                   'ag_%(rarefaction_depth)s_%(bodysite)s.biom'
                                   % self.data_set)
        self.beta_fp = {'unweighted_unifrac':
                        os.path.join(self.data_dir,
                                     'unweighted_unifrac_ag_'
                                     '%(rarefaction_depth)s_'
                                     '%(bodysite)s.txt'
                                     % self.data_set),
                        'weighted_unifrac':
                        os.path.join(self.data_dir,
                                     'weighted_unifrac_ag_'
                                     '%(rarefaction_depth)s'
                                     '_%(bodysite)s.txt'
                                     % self.data_set)}

        self.map_ = pd.read_csv(self.map_fp,
                                sep='\t',
                                dtype=str,
                                na_values=self.na_values)
        self.map_.set_index('#SampleID', inplace=True)
        a_columns = self.alpha_columns[self.data_set['rarefaction_depth']]
        self.map_[a_columns] = self.map_[a_columns].astype(float)

        self.otu_ = biom.load_table(self.otu_fp)

        self.beta = {metric:
                     skbio.DistanceMatrix.read(fp)
                     for metric, fp in self.beta_fp.iteritems()}

    def filter(self, ids, inplace=True):
        """Filter the dataset with a list of ids

        Parameters
        ----------
        ids : array-like
            The sample ids to keep in the new dataset
        inplace : bool
            Should the filtering be performed locally, or should the dataset
            be returned

        Returns
        -------
        map_ : DataFrame
            The metadata represented as a pandas DataFrame. Returned when
            `inplace` is `False`.
        otu_ : biom
            The Biom object containing the sample x OTU counts. Returned when
            `inplace` is `False`.
        beta : dict
            A dictioary keying beta diversity metric(s) to DistanceMatrix
            object(s). Returned when `inplace` is `False`.

        """

        map_ = self.map_.loc[ids]
        otu_ = copy.copy(self.otu_).filter(ids)
        beta = {k: self.beta[k].filter(ids) for k in self.beta.keys()}

        if inplace:
            self.map_ = map_
            self.otu_ = otu_
            self.beta = beta
        else:
            return map_, otu_, beta

    def clean_age(self):
        """..."""
        continous_cols = ['AGE_YEARS', 'HEIGHT_CM', 'WEIGHT_KG']
        self.map_['AGE_CORRECTED'] = self.map_['AGE_YEARS']
        self.map_[continous_cols] = self.map_[continous_cols].astype(float)
        age_check = self.map_['AGE_YEARS'] < 3
        height_check = (~ pd.isnull(self.map_['HEIGHT_CM']) &
                        self.map_['HEIGHT_CM'] > 91.4)
        weight_check = (~ pd.isnull(self.map_['WEIGHT_KG']) &
                        self.map_['WEIGHT_KG'] > 16.3)

        def check_etoh(x):
            return not (x == 'Never') and not pd.isnull(x)

        def bin_age(x):
            x = float(x)
            if np.isnan(x):
                return x
            elif x < 3:
                return 'baby'
            elif x < 13:
                return 'child'
            elif x < 20:
                return 'teen'
            elif x < 30:
                return '20s'
            elif x < 40:
                return '30s'
            elif x < 50:
                return '40s'
            elif x < 60:
                return '50s'
            elif x < 70:
                return '60s'
            else:
                return '70+'

        etoh_check = self.map_['ALCOHOL_FREQUENCY'].apply(check_etoh)
        replace = age_check & height_check & weight_check & etoh_check

        self.map_.loc[replace, 'AGE_CORRECTED'] = np.nan

        self.map_['AGE_CAT'] = self.map_['AGE_CORRECTED'].apply(bin_age)

    def drop_alpha_outliers(self, metric='PD_whole_tree_10k', limit=55,
                            inplace=True):
        """Removes samples with alpha diversity above the specified range

        In rounds 1-21 of the AmericanGut, there is a participant who has an
        alpha diverisity value 4 standard deviations about the next highest
        sample. Therefore, we remove this individual as an outlier.

        Parameters
        ----------
        metric : str, optional
            The alpha diversity metric used to identify the outlier.
            Default is `PD_whole_tree_1k`.
        limit : float, optional
            The alpha diversity value which should serve as the cutoff.
            Default is 55.

        """

        if metric not in self.alpha_columns[
                self.data_set['rarefaction_depth']]:
            raise ValueError('%s is not a supported metric!' % metric)

        self.outliers = self.map_.loc[(self.map_[metric] >= limit)].index
        keep = self.map_.loc[(self.map_[metric] < limit)].index

        self.filter(keep, inplace)

    def drop_bmi_outliers(self, bmi_limits=[5, 55]):
        """Corrects BMI values outside a specified range

        In rounds 1-21 of American Gut, participants entered BMI values which
        are imprbably (greater than 200). Therefore, we limit the BMI range
        to a more probable set of values.

        Parameters
        ----------
        bmi_limits : list, optional
            The range of BMI values to keep in a new column, called
            `BMI_CORRECTED` and `BMI_CAT`. Default is between 5 and 55.
        """

        self.map_['BMI'] = self.map_['BMI'].astype(float)

        keep = ((self.map_.BMI > bmi_limits[0]) &
                (self.map_.BMI < bmi_limits[1]))
        discard = ~ keep

        self.map_['BMI_CORRECTED'] = self.map_.loc[keep, 'BMI']
        self.map_.loc[discard, 'BMI_CAT'] = np.nan

    def filter_by_question(self, question, inplace=True):
        """Removes samples from people who didn't answer

        Many tests curerntly avaliable do not respond well to missing data.
        So, we filter the dataset to remove any samples not in the dataset.

        Parameters
        ----------
        question : AgQuestion
            An AqQuestion data dictionary object specifying the field
            to be used.

        Returns
        -------
        map_ : DataFrame
            The metadata represented as a pandas DataFrame. Returned when
            `inplace` is `False`.
        otu_ : biom
            The Biom object containing the sample x OTU counts. Returned when
            `inplace` is `False`.
        beta : dict
            A dictioary keying beta diversity metric(s) to DistanceMatrix
            object(s). Returned when `inplace` is `False`.

        """
        defined = self.map_[~ pd.isnull(self.map_[question.name])].index

        return self.filter(defined, inplace=inplace)

    def filter_subset(self, age=True, antibiotics=True, bmi=True,
                      diabetes=True, ibd=True, inplace=True):
        """Filters the data for the supplied groups

        Parameters
        ----------
        age, antibiotics, bmi, diabetes, ibd: bool
            Whether the data should be filtered using the specified subset
            field

        """
        def convert_to_binary(x):
            if not isinstance(x, str):
                return x
            elif x.lower() in {'true', 'y', 'yes', 't'}:
                return True
            elif x.lower() in {'false', 'f', 'n', 'no', 'none'}:
                return False

        binary = np.array([age, antibiotics, bmi, diabetes, ibd])
        columns = np.array(['SUBSET_AGE', 'SUBSET_ANTIBIOTIC_HISTORY',
                            'SUBSET_BMI', 'SUBSET_DIABETES', 'SUBSET_IBD',
                            ])
        for column in columns:
            self.map_[column] = self.map_[column].apply(convert_to_binary)
        columns = columns[binary]
        keep_ids = self.map_[self.map_[columns].all(1)].index

        return self.filter(keep_ids, inplace)

    def clean_group(self, group):
        """Leverages question structure to format the specified column in map_

        Parameters
        ----------
        group : AgQuestion
            An AqQuestion data dictionary object specifying the field
            to be used.

        """
        group.remap_data_type(self.map_)

        if group.type == 'Continous':
            group.drop_outliers(self.map_)

        elif group.type in {'Bool', 'Multiple'}:
            group.remap_data_type(self.map_)
            group.convert_to_word(self.map_)

        elif group.type == 'Categorical':
            group.remove_ambiguity(self.map_)
            group.remap_groups(self.map_)
            group.drop_infrequent(self.map_)

        elif group.type == 'Clinical':
            group.remap_clinical(self.map_)

        elif group.type == 'Frequency':
            group.clean(self.map_)
            group.combine_frequency(self.map_)

        if group.type == 'Multiple':
            group.correct_unspecified(self.map_)
