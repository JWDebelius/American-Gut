import copy
import inspect
import os
import types

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

        self.map_ = self.map_.loc[keep]
        self.otu_ = self.otu_.filter(keep)
        self.beta = {k: self.beta[k].filter(keep) for k in self.beta.keys()}

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

    def return_dataset(self, group):
        """Returns a map, otu table, and distance matrix for the group

        Many tests curerntly avaliable do not respond well to missing data.
        So, we filter the dataset to remove any samples not in the dataset.

        Parameters
        ----------
        group : AgQuestion
            An AqQuestion data dictionary object specifying the field
            to be used.

        Returns
        -------
        map_ : DataFrame
            The metadata represented as a pandas DataFrame
        otu_ : biom
            The Biom object containing the sample x OTU counts
        beta : dict
            A dictioary keying beta diversity metric(s) to DistanceMatrix
            object(s)

        """
        defined = self.map_[~ pd.isnull(self.map_[group.name])].index

        map_ = self.map_.loc[defined]
        otu_ = self.otu_.filter(defined, axis='sample')
        beta = {k: v.filter(defined) for k, v in self.beta.iteritems()}

        return map_, otu_, beta


class AgQuestion:
    true_values = {'yes', 'y', 'true', 1}
    false_values = {'no', 'n', 'false', 0}

    def __init__(self, name, description, dtype, clean_name=None, **kwargs):
        """A base object for describing single question outputs

        Parameters
        ----------
        name : str
            The name of column where the group is stored
        description : str
            A brief description of the data contained in the question
        dtype : object
            The datatype in which the responses should be represented. (i.e.
            float, int, str).
        clean_name : str, optional
            A nicer version of the way the column should be named.
        remap : function
            A function used to remap the data in the question.
        free_response: bool, optional
            Whether the question is a free response question or controlled
            vocabulary
        mimmarks : bool, optional
            If the question was a mimmarks standard field
        ontology : str, optional
            The type of ontology, if any, which was used in the field value.

        """

        # Checks the arguments
        if not isinstance(name, str):
            raise TypeError('name must be a string.')
        if not isinstance(description, str):
            raise TypeError('description must be a string')
        if not inspect.isclass(dtype):
            raise TypeError('dtype must be a class')
        if not isinstance(clean_name, str) and clean_name is not None:
            raise TypeError('If supplied, clean_name must be a string')

        # Handles the main information about the data
        self.name = name
        self.description = description
        self.dtype = dtype

        self.type = 'Question'
        if clean_name is None:
            self.clean_name = name.replace('_', ' ').title()
        else:
            self.clean_name = clean_name

        # Sets up
        self.free_response = kwargs.get('free_response', False)
        self.mimmarks = kwargs.get('mimmarks', False)
        self.ontology = kwargs.get('ontology', None)
        self.remap_ = kwargs.get('remap', None)

    def check_map(self, map_):
        """Checks the group exists in the metadata

        Parameters
        ----------
        map_ : DataFrame
            A pandas object containing the data to be analyzed. The
            Question `name` should be a column in the `map_`.

        """
        if self.name not in map_.columns:
            raise ValueError('%s is not a column in the supplied map!'
                             % self.name)

    def remap_data_type(self, map_):
        """Makes sure the target column in map_ has the correct datatype

        map_ : DataFrame
            A pandas dataframe containing the column described by the question
            name.

        """
        if self.dtype == bool:
            if not (set(map_[self.name].dropna()).issubset(
                    self.true_values.union(self.false_values))):
                raise TypeError('%s cannot be cast to a bool value.'
                                % self.name)

            def remap_(x):
                if isinstance(x, str) and x.lower() in self.true_values:
                    return True
                elif isinstance(x, str) and x.lower() in self.false_values:
                    return False
                elif np.isnan(x):
                    return x
                else:
                    try:
                        return bool(x)
                    except:
                        return np.nan
        else:
            def remap_(x):
                return self.dtype(x)

        map_[self.name] = map_[self.name].apply(remap_).astype(self.dtype)

        if self.type in {'Categorical', 'Multiple', 'Clinical', 'Bool',
                         'Frequency'}:
            self._update_order(remap_)


class AgCategorical(AgQuestion):

    def __init__(self, name, description, dtype, order, **kwargs):
        """A question object for categorical or ordinal questions

        Parameters
        ----------
        name : str
            The name of column where the group is stored
        description : str
            A brief description of the data contained in the question
        dtype : object
            The datatype in which the responses should be represented. (i.e.
            float, int, str).
        order : list
            The list of responses to the question
        clean_name : str, optional
            A nicer version of the way the column should be named.
        remap : function
            A function used to remap the data in the question.
        frequency_cutoff : float, optional
            The minimum number of observations required to keep a sample group.
        drop_ambiguous : bool, optional
            Whether ambigious values should be removed from the dataset.
        ambigigous_values : set, optional
            A list of values which are considered ambiguous and removed when
            `drop_ambiguous` is `True`.
        free_response: bool, optional
            Whether the question is a free response question or controlled
            vocabulary
        mimmarks : bool, optional
            If the question was a mimmarks standard field
        ontology : str, optional
            The type of ontology, if any, which was used in the field value.

        """
        if dtype not in {str, bool, int, float}:
            raise ValueError('%s is not a supported datatype for a '
                             'categorical variable.' % dtype)
        # Initializes the question
        AgQuestion.__init__(self, name, description, dtype, **kwargs)

        self.type = 'Categorical'

        if len(order) == 1:
            raise ValueError('There cannot be possible response.')
        self.order = order
        self.extremes = kwargs.get('extremes', None)
        if self.extremes is None:
            self.extremes = [order[0], order[-1]]

        self.earlier_order = []
        self.frequency_cutoff = kwargs.get('frequency_cutoff', None)
        self.drop_ambiguous = kwargs.get('drop_ambiguous', True)
        self.ambiguous_values = kwargs.get(
            'ambiguous_values',
            {"none of the above",
             "not sure",
             "other",
             "i don't know, i do not have a point of reference"
             })

    def _update_order(self, remap_):
        """Updates the order and earlier order arguments"""
        order = copy.copy(self.order)
        self.earlier_order.append(order)
        self.order = []
        for o in order:
            new_o = remap_(o)
            if new_o not in self.order and not pd.isnull(new_o):
                self.order.append(new_o)
        self.extremes = [remap_(e) for e in self.extremes]

    def remove_ambiguity(self, map_):
        """Removes ambiguous groups from the mapping file

        Parameters
        ----------
        map_ : DataFrame
            A pandas object containing the data to be analyzed. The
            Question `name` should be a column in the `map_`.

        """
        if self.drop_ambiguous:
            self.check_map(map_)
            self.remap_data_type(map_)

            def _remap(x):
                if x.lower() in self.ambiguous_values:
                    return np.nan
                else:
                    return x

            map_[self.name] = map_[self.name].apply(_remap)
            self._update_order(_remap)

    def remap_groups(self, map_):
        """Remaps columns in the column

        Parameters
        ----------
        map_ : DataFrame
            A pandas object containing the data to be analyzed. The
            Question `name` should be a column in the `map_`.

        """
        self.check_map(map_)
        self.remap_data_type(map_)

        if self.remap_ is not None:
            map_[self.name] = map_[self.name].apply(self.remap_)
            self._update_order(self.remap_)

    def drop_infrequent(self, map_):
        """Removes groups below a frequency cutoff

        Parameters
        ----------
        map_ : DataFrame
            A pandas object containing the data to be analyzed. The
            Question `name` should be a column in the `map_`.

        """
        self.check_map(map_)
        self.remap_data_type(map_)

        counts = map_.groupby(self.name).count().max(1)
        below = counts.loc[counts <= self.frequency_cutoff].index

        def _remap(x):
            if x in below:
                return np.nan
            else:
                return x

        map_[self.name] = map_[self.name].apply(_remap)
        self._update_order(_remap)

    def convert_to_numeric(self, map_):
        """Converts the data to integer values

        Parameters
        ----------
        map_ : DataFrame
            A pandas object containing the data to be analyzed. The
            Question `name` should be a column in the `map_`.

        """
        self.check_map(map_)
        order = {g: i for i, g in enumerate(self.order)}

        def remap_(x):
            if isinstance(x, (int, float)):
                return x
            elif x in order:
                return order[x]
            else:
                return np.nan

        map_[self.name] = map_[self.name].apply(remap_)
        self._update_order(remap_)

    def label_order(self, map_):
        """Prefixes the data with an ordinal integer

        Parameters
        ----------
        map_ : DataFrame
            A pandas object containing the data to be analyzed. The
            Question `name` should be a column in the `map_`.

        """
        self.check_map(map_)
        order = {g: '(%i) %s' % (i, g) for i, g in enumerate(self.order)}

        def remap_(x):
            if isinstance(x, (int, float)):
                return x
            elif x in order:
                return order[x]
            else:
                return np.nan

        map_[self.name] = map_[self.name].apply(remap_)
        self._update_order(remap_)


class AgBool(AgCategorical):
    def __init__(self, name, description, **kwargs):
        """A question object for boolean question

        Parameters
        ----------
        name : str
            The name of column where the group is stored
        description : str
            A brief description of the data contained in the question
        clean_name : str, optional
            A nicer version of the way the column should be named.
        remap : function
            A function used to remap the data in the question.
        mimmarks : bool, optional
            If the question was a mimmarks standard field
        ontology : str, optional
            The type of ontology, if any, which was used in the field value.

        """
        AgCategorical.__init__(self, name, description, bool,
                               ['true', 'false'], **kwargs)
        self.type = 'Bool'


class AgMultiple(AgBool):
    def __init__(self, name, description, unspecified, **kwargs):
        """A question object for a multiple response question

        Multiple response questions have a checkbox, where each response
        can be coded as a boolean. One of the checkboxes, the unspecified
        column, is used to indicate the participant did not respond to the
        question block.

        Parameters
        ----------
        name : str
            The name of column where the group is stored
        description : str
            A brief description of the data contained in the question
        unspecified : str
            The name of a reference column in the map_ which was used to
            indicate the participant did not respond. This should be an AgBool
            type question.
        clean_name : str, optional
            A nicer version of the way the column should be named.
        remap : function
            A function used to remap the data in the question.
        mimmarks : bool, optional
            If the question was a mimmarks standard field
        ontology : str, optional
            The type of ontology, if any, which was used in the field value.

        """

        AgBool.__init__(self, name, description, **kwargs)
        self.type = 'Multiple'
        self.unspecified = unspecified

    def correct_unspecified(self, map_):
        """Corrects column for missing responses"""
        self.check_map(map_)
        self.remap_data_type(map_)
        if self.unspecified not in map_.columns:
            raise ValueError('The unspecified reference (%s) is not a column'
                             ' in the metadata!' % self.unspecified)

        def remap_(x):
            if isinstance(x, str) and x.lower() in self.true_values:
                return True
            elif isinstance(x, str) and x.lower() in self.false_values:
                return False
            elif np.isnan(x):
                return x
            else:
                try:
                    return bool(x)
                except:
                    return np.nan

        if not map_[self.unspecified].dtype == np.bool_:
            map_[self.unspecified] = map_[self.unspecified].apply(remap_)
            map_[self.unspecified] = map_[self.unspecified].astype(bool)
        map_.loc[map_[self.unspecified], self.name] = np.nan


class AgClinical(AgCategorical):
    def __init__(self, name, description, strict=True, **kwargs):
        """A question object for questions describing clincial conditions

        The American Gut addresses several clincial conditions, based on the
        diagnosis method.

        Parameters
        ----------
        name : str
            The name of column where the group is stored
        description : str
            A brief description of the data contained in the question
        strict : bool
            Whether clincial conditions should consider only those diagnosed
            by a conventional medicine practioner (`True`), or include
            alternative or self diagnosis.
        clean_name : str, optional
            A nicer version of the way the column should be named.
        remap : function
            A function used to remap the data in the question.
        """

        order = ['Diagnosed by a medical professional (doctor, physician '
                 'assistant)',
                 'Diagnosed by an alternative medicine practitioner',
                 'Self-diagnosed',
                 'I do not have this condition'
                 ]
        AgCategorical.__init__(self, name, description, str, order, **kwargs)
        self.strict = strict
        self.type = 'Clinical'
        self.extremes = ['I do not have this condition',
                         'Diagnosed by a medical professional (doctor, '
                         'physician assistant)']

    def remap_clinical(self, map_):
        """Converts the clincial condtion to a boolean value"""
        self.check_map(map_)
        self.remap_data_type(map_)
        if self.strict:
            def remap_(x):
                if x in {'Diagnosed by a medical professional (doctor, '
                         'physician assistant)', 'yes'}:
                    return 'Yes'
                elif x in {'Diagnosed by an alternative medicine practitioner',
                           'Self-diagnosed'}:
                    return np.nan
                elif x in {'I do not have this condition', 'No'}:
                    return 'No'
                else:
                    return np.nan
        else:
            def remap_(x):
                    if x in {'Diagnosed by a medical professional (doctor, '
                             'physician assistant)',
                             'Diagnosed by an alternative medicine '
                             'practitioner', 'Self-diagnosed', 'Yes'}:
                        return 'Yes'
                    elif x in {'I do not have this condition',
                               'No'}:
                        return 'No'
                    else:
                        return np.nan

        map_[self.name] = map_[self.name].apply(remap_)
        self._update_order(remap_)
        self.extremes = ['No', 'Yes']


class AgFrequency(AgCategorical):
    def __init__(self, name, description, **kwargs):
        """
        A question object for frequency questions relating to weekly occurance

        Parameters
        ----------
        name : str
            The name of column where the group is stored
        description : str
            A brief description of the data contained in the question
        clean_name : str, optional
            A nicer version of the way the column should be named.
        remap : function
            A function used to remap the data in the question.
        """
        order = ['Never',
                 'Rarely (a few times/month)',
                 'Occasionally (1-2 times/week)',
                 'Regularly (3-5 times/week)',
                 'Daily']
        AgCategorical.__init__(self, name, description, dtype=str, order=order,
                               **kwargs)
        self.type = 'Frequency'

    def clean(self, map_):
        """Converts the frequency values to shorter strings"""
        self.check_map(map_)

        def remap_(x):
            if isinstance(x, str) and '(' in x:
                return x.split('(')[1].replace(')', '').capitalize()
            else:
                return x

        map_[self.name] = map_[self.name].apply(remap_)
        self._update_order(remap_)
        if 'Less than once/week' in set(map_[self.name]):
            self.order = ['Never',
                          'Less than once/week',
                          '1-2 times/week',
                          '3-5 times/week',
                          'Daily']


class AgContinous(AgQuestion):

    def __init__(self, name, description, unit=None, **kwargs):
        """A Question object with continous responses

        Parameters
        ----------
        name : str
            The name of column where the group is stored
        description : str
            A brief description of the data contained in the question
        unit : str, optional
            The unit of measure for the data type, if relevant. Units
            are awesome, and should be included whenever relevant.
        clean_name : str, optional
            A nicer version of the way the column should be named.
        remap : function
            A function used to remap the data in the question.
        range : two-elemant iterable
            The range of values pertinant to analysis.
        """

        AgQuestion.__init__(self, name, description, float, **kwargs)
        self.unit = unit
        lower, upper = kwargs.get('range', [None, None])
        if lower > upper:
            raise ValueError('The lower limit cannot be greater than '
                             'the upper!')
        self.lower = lower
        self.upper = upper
        self.type = 'Continous'

    def drop_outliers(self, map_):
        """Removes datapoints outside of the `range`"""
        self.check_map(map_)
        self.remap_data_type(map_)

        if self.lower is not None:

            def remap_(x):
                if x < self.lower:
                    return np.nan
                elif x > self.upper:
                    return np.nan
                else:
                    return x

            map_[self.name] = map_[self.name].apply(remap_)


def _remap_abx(x):
    if x == 'I have not taken antibiotics in the past year':
        return 'More than a year'
    else:
        return x


def _remap_bowel_frequency(x):
    if x in {'Four', 'Five or more'}:
        return 'Four or more'
    else:
        return x


def _remap_fecal_quality(x):
    if x == 'I tend to be constipated (have difficulty passing stool)':
        return 'Constipated'
    elif x == 'I tend to have diarrhea (watery stool)':
        return 'Diarrhea'
    elif x == 'I tend to have normal formed stool':
        return 'Normal'
    else:
        return x


def _remap_contraceptive(x):
    if isinstance(x, str):
        return x.split(',')[0]
    else:
        return x


def _remap_gluten(x):
    if x == 'I was diagnosed with celiac disease':
        return 'Celiac Disease'
    elif x == 'I do not eat gluten because it makes me feel bad':
        return 'Non Medical'
    elif x == ('I was diagnosed with gluten allergy (anti-gluten IgG), but '
               'not celiac disease'):
        return "Anti-Gluten Allergy"
    else:
        return x


def _remap_last_travel(x):
    if x in {'I have not been outside of my country of residence in'
             ' the past year.'}:
        return 'More than a year'
    else:
        return x


def _remap_pool_frequency(x):
    if x in {'Occasionally (1-2 times/week)',
             'Regularly (3-5 times/week)', 'Daily',
             '1-2 times/week', '3-5 times/week'}:
        return 'More than once a week'
    else:
        return x


def _remap_sleep(x):
    if x in {'Less than 5 hours', '5-6 hours'}:
        return 'Less than 6 hours'
    else:
        return x


def _remap_diet(x):
    if x == 'Vegetarian but eat seafood':
        return 'Pescetarian'
    elif x == 'Omnivore but do not eat red meat':
        return 'No Red Meat'
    else:
        return x


def _remap_fed_as_infant(x):
    if x == 'Primarily breast milk':
        return 'Breast milk'
    elif x == 'Primarily infant formula':
        return 'Formula'
    elif x == 'A mixture of breast milk and formula':
        return 'Mix'
    else:
        return x


def _remap_country(x):
    if x in {'United Kingdom', 'Jersey', 'Isle of Man', 'Ireland', 'Scotland'}:
        return 'British Isles'
    else:
        return x


ag_data_dictionary = {
    'AGE_CAT': AgCategorical(
        name='AGE_CAT',
        description=("Age categorized by decade, with the exception of babies "
                     "(0-2 years), children (3-12 years) and teens "
                     "(13-19 years). Calculated from AGE_CORRECTED"),
        dtype=str,
        order=['20s', '30s', '40s', '50s', '60s'],
        extremes=['20s', '60s'],
        ),
    'ALCOHOL_FREQUENCY': AgFrequency(
        name='ALCOHOL_FREQUENCY',
        description='How often does the participant partake of alcohol?',
        ),
    'ALCOHOL_TYPES_BEERCIDER': AgMultiple(
        name='ALCOHOL_TYPES_BEERCIDER',
        description='The par ticipant consumes beer or cider',
        unspecified='ALCOHOL_TYPES_UNSPECIFIED',
        ),
    'ALCOHOL_TYPES_RED_WINE': AgMultiple(
        name='ALCOHOL_TYPES_RED_WINE',
        description='The participant consumes red wine',
        unspecified='ALCOHOL_TYPES_UNSPECIFIED',
        ),
    'ALCOHOL_TYPES_SOUR_BEERS': AgMultiple(
        name='ALCOHOL_TYPES_SOUR_BEERS',
        description='The participant consumes sour beers',
        unspecified='ALCOHOL_TYPES_UNSPECIFIED',
        ),
    'ALCOHOL_TYPES_SPIRITSHARD_ALCOHOL': AgMultiple(
        name='ALCOHOL_TYPES_SPIRITSHARD_ALCOHOL',
        description='The participant consumes spirits or hard liquor',
        unspecified='ALCOHOL_TYPES_UNSPECIFIED',
        ),
    'ALCOHOL_TYPES_WHITE_WINE': AgMultiple(
        name='ALCOHOL_TYPES_WHITE_WINE',
        description='TThe participant consumes white wine',
        unspecified='ALCOHOL_TYPES_UNSPECIFIED',
        ),
    'ANTIBIOTIC_HISTORY': AgCategorical(
        name='ANTIBIOTIC_HISTORY',
        description=('How recently has the participant taken antibiotics?'),
        dtype=str,
        order=['Week', 'Month', '6 months', 'Year',
               'I have not taken antibiotics in the past year'],
        remap=_remap_abx,
        ),
    'BMI_CAT': AgCategorical(
        name='BMI_CAT',
        description='The body mass index, categorized according to the WHO.',
        dtype=str,
        order=['Underweight', 'Normal', 'Overweight', 'Obese'],
        extremes=['Normal', 'Obese'],
        ),
    'BOWEL_MOVEMENT_FREQUENCY': AgCategorical(
        name='BOWEL_MOVEMENT_FREQUENCY',
        description=('Number of daily bowel movements'),
        dtype=str,
        order=['Less than one', 'One', 'Two', 'Three', 'Four', 'Five or more'],
        remap=_remap_bowel_frequency,
        extremes=['Less than One', 'Five or more'],
        ),
    'BOWEL_MOVEMENT_QUALITY': AgCategorical(
        name='BOWEL_MOVEMENT_QUALITY',
        description=('Does the participant tend toward constipation or '
                     'diarrhea?'),
        dtype=str,
        order=['I tend to have normal formed stool',
               "I don't know, I do not have a point of reference",
               'I tend to be constipated (have difficulty passing stool)',
               'I tend to have diarrhea (watery stool)',
               ],
        remap=_remap_fecal_quality,
        extremes=['Normal', 'Diarrhea'],
        ),
    'CAT': AgBool(
        name='CAT',
        description='Does the participant have a pet cat?',
        ),
    'CHICKENPOX': AgCategorical(
        name='CHICKENPOX',
        description='Has the participant has chickenpox?',
        dtype=str,
        order=['No', 'Not sure', 'Yes'],
        ),
    'COLLECTION_SEASON': AgCategorical(
        name='COLLECTION_SEASON',
        dtype=str,
        description=("Season in which the sample was collected. Winter: "
                     "Dec 1 - Feb 28/29; Spring: March 1-May 31; "
                     "Summer: June 1 - August 31; Fall: Sept 1-Nov 30"),
        order=['Winter', 'Spring', 'Summer', 'Fall'],
        extremes=['Winter', 'Summer']
        ),
    'COLLECTION_MONTH': AgCategorical(
        name='COLLECTION_MONTH',
        description=('Month in which the sample was collected'),
        dtype=str,
        order=['January', 'February', 'March', 'April', 'May', 'June',
               'July', 'August', 'September', 'October', 'November',
               'December'
               ],
        extremes=['January', 'July'],
        ),
    'CONSUME_ANIMAL_PRODUCTS_ABX': AgCategorical(
        name="CONSUME_ANIMAL_PRODUCTS_ABX",
        description=("Does the participant eat animal products treated with "
                     "antibiotics?"),
        dtype=str,
        order=['No', 'Not sure', 'Yes'],
        extremes=['Yes', 'No'],
        ),
    'COUNTRY': AgCategorical(
        name='COUNTRY',
        description=("country of residence"),
        dtype=str,
        ontology='GAZ',
        mimmarks=True,
        order=['Australia', 'Belgium', 'Brazil', 'Canada', 'China',
               'Czech Republic', 'Denmark', 'Finland', 'France', 'Germany',
               'Ireland', 'Isle of Man', 'Italy', 'Japan', 'Jersey',
               'Netherlands', 'New Zealand', 'Norway', 'Poland', 'Spain',
               'Sweden', 'Switzerland', 'Thailand', 'USA',
               'United Arab Emirates', 'United Kingdom'],
        extremes=['US', 'United Kingdom'],
        frequency_cutoff=50,
        ),
    'CONTRACEPTIVE': AgCategorical(
        name="CONTRACEPTIVE",
        description="Does the participant use contraceptives?",
        dtype=str,
        order={'No', 'Yes, I am taking the pill',
               'Yes, I use a contraceptive patch (Ortho-Evra)',
               'Yes, I use a hormonal IUD (Mirena)',
               'Yes, I use an injected contraceptive (DMPA)',
               'Yes, I use the NuvaRing'},
        extremes=['No', 'Yes, I am taking the pill'],
        remap=_remap_contraceptive,
        ),
    'CSECTION': AgCategorical(
        name="CSECTION",
        description=("Was the participant born via c-section?"),
        dtype=str,
        order=['No', 'Not sure', 'Yes']
        ),
    'DIABETES': AgClinical(
        name="DIABETES",
        description=("Has the participant been diganosed with diabetes"),
        ),
    'DIET_TYPE': AgCategorical(
        name="DIET_TYPE",
        description=(""),
        dtype=str,
        order=['Vegan', 'Vegetarian', 'Vegetarian but eat seafood',
               'Omnivore but do not eat red meat', 'Omnivore'],
        extremes=['Vegetarian', 'Omnivore'],
        remap=_remap_diet,
        ),
    'DRINKING_WATER_SOURCE': AgCategorical(
        name="DRINKING_WATER_SOURCE",
        description="Primary source of water",
        dtype=str,
        order=['Bottled', 'City', 'Filtered', 'Not sure', 'Well'],
        extreme=['Bottled', 'Well'],
        ),
    'EXERCISE_FREQUENCY': AgFrequency(
        name="EXERCISE_FREQUENCY",
        description=("How often the participant exercises."),
        ),
    'EXERCISE_LOCATION': AgCategorical(
        name="EXERCISE_LOCATION",
        description=("Primary exercise location - indoor, outdoor, both"),
        dtype=str,
        order=['Indoors', 'Both', 'Depends on the season',
               'None of the above', 'Outdoors'],
        extremes=['Indoors', 'Outdoors'],
        ),
    'FED_AS_INFANT': AgCategorical(
        name='FED_AS_INFANT',
        description='Food source as an infant (breast milk or formula)',
        dtype=str,
        order=['Primarily breast milk',
               'A mixture of breast milk and formula', 'Not sure',
               'Primarily infant formula'],
        extremes=['Primarily breast milk', 'Primarily infant formula'],
        remap=_remap_fed_as_infant,
        ),
    'FERMENTED_PLANT_FREQUENCY': AgFrequency(
        name='FERMENTED_PLANT_FREQUENCY',
        description=("Participant reported onsumption of at least one "
                     "serving of fermented vegetables or tea in a day"),
        ),
    'FLOSSING_FREQUENCY': AgFrequency(
        name='FLOSSING_FREQUENCY',
        description=("How of the participant flosses their teeth"),
        ),
    'FRUIT_FREQUENCY': AgFrequency(
        name='FRUIT_FREQUENCY',
        description=("How often the participant consumes at least 2-3 "
                     "serving of fruit a day"),
        ),
    'GLUTEN': AgCategorical(
        name="GLUTEN",
        description=("Why does the participant follow a gluten free diet?"),
        dtype=str,
        order=['I was diagnosed with celiac disease',
               'I do not eat gluten because it makes me feel bad',
               'I was diagnosed with gluten allergy (anti-gluten IgG),'
               ' but not celiac disease',
               'No'],
        extremes=['No', 'Celiac Disease'],
        remap=_remap_gluten
        ),
    'HOMECOOKED_MEALS_FREQUENCY': AgFrequency(
        name='HOMECOOKED_MEALS_FREQUENCY',
        description=("How often does the participant consume meals cooked"
                     " at home?"),
        remap=_remap_pool_frequency,
        ),
    'IBD': AgClinical(
        name='IBD',
        description=("Has the participant been diagnosed with inflammatory "
                     "bowel disease, include crohns disease or ulcerative "
                     "colitis?"),
        ),
    'LACTOSE': AgBool(
        name="LACTOSE",
        description=("Is the participant lactose intolerant?"),
        ),
    'LAST_TRAVEL': AgCategorical(
        name="LAST_TRAVEL",
        description=("When the participant last traveled away from home"),
        dtype=str,
        order=['I have not been outside of my country of residence in'
               ' the past year.', '1 year', '6 months', '3 months', 'Month'],
        remap_=_remap_last_travel,
        extremes=['I have not been outside of my country of residence in'
                  ' the past year.', 'Month'],
        ),
    'LOWGRAIN_DIET_TYPE': AgBool(
        name="LOWGRAIN_DIET_TYPE",
        description=("Does the participant eat a low grain diet?"),
        ),
    'LUNG_DISEASE': AgClinical(
        name='LUNG_DISEASE',
        description=('Does the participant have diagnosed lung disease '
                     '(asthma, COPD, etc)?'),
        ),
    'MIGRAINE': AgClinical(
        name='MIGRAINE',
        description=('Does the participant experience migraines?'),
        ),
    'MULTIVITAMIN': AgBool(
        name="MULTIVITAMIN",
        description=("Does the participant take a multivitamin?"),
        ),
    'OLIVE_OIL': AgFrequency(
        name="OLIVE_OIL",
        description=("Frequency participant eats food cooked with Olive Oil"),
        ),
    'ONE_LITER_OF_WATER_A_DAY_FREQUENCY': AgFrequency(
        name="ONE_LITER_OF_WATER_A_DAY_FREQUENCY",
        description=("How frequently does the participant consume at least"
                     " one liter of water"),
        ),
    'POOL_FREQUENCY': AgFrequency(
        name="POOL_FREQUENCY",
        description=("How often the participant uses a pool or hot tub"),
        remap_=_remap_pool_frequency,
        extremes=["Never", "Daily"]
        ),
    'PREPARED_MEALS_FREQUENCY': AgFrequency(
        name="PREPARED_MEALS_FREQUENCY",
        description=("frequency with which the participant eats out at a "
                     "resturaunt, including carryout"),
        ),
    'PROBIOTIC_FREQUENCY': AgFrequency(
        name="PROBIOTIC_FREQUENCY",
        description=("how often the participant uses a probiotic"),
        ),
    'RACE': AgCategorical(
        name="RACE",
        description=("Participant's race or ethnicity"),
        dtype=str,
        order=['Hispanic', 'African American', 'Other',
               'Caucasian', 'Asian or Pacific Islander'],
        frequency_cutoff=50,
        ),
    'SEASONAL_ALLERGIES': AgBool(
        name="SEASONAL_ALLERGIES",
        description=("Does the participant have seasonal allergies?"),
        ),
    'SEX': AgCategorical(
        name="SEX",
        description=("MIMARKS standard field - participant biological "
                     "sex, not sexual identity"),
        dtype=str,
        order=['female', 'other', 'male'],
        mimmarks=True,
        extreme=['male', 'female'],
        ),
    "SLEEP_DURATION": AgCategorical(
        name="SLEEP_DURATION",
        description=("How long the participant sleeps in the average night?"),
        dtype=str,
        order=['Less than 5 hours', '5-6 hours', '6-7 hours', '7-8 hours',
               '8 or more hours'],
        remap=_remap_sleep,
        extreme=['Less than 5 hours', '8 or more hours'],
        ),
    'SMOKING_FREQUENCY': AgFrequency(
        name="SMOKING_FREQUENCY",
        description="How often the participant smokes?",
        remap=_remap_pool_frequency
        ),
    'SUGARY_SWEETS_FREQUENCY': AgFrequency(
        name="SUGARY_SWEETS_FREQUENCY",
        description=("how often does the participant eat sweets (candy, "
                     "ice-cream, pastries etc)?"),
        ),
    'TYPES_OF_PLANTS': AgCategorical(
        name="TYPES_OF_PLANTS",
        description=("Number of plant species eaten in week of observation"),
        dtype=str,
        order=['Less than 5', '6 to 10', '11 to 20', '21 to 30',
               'More than 30'],
        extreme=['Less than 5', 'More than 30'],
        ),
    'VEGETABLE_FREQUENCY': AgFrequency(
        name="VEGETABLE_FREQUENCY",
        description=("How many times a week the participant eats vegetables"),
        ),
    'VITAMIN_B_SUPPLEMENT_FREQUENCY': AgFrequency(
        name="VITAMIN_B_SUPPLEMENT_FREQUENCY",
        description=("How often the participant takes a vitamin B or "
                     "vitamin B complex supplement"),
        ),
    'VITAMIN_D_SUPPLEMENT_FREQUENCY': AgFrequency(
        name="VITAMIN_D_SUPPLEMENT_FREQUENCY",
        description=("How often the participant takes a vitamin D supplement"),
        ),
    'WEIGHT_CHANGE': AgCategorical(
        name="WEIGHT_CHANGE",
        description=("Has the participants weight has changed more than 10 "
                     "lbs in the past year?"),
        dtype=str,
        order=['Increased more than 10 pounds',
               'Decreased more than 10 pounds', 'Remained stable'],
        extremes=['Remained stable', 'Increased more than 10 pounds']
        ),
    }
