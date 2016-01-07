
import inspect
import types
import copy

import numpy as np
import pandas as pd


class AgQuestion:
    true_values = {'yes', 'y', 'true', 1}
    false_values = {'no', 'n', 'false', 0}

    def __init__(self, name, description, dtype, clean_name=None,
                 free_response=False, ontology=None):
        """A base object for describing single question outputs
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

        # Handles the main information
        self.name = name
        self.description = description
        self.dtype = dtype
        if clean_name is None:
            self.clean_name = name.replace('_', ' ')
        else:
            self.clean_name = clean_name
        self.free_response = free_response
        self.ontology = ontology

    def remap_data_type(self, map_):
        """Makes sure the target column in map_ has the correct datatype"""
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
            map_[self.name] = map_[self.name].apply(remap_).astype(bool)
        else:
            map_[self.name] = map_[self.name].dropna().astype(self.dtype)

    def check_map(self, map_):
        if self.name not in map_.columns:
            raise ValueError('%s is not a column in the supplied map!'
                             % self.name)


class AgCategorical(AgQuestion):
    _ambigious = {"None of the above",
                  "Not sure",
                  "I don't know, I do not have a point of reference",
                  }

    def __init__(self, name, description, dtype, groups, clean_name=None,
                 remap=None, clinical=False, free_response=False,
                 ontology=None, frequency_cutoff=None, clincially_strict=True,
                 drop_ambigous=False):
        """..."""
        if dtype not in {str, bool, int, float}:
            raise ValueError('%s is not a supported datatype for a '
                             'categorical variable.' % dtype)
        AgQuestion.__init__(self, name, description, dtype, clean_name,
                            free_response, ontology)
        self.type = 'Categorical'
        self.groups = groups
        self.remap_ = remap
        self.earlier_groups = []
        self.clinical = clinical
        self.frequency_cutoff = frequency_cutoff
        self.clincially_strict = clincially_strict
        self.drop_ambigous = drop_ambigous

    def remove_ambigious(self, map_, ambigious=None):
        """Removes ambigious groups from the mapping file"""
        self.check_map(map_)

        self.remap_data_type(map_)

        if ambigious is None:
            ambigious = self._ambigious

        def _remap(x):
            if x in ambigious:
                return np.nan
            else:
                return x
        if self.drop_ambigious:
            map_[self.name] = map_[self.name].apply(_remap)
            self.earlier_groups.append(copy.copy(self.groups))
            self.groups = self.groups - ambigious

    def remap_groups(self, map_):
        """Remaps columns in the column"""
        self.check_map(map_)
        self.remap_data_type(map_)

        if self.remap_ is not None:
            map_[self.name] = map_[self.name].apply(self.remap_)
            self.earlier_groups.append(copy.copy(self.groups))
            self.groups = set([self.remap_(g) for g in
                               self.earlier_groups[-1]])

    def drop_infrequency(self, map_):
        """Removes groups below a frequency cutoff"""
        self.check_map(map_)
        self.remap_data_type(map_)

        self.infrequency_cutoff = None
        counts = map_.groupby(self.name).count().max(1)
        below = counts.loc[counts <= cutoff].index

        def filter_low(x):
            if x in below:
                return np.nan
            else:
                return x

        map_[self.name] = map_[self.name].apply(filter_low)

        self.earlier_groups.append(copy.copy(self.groups))
        self.groups = (set([filter_low(g) for g in self.earlier_groups[-1]]) -
                       {np.nan})

    def remap_clinical(self, map_):
        self.check_map(map_)
        self.remap_data_type(map_)
        if strict:
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
        else:
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
            if self.clincial:
                map_[self.name] = map_[self.name].apply(remap_)
                self.earlier_groups.append(copy.copy(self.groups))
                self.groups = (set([remap_(g) for g in
                               self.earlier_groups[-1]]) - {np.nan})


class AgOrdinal(AgCategorical):

    def __init__(self, name, description, dtype, order, clean_name=None,
                 remap=None, free_response=False, ontology=None):
        AgCategorical.__init__(self, name, description, dtype, set(order),
                               clean_name=clean_name, remap=remap,
                               free_response=free_response, ontology=ontology)
        self.type = 'Ordinal'
        self.order = order

    def clean_frequency(self, map_):
        self.check_map(map_)

        def remap_(x):
            if isinstance(x, str) and '(' in x:
                return x.split('(')[1].replace(')', '').capitalize()
            else:
                return x

        map_[self.name] = map_[self.name].apply(remap_)
        self.earlier_groups.append(copy.copy(self.order))
        self.groups = (set([remap_(g) for g in self.earlier_groups[-1]]) -
                       {np.nan})
        self.order = [remap_(g) for g in self.order if not pd.isnull(g)]

    def convert_to_numeric(self, map_):
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
        self.earlier_groups.append(copy.copy(self.order))
        self.groups = (set([remap_(g) for g in self.earlier_groups[-1]]) -
                       {np.nan})
        self.order = [remap_(g) for g in self.order if not pd.isnull(g)]

    def label_order(self, map_):
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
        self.earlier_groups.append(copy.copy(self.order))
        self.groups = (set([remap_(g) for g in self.earlier_groups[-1]]) -
                       {np.nan})
        self.order = [remap_(g) for g in self.order if not pd.isnull(g)]


class AgContinous(AgQuestion):
    type = 'Continous'

    def __init__(self, name, description, dtype, clean_name=None, remap=None,
                 unit=None, free_response=False, ontology=None):
        AgQuestion.__init__(self, name, description, dtype, clean_name=None,
                            free_response=free_response, ontology=ontology)
        self.unit = unit

    def drop_outliers(self, map_, lower, upper):
        self.check_map(map_)
        self.remap_data_type(map_)

        if lower >= upper:
            raise ValueError('The lower limit cannot be greater than '
                             'the upper!')

        def remap_(x):
            if x < lower:
                return np.nan
            elif x > upper:
                return np.nan
            else:
                return x

        map_[self.name] = map_[self.name].apply(remap_)
        self.range_limits = [lower, upper]


class AgMultiple(AgCategorical):
    def __init__(self, name, description, unspecified, clean_name=None,
                 ontology=None):
        AgCategorical.__init__(self, name, description, dtype=bool,
                               groups={True, False}, clean_name=clean_name,
                               free_response=False, ontology=ontology)
        self.unspecified = unspecified
        self.type = 'Multiple'

    def correct_unspecified(self, map_):
        self.check_map(map_)
        self.remap_data_type(map_)
        if self.unspecified not in map_.columns:
            raise ValueError('The unspecified reference (%s) is not a column'
                             ' in the mapping file!' % self.unspecified)

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

ag_data_dictionary = {
    'ACNE_MEDICATION': AgCategorical(
        name='ACNE_MEDICATION',
        description='Does the participant use of acne medication?',
        dtype=bool,
        groups={True, False}
        ),
    'ACNE_MEDICATION_OTC': AgCategorical(
        name='ACNE_MEDICATION',
        description=('Does the participant use over the counter acne '
                     'medication?'),
        dtype=bool,
        groups={True, False}
        ),
    'ADD_ADHD': AgCategorical(
        name='ADD_ADHD',
        description=("Has the participant been diagnosed with ADD_ADHD?"),
        dtype=str,
        groups={('Diagnosed by a medical professional (doctor, physician '
                 'assistant)'), 'Self-diagnosed',
                'Diagnosed by an alternative medicine practitioner',
                'I do not have this condition'},
        clinical=True
        ),
    'AGE_CAT': AgOrdinal(
        name='AGE_CAT',
        description=("Age categorized by decade, with the exception of babies"
                     " (0-2 years), children ((3-12 years) and teens (13-19 "
                     "years). Calculated from AGE_CORRECTED"),
        dtype=str,
        order=['baby', 'child', 'teen', '20s', '30s', '40s', '50s', '60s',
               '70+']
        ),
    'AGE_CORRECTED': AgContinous(
        name='AGE_CORRECTED',
        description=("Age, corrected for people with negative age, age "
                     "greater than 120 years, and individuals under 3 years"
                     " of age who are larger than the 95% growth curve given"
                     " by (the WHO or report any alcohol consumption"),
        dtype=float,
        unit='years'
        ),
    'AGE_YEARS': AgContinous(
        name='AGE_YEARS',
        description=("Age of the participant in years"),
        dtype=float,
        unit='years'
        ),
    'ALCOHOL_CONSUMPTION': AgCategorical(
        name='ALCOHOL_CONSUMPTION',
        description=("Does the participant consume any alcohol"),
        dtype=bool,
        groups={True, False}
        ),
    'ALCOHOL_FREQUENCY': AgOrdinal(
        name='ALCOHOL_FREQUENCY',
        description=("How often does the participant partake of alcohol?"),
        dtype=str,
        order=['Never',
               'Rarely (a few times/month)',
               'Occasionally (1-2 times/week)',
               'Regularly (3-5 times/week)',
               'Daily']
        ),
    'ALCOHOL_TYPES_BEERCIDER': AgMultiple(
        name='ALCOHOL_TYPES_BEERCIDER',
        description=("The participant consumes beer or cider"),
        unspecified='ALCOHOL_TYPES_UNSPECIFIED',
        ),
    'ALCOHOL_TYPES_RED_WINE': AgMultiple(
        name='ALCOHOL_TYPES_RED_WINE',
        description="The participant consumes red wine",
        unspecified='ALCOHOL_TYPES_UNSPECIFIED',
        ),
    'ALCOHOL_TYPES_SOUR_BEERS': AgMultiple(
        name='ALCOHOL_TYPES_SOUR_BEERS',
        description=("The participant consumes sour beers"),
        unspecified='ALCOHOL_TYPES_UNSPECIFIED',
        ),
    'ALCOHOL_TYPES_SPIRITS_HARD_ALCOHOL': AgMultiple(
        name='ALCOHOL_TYPES_SPIRITS_HARD_ALCOHOL',
        description=("The participant consumes spirits or hard liquor"),
        unspecified='ALCOHOL_TYPES_UNSPECIFIED',
        ),
    'ALCOHOL_TYPES_UNSPECIFIED': AgCategorical(
        name='ALCOHOL_TYPES_UNSPECIFIED',
        description=("The participant has not reported types of alcohol consumed"),
        dtype=bool,
        groups={True, False}
        ),
    'ALCOHOL_TYPES_WHITE_WINE': AgMultiple(
        name='ALCOHOL_TYPES_WHITE_WINE',
        description=("The participant consumes white wine"),
        unspecified='ALCOHOL_TYPES_UNSPECIFIED',
        ),
    'ALLERGIC_TO_I_HAVE_NO_FOOD_ALLERGIES_THAT_I_KNOW_OF': AgMultiple(
        name='ALLERGIC_TO_I_HAVE_NO_FOOD_ALLERGIES_THAT_I_KNOW_OF',
        description=("The participant does not have any food allergies"),
        unspecified='ALLERGIC_TO_UNSPECIFIED',
        ),
    'ALLERGIC_TO_OTHER': AgMultiple(
        name='ALLERGIC_TO_OTHER',
        description=("Is the participant allergic to some food other than peanuts, shellfish or tree (nuts? Maps to FOODALLERGIES_OTHER in rounds 1-15."),
        unspecified='ALLERGIC_TO_UNSPECIFIED',
        ),
    'ALLERGIC_TO_PEANUTS': AgMultiple(
        name='ALLERGIC_TO_PEANUTS',
        description=("Is the participant allergic to peanuts? Maps to FOODALLERGIES_PEANUTS in rounds (1-15"),
        unspecified='ALLERGIC_TO_UNSPECIFIED',
        ),
    'ALLERGIC_TO_SHELLFISH': AgMultiple(
        name='ALLERGIC_TO_SHELLFISH',
        description=("Is the participant allergic to shellfish? Maps to FOODALLERGIES_SHELLFISH in (rounds 1-15"),
        unspecified='ALLERGIC_TO_UNSPECIFIED',
        ),
    'ALLERGIC_TO_TREE_NUTS': AgMultiple(
        name='ALLERGIC_TO_TREE_NUTS',
        description=("Is the participant allergic to tree nuts? Map to FOODALLERGIES_TREE_NUTS in (rounds 1-15"),
        unspecified='ALLERGIC_TO_UNSPECIFIED',
        ),
    'ALLERGIC_TO_UNSPECIFIED': AgCategorical(
        name='ALLERGIC_TO_UNSPECIFIED',
        description=("The participant skipped the question about allergies."),
        dtype=bool,
        groups={True, False},
        ),
    'ANTIBIOTIC_HISTORY': AgOrdinal(
        name='ANTIBIOTIC_HISTORY',
        description=("How recently has the participant taken antibiotics?"),
        dtype=str,
        order=['Week', 'Month', '6 months', 'Year',
               'I have not taken antibiotics in the past year.']
        )
    }

   