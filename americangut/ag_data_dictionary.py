
import inspect
import types
import copy

import numpy as np
import pandas as pd


class AgQuestion:
    true_values = {'yes', 'y', 'true', 1}
    false_values = {'no', 'n', 'false', 0}

    def __init__(self, name, description, dtype, clean_name=None,
                 free_response=False):
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
        self.clean_name = clean_name
        self.free_response = free_response

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
                 remap=None, clinical=False, free_response=False):
        """..."""
        if dtype not in {str, bool, int, float}:
            raise ValueError('%s is not a supported datatype for a '
                             'categorical variable.' % dtype)
        AgQuestion.__init__(self, name, description, dtype, clean_name,
                            free_response)
        self.type = 'Categorical'
        self.groups = groups
        self.remap_ = remap
        self.earlier_groups = []
        self.clinical = clinical

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

    def drop_infrequency(self, map_, cutoff=25):
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

    def remap_clinical(self, map_, strict=True):
        if not self.clinical:
            raise ValueError('%s is not designated as clinical.' % self.name)

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
        map_[self.name] = map_[self.name].apply(remap_)
        self.earlier_groups.append(copy.copy(self.groups))
        self.groups = (set([remap_(g) for g in self.earlier_groups[-1]]) -
                       {np.nan})


class AgOrdinal(AgCategorical):

    def __init__(self, name, description, dtype, order, clean_name=None,
                 remap=None, free_response=False):
        AgCategorical.__init__(self, name, description, dtype, set(order),
                               clean_name=clean_name, remap=remap,
                               free_response=free_response)
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
                 unit=None, free_response=False):
        AgQuestion.__init__(self, name, description, dtype, clean_name=None,
                            free_response=free_response)
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


class AgMultiple(AgQuestion):
    def __init__(self, name, description, dtype, unspecified, clean_name=None,
                 free_response=free_response):
        AgQuestion.__init__(self, name, description, dtype,
                            clean_name=clean_name, free_response=free_response)
        self.unspecified = unspecified

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

# ag_data_dictionary = {
#     '': 
# }