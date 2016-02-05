import numpy as np

from americangut.question.ag_question import AgQuestion


class AgContinous(AgQuestion):

    def __init__(self, name, description, unit=None, limits=None,
                 clean_name=None, remap=None, free_response=False,
                 mimarks=False, ontology=None):
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
        limits : two-elemant iterable
            The range of values pertinant to analysis.
        """

        AgQuestion.__init__(self, name, description, float, clean_name=None,
                            remap=None, free_response=False, mimarks=False,
                            ontology=None)
        self.unit = unit
        if limits is not None:
            lower, upper = limits
        else:
            lower, upper = (None, None)

        if lower > upper:
            raise ValueError('The lower limit cannot be greater than '
                             'the upper!')
        self.lower = lower
        self.upper = upper
        self.type = 'Continous'

    def remap_data_type(self, map_):
        """Makes sure the target column in map_ has the correct datatype

        map_ : DataFrame
            A pandas dataframe containing the column described by the question
            name.

        """
        def remap_(x):
            return self.dtype(x)

        map_[self.name] = map_[self.name].apply(remap_).astype(self.dtype)
        map_.replace('nan', np.nan, inplace=True)

    def drop_outliers(self, map_):
        """Removes datapoints outside of the `range`"""
        self.check_map(map_)

        if self.lower is not None:

            def remap_(x):
                if x < self.lower:
                    return np.nan
                elif x > self.upper:
                    return np.nan
                else:
                    return x

            map_[self.name] = map_[self.name].apply(remap_)
