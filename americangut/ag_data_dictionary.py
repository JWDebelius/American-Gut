import copy
import inspect
import types

import numpy as np
import pandas as pd


class AgQuestion:
    true_values = {'yes', 'y', 'true', 1}
    false_values = {'no', 'n', 'false', 0}

    def __init__(self, name, description, dtype, clean_name=None,
                 **kwargs):
        """A base object for describing single question outputs

        Parameters
        ----------
        name : str
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
        self.type = 'Question'
        if clean_name is None:
            self.clean_name = name.replace('_', ' ').title()
        else:
            self.clean_name = clean_name
        self.free_response = kwargs.get('free_response', False)
        self.ontology = kwargs.get('ontology', None)
        self.frequency_cutoff = kwargs.get('frequency_cutoff', None)
        self.remap_ = kwargs.get('remap', None)
        self.extremes = kwargs.get('extremes', None)
        self.subset = kwargs.get('subset', True)

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

    def __init__(self, name, description, dtype, groups, **kwargs):
        """..."""
        if dtype not in {str, bool, int, float}:
            raise ValueError('%s is not a supported datatype for a '
                             'categorical variable.' % dtype)
        AgQuestion.__init__(self, name, description, dtype, **kwargs)
        self.type = 'Categorical'
        self.groups = groups
        self.earlier_groups = []
        self.frequency_cutoff = kwargs.get('frequency_cutoff', None)
        self.drop_ambiguous = kwargs.get('drop_ambiguous', True)
        self.ambiguous_values = kwargs.get(
            'ambiguous_values',
            {"None of the above",
             "Not sure",
             "I don't know, I do not have a point of reference"
             })

    def remove_ambiguity(self, map_):
        """Removes ambiguous groups from the mapping file"""
        if self.drop_ambiguous:
            self.check_map(map_)
            self.remap_data_type(map_)

            def _remap(x):
                if x in self.ambiguous_values:
                    return np.nan
                else:
                    return x

            map_[self.name] = map_[self.name].apply(_remap)
            self._update_groups(_remap)

    def remap_groups(self, map_):
        """Remaps columns in the column"""
        self.check_map(map_)
        self.remap_data_type(map_)

        if self.remap_ is not None:
            map_[self.name] = map_[self.name].apply(self.remap_)
            self._update_groups(self.remap_)

    def drop_infrequent(self, map_):
        """Removes groups below a frequency cutoff"""
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
        self._update_groups(_remap)

    def _update_groups(self, remap_):
        self.earlier_groups.append(copy.copy(self.groups))
        self.groups = (set([remap_(g) for g in self.earlier_groups[-1]]) -
                       {np.nan})


class AgBool(AgCategorical):
    def __init__(self, name, description, **kwargs):
        """..."""
        AgCategorical.__init__(self, name, description, dtype=bool,
                               groups={True, False}, **kwargs)
        self.type = 'Bool'
        self.extremes = [True, False]


class AgMultiple(AgBool):
    def __init__(self, name, description, unspecified, **kwargs):
        """..."""
        AgBool.__init__(self, name, description, **kwargs)
        self.type = 'Multiple'
        self.unspecified = unspecified

    def correct_unspecified(self, map_):
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
        groups = {'Diagnosed by a medical professional (doctor, physician '
                  'assistant)',
                  'Diagnosed by an alternative medicine practitioner',
                  'Self-diagnosed',
                  'I do not have this condition'
                  }
        AgCategorical.__init__(self, name, description, str, groups, **kwargs)
        self.strict = strict
        self.type = 'Clinical'
        self.extremes = ['I do not have this condition',
                         'Diagnosed by a medical professional (doctor, '
                         'physician assistant)']

    def remap_clinical(self, map_):
        self.check_map(map_)
        self.remap_data_type(map_)
        if self.strict:
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
        self._update_groups(remap_)
        self.extremes = ['No', 'Yes']


class AgOrdinal(AgQuestion):
    def __init__(self, name, description, dtype, order, **kwargs):
        AgQuestion.__init__(self, name, description, dtype, **kwargs)
        self.order = order
        self.earlier_order = []
        self.type = 'Ordinal'

    def _update_order(self, remap_):
        """Updates the order and earlier order arguments"""
        order = copy.copy(self.order)
        self.earlier_order.append(order)
        self.order = []
        for o in order:
            new_o = remap_(o)
            if new_o not in self.order:
                self.order.append(new_o)

    def remap_groups(self, map_):
        """Remaps columns in the column"""
        self.check_map(map_)
        self.remap_data_type(map_)

        if self.remap_ is not None:
            map_[self.name] = map_[self.name].apply(self.remap_)
            self._update_order(self.remap_)

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
        self._update_order(remap_)

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
        self._update_order(remap_)


class AgFrequency(AgOrdinal):
    def __init__(self, name, description, **kwargs):
        order = ['Never',
                 'Rarely (a few times/month)',
                 'Occasionally (1-2 times/week)',
                 'Regularly (3-5 times/week)',
                 'Daily']
        AgOrdinal.__init__(self, name, description, dtype=str, order=order,
                           **kwargs)
        self.type = 'Frequency'
        self.extremes = kwargs.get('extremes', ['Never', 'Daily'])

    def clean(self, map_):
        self.check_map(map_)

        def remap_(x):
            if isinstance(x, str) and '(' in x:
                return x.split('(')[1].replace(')', '').capitalize()
            else:
                return x

        map_[self.name] = map_[self.name].apply(remap_)
        self._update_order(remap_)


class AgContinous(AgQuestion):
    type = 'Continous'

    def __init__(self, name, description, unit=None, **kwargs):
        AgQuestion.__init__(self, name, description, float, **kwargs)
        self.unit = unit
        lower, upper = kwargs.get('range', [None, None])
        if lower > upper:
            raise ValueError('The lower limit cannot be greater than '
                             'the upper!')
        self.lower = lower
        self.upper = upper

    def drop_outliers(self, map_):
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
    if x == ('I have not been outside of my country of residence in'
             ' the past year.'):
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

ag_data_dictionary = {
    'AGE_CAT': AgOrdinal(
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
    'ANTIBIOTIC_HISTORY': AgOrdinal(
        name='ANTIBIOTIC_HISTORY',
        description=('How recently has the participant taken antibiotics?'),
        dtype=str,
        order=['Week', 'Month', '6 months', 'Year',
               'I have not taken antibiotics in the past year'],
        remap=_remap_abx,
        extremes=['Week', 'More than a year'],
        ),
    'BMI_CAT': AgOrdinal(
        name='BMI_CAT',
        description='The body mass index, categorized according to the WHO.',
        dtype=str,
        order=['Underweight', 'Normal', 'Overweight', 'Obese'],
        extremes=['Normal', 'Obese'],
        ),
    'BOWEL_MOVEMENT_FREQUENCY': AgOrdinal(
        name='BOWEL_MOVEMENT_FREQUENCY',
        description=('Number of daily bowel movements'),
        dtype=str,
        order=['Less than one', 'One', 'Two', 'Three', 'Four', 'Five or more'],
        remap=_remap_bowel_frequency,
        extremes=['Less than One', 'Four or more'],
        ),
    'BOWEL_MOVEMENT_QUALITY': AgCategorical(
        name='BOWEL_MOVEMENT_QUALITY',
        description=('Does the participant tend toward constipation or '
                     'diarrhea?'),
        dtype=str,
        groups={"I don't know, I do not have a point of reference",
                'I tend to be constipated (have difficulty passing stool)',
                'I tend to have diarrhea (watery stool)',
                'I tend to have normal formed stool'},
        remap=_remap_fecal_quality,
        extremes=['Normal', 'Diarrhea'],
        ),
    'CAT': AgBool(
        name='CAT',
        description='Does the participant have a pet cat?',
        ),
    'CHICKENPOX': AgBool(
        name='CHICKENPOX',
        description='Has the participant has chickenpox?'
        ),
    'COLLECTION_SEASON': AgOrdinal(
        name='COLLECTION_SEASON',
        dtype=str,
        description=("Season in which the sample was collected. Winter: "
                     "Dec 1 - Feb 28/29; Spring: March 1-May 31; "
                     "Summer: June 1 - August 31; Fall: Sept 1-Nov 30"),
        order=['Winter', 'Spring', 'Summer', 'Fall'],
        extremes=['Winter', 'Summer']
        ),
    'COLLECTION_MONTH': AgOrdinal(
        name='COLLECTION_MONTH',
        description=('Month in which the sample was collected'),
        dtype=str,
        order=['January', 'February', 'March', 'April', 'May', 'June',
               'July', 'August', 'September', 'October', 'November',
               'December'
               ],
        ),
    'CONSUME_ANIMAL_PRODUCTS_ABX': AgCategorical(
        name="CONSUME_ANIMAL_PRODUCTS_ABX",
        description=("Does the participant eat animal products treated with "
                     "antibiotics?"),
        dtype=str,
        groups={'Yes', 'No', 'Not sure'},
        extremes=['Yes', 'No'],
        ),
    'COUNTRY': AgCategorical(
        name='COUNTRY',
        description=("country of residence"),
        dtype=str,
        ontology='GAZ',
        mimmarks=True,
        groups={'Australia', 'Belgium', 'Brazil', 'Canada', 'China',
                'Czech Republic', 'Denmark', 'Finland', 'France', 'Germany',
                'Ireland', 'Isle of Man', 'Italy', 'Japan', 'Jersey',
                'Netherlands', 'New Zealand', 'Norway', 'Poland', 'Spain',
                'Sweden', 'Switzerland', 'Thailand', 'USA',
                'United Arab Emirates', 'United Kingdom'},
        extremes=['US', 'United Kingdom'],
        frequency_cutoff=50,
        ),
    'CONTRACEPTIVE': AgCategorical(
        name="CONTRACEPTIVE",
        description="Does the participant use contraceptives?",
        dtype=str,
        groups={'No', 'Yes, I am taking the pill',
                'Yes, I use a contraceptive patch (Ortho-Evra)',
                'Yes, I use a hormonal IUD (Mirena)',
                'Yes, I use an injected contraceptive (DMPA)',
                'Yes, I use the NuvaRing'},
        remap=_remap_contraceptive,
        ),
    'CSECTION': AgCategorical(
        name="CSECTION",
        description=("Was the participant born via c-section?"),
        dtype=str,
        groups={'Yes', 'No', 'Not sure'}
        ),
    'DIABETES': AgClinical(
        name="DIABETES",
        description=("Has the participant been diganosed with diabetes"),
        extremes=["No", "Yes"],
        ),
    'DIET_TYPE': AgOrdinal(
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
        groups={'Bottled', 'City', 'Filtered', 'Not sure', 'Well'},
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
        groups={'Both', 'Depends on the season', 'Indoors',
                'None of the above', 'Outdoors'},
        extremes=['Indoors', 'Outdoors'],
        ),
    'FED_AS_INFANT': AgCategorical(
        name='FED_AS_INFANT',
        description='Food source as an infant (breast milk or formula)',
        dtype=str,
        groups={'A mixture of breast milk and formula', 'Not sure',
                'Primarily breast milk', 'Primarily infant formula'},
        extremes=['Primarily breast milk', 'Primarily infant formula'],
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
        groups={'I was diagnosed with celiac disease', 'No',
                'I do not eat gluten because it makes me feel bad',
                'I was diagnosed with gluten allergy (anti-gluten IgG),'
                ' but not celiac disease'},
        extremes=['No', 'Celiac Disease'],
        remap=_remap_gluten
        ),
    'HOMECOOKED_MEALS_FREQUENCY': AgFrequency(
        name='HOMECOOKED_MEALS_FREQUENCY',
        description=("How often does the participant consume meals cooked"
                     " at home?"),
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
    'LAST_TRAVEL': AgOrdinal(
        name="LAST_TRAVEL",
        description=("When the participant last traveled away from home"),
        dtype=str,
        order=['I have not been outside of my country of residence in the '
               'past year.', '1 year', '6 months', '3 months', 'Month'],
        remap_=_remap_last_travel,
        extremes=['More than a year', 'Month'],
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
        extremes=["Never", "More than once a week"]
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
        groups={'Hispanic', 'African American', 'Other',
                'Caucasian', 'Asian or Pacific Islander'},
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
        groups={'female', 'male', 'other'},
        mimmarks=True,
        extreme=['male', 'female']
        ),
    "SLEEP_DURATION": AgOrdinal(
        name="SLEEP_DURATION",
        description=("How long the participant sleeps in the average night?"),
        dtype=str,
        order=['Less than 5 hours', '5-6 hours', '6-7 hours', '7-8 hours',
               '8 or more hours'],
        remap=_remap_sleep,
        extreme=['Less than 6 hours', '8 or more hours'],
        ),
    'SMOKING_FREQUENCY': AgFrequency(
        name="SMOKING_FREQUENCY",
        description="How often the participant smokes?",
        remap=_remap_pool_frequency
        ),
    'SUGARY_SWEET_FREQUENCY': AgFrequency(
        name="SUGARY_SWEET_FREQUENCY",
        description=("how often does the participant eat sweets (candy, "
                     "ice-cream, pastries etc)?"),
        ),
    'TYPES_OF_PLANTS': AgOrdinal(
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
    'VEGETABLE_FREQUENCY': AgFrequency(
        name="VITAMIN_B_SUPPLEMENT_FREQUENCY",
        description=("How often the participant takes a vitamin B or "
                     "vitamin B complex supplement"),
        ),
    'VEGETABLE_FREQUENCY': AgFrequency(
        name="VITAMIN_B_SUPPLEMENT_FREQUENCY",
        description=("How often the participant takes a vitamin D supplement"),
        ),
    'WEIGHT_CHANGE': AgCategorical(
        name="WEIGHT_CHANGE",
        description=("Has the participants weight has changed more than 10 "
                     "lbs in the past year?"),
        dtype=str,
        groups={'Increased more than 10 pounds',
                'Decreased more than 10 pounds', 'Remained stable'},
        extremes=['Remained stable', 'Increased more than 10 pounds']
        ),
    }
