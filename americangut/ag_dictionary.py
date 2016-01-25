from americangut.question.ag_bool import (AgBool,
                                          AgMultiple,
                                          )
from americangut.question.ag_categorical import (AgCategorical,
                                                 AgClinical,
                                                 AgFrequency,
                                                 )
from americangut.question.ag_continous import AgContinous


def ag_dictionary(name):
    """Returns a question object for the metadata category described

    Parameters
    ----------
    name : str
        The name of the field being returned

    Returns
    -------
    question : AgQuestion
        A question object describing the question and possible
        responses

    Raises
    ------
    ValueError
        If there is not a question object assoicated with that name

    """
    if name not in dictionary:
        raise ValueError('%s is cannot be found in the data dictionary.' % name)
    else:
        return dictionary[name]


def _remap_abx(x):
    if x in {'I have not taken antibiotics in the past year.',
             'I have not taken antibiotics in the past year'}:
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


def _remap_sleep(x):
    if x in {'Less than 5 hours', '5-6 hours'}:
        return 'Less than 6 hours'
    else:
        return x


def _remap_diet(x):
    if x == 'Vegetarian but eat seafood':
        return 'Pescetarian'
    elif x == 'Omnivore but do not eat red meat':
        return 'No red meat'
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

def _remap_weight(x):
    if x == 'Increased more than 10 pounds':
        return 'Weight gain'
    elif x == 'Decreased more than 10 pounds':
        return 'Weight loss'
    else:
        return x


dictionary = {
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
        extremes=['USA', 'United Kingdom'],
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
        order=['Never', 'Rarely (less than once/week)',
               'Occasionally (1-2 times/week)', 'Regularly (3-5 times/week)',
               'Daily']
        ),
    'FLOSSING_FREQUENCY': AgFrequency(
        name='FLOSSING_FREQUENCY',
        description=("How of the participant flosses their teeth"),
        ),
    'FRUIT_FREQUENCY': AgFrequency(
        name='FRUIT_FREQUENCY',
        description=("How often the participant consumes at least 2-3 "
                     "serving of fruit a day"),
        order=['Never', 'Rarely (less than once/week)',
               'Occasionally (1-2 times/week)', 'Regularly (3-5 times/week)',
               'Daily']
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
        combine='rarely',
        ),
    'IBD': AgClinical(
        name='IBD',
        description=("Has the participant been diagnosed with inflammatory "
                     "bowel disease, include crohns disease or ulcerative "
                     "colitis?"),
        clean_name='IBD',
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
        remap=_remap_last_travel,
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
        order=['Never', 'Rarely (less than once/week)',
               'Occasionally (1-2 times/week)', 'Regularly (3-5 times/week)',
               'Daily'],
        ),
    'ONE_LITER_OF_WATER_A_DAY_FREQUENCY': AgFrequency(
        name="ONE_LITER_OF_WATER_A_DAY_FREQUENCY",
        description=("How frequently does the participant consume at least"
                     " one liter of water"),
        order=['Never', 'Rarely (less than once/week)',
               'Occasionally (1-2 times/week)', 'Regularly (3-5 times/week)',
               'Daily'],
        combine='rarely'
        ),
    'POOL_FREQUENCY': AgFrequency(
        name="POOL_FREQUENCY",
        description=("How often the participant uses a pool or hot tub"),
        combine='weekly',
        extremes=["Never", "Daily"]
        ),
    'PREPARED_MEALS_FREQUENCY': AgFrequency(
        name="PREPARED_MEALS_FREQUENCY",
        description=("frequency with which the participant eats out at a "
                     "resturaunt, including carryout"),
        order=['Never', 'Rarely (less than once/week)',
               'Occasionally (1-2 times/week)', 'Regularly (3-5 times/week)',
               'Daily'],
        clean_name='Resturaunt Frequency'
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
        combine='weekly',
        ),
    'SUGARY_SWEETS_FREQUENCY': AgFrequency(
        name="SUGARY_SWEETS_FREQUENCY",
        description=("how often does the participant eat sweets (candy, "
                     "ice-cream, pastries etc)?"),
        order=['Never', 'Rarely (less than once/week)',
               'Occasionally (1-2 times/week)', 'Regularly (3-5 times/week)',
               'Daily'],
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
        combine='rarely',
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
        extremes=['Remained stable', 'Increased more than 10 pounds'],
        remap=_remap_weight,
        ),
    }

