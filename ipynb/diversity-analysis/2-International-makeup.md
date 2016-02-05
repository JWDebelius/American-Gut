The goal of this notebook is to provide demographic summaries for participants in the American Gut and assoicated projects. We look at metadata, and summarize the avaliable information.

The information generated here will be used for table 1 the American Gut paper.

We'll start by importing the necessary libraries.

```python
>>> import os
...
>>> import numpy as np
>>> import pandas as pd
```

Next, we're going to locate the mapping file and raw OTU table.

```python
>>> processing_dir = os.path.abspath('../primary-processing/agp_processing/')
>>> map_fp = os.path.join(processing_dir, '01-raw/metadata.txt')
>>> otu_fp = os.path.join(processing_dir, '03-otus/100nt/gg-13_8-97-percent/otu_table.biom')
```

We'll start by reading in the mapping file.

```python
>>> map_ = pd.read_csv(map_fp, sep='\t', dtype=str, na_values=['NA', 'no_data', 'unknown', 'Unspecified', 'Unknown', 'None'])
>>> map_.set_index('#SampleID', inplace=True)
```

We're going to start by dropping the blank samples. These are samples where the body habitat is not defined.

```python
>>> map_ = map_[map_['BODY_HABITAT'].notnull()]
```

Next, we're going to get a single sample for each person. The `HOST_SUBEJCT_ID` identfies the individual.

```python
>>> single = []
>>> for bodysite, sub_map in map_.groupby('BODY_HABITAT'):
...     for indv, ids in sub_map.groupby('HOST_SUBJECT_ID').groups.iteritems():
...         single.append(ids[0])
...
>>> map_['Participants'] = np.nan
>>> map_.loc[single, 'Participants'] = 1
>>> map_['Samples'] = 1
```

```python
>>> map_[['Samples', 'Participants']].sum()
Samples         6793
Participants    6006
dtype: float64
```

```python
>>> map_.groupby('BODY_HABITAT').count()[['Samples', 'Participants']]
                    Samples  Participants
BODY_HABITAT                             
UBERON:feces           5952          5380
UBERON:hair               5             5
UBERON:nose               7             7
UBERON:oral cavity      477           436
UBERON:skin             337           165
UBERON:vagina            15            13
```

```python
>>> print map_['AGE_YEARS'].astype(float).min(), map_['AGE_YEARS'].astype(float).max()
0.0 101.0
```

```python
>>> print map_['BMI'].astype(float).min(), map_['BMI'].astype(float).max()
0.0 1130000.0
```

```python
>>> import seaborn as sn
>>> % matplotlib inline
>>> sn.distplot(map_['BMI'].astype(float).dropna())
```

```python
>>> sn.distplot(map_.loc[map_.BMI.astype(float).apply(lambda x: x < 100), 'BMI'].astype(float).dropna())
```

Now, let's look at the countries where participants come from.


```python
>>> countries = md.groupby('COUNTRY').count()[['Samples', 'Participants']]
>>> print 'There are %i countries and soverign states represented here.' % len(countries)
>>> countries.sort_values('Samples', ascending=False)
```

    There are 28 countries and soverign states represented here.

<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Samples</th>
      <th>Participants</th>
    </tr>
    <tr>
      <th>COUNTRY</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>USA</th>
      <td>5604</td>
      <td>4589</td>
    </tr>
    <tr>
      <th>United Kingdom</th>
      <td>721</td>
      <td>560</td>
    </tr>
    <tr>
      <th>Australia</th>
      <td>159</td>
      <td>149</td>
    </tr>
    <tr>
      <th>Belgium</th>
      <td>87</td>
      <td>9</td>
    </tr>
    <tr>
      <th>Canada</th>
      <td>75</td>
      <td>70</td>
    </tr>
    <tr>
      <th>Switzerland</th>
      <td>12</td>
      <td>12</td>
    </tr>
    <tr>
      <th>Germany</th>
      <td>11</td>
      <td>11</td>
    </tr>
    <tr>
      <th>Thailand</th>
      <td>9</td>
      <td>9</td>
    </tr>
    <tr>
      <th>France</th>
      <td>8</td>
      <td>8</td>
    </tr>
    <tr>
      <th>Czech Republic</th>
      <td>7</td>
      <td>7</td>
    </tr>
    <tr>
      <th>Netherlands</th>
      <td>6</td>
      <td>6</td>
    </tr>
    <tr>
      <th>Norway</th>
      <td>5</td>
      <td>5</td>
    </tr>
    <tr>
      <th>New Zealand</th>
      <td>5</td>
      <td>5</td>
    </tr>
    <tr>
      <th>Spain</th>
      <td>4</td>
      <td>3</td>
    </tr>
    <tr>
      <th>Italy</th>
      <td>4</td>
      <td>4</td>
    </tr>
    <tr>
      <th>Poland</th>
      <td>3</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Japan</th>
      <td>3</td>
      <td>3</td>
    </tr>
    <tr>
      <th>China</th>
      <td>3</td>
      <td>2</td>
    </tr>
    <tr>
      <th>Ireland</th>
      <td>2</td>
      <td>2</td>
    </tr>
    <tr>
      <th>Sweden</th>
      <td>2</td>
      <td>2</td>
    </tr>
    <tr>
      <th>Isle of Man</th>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Puerto Rico</th>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>India</th>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Finland</th>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Denmark</th>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Brazil</th>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>United Arab Emirates</th>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>Jersey</th>
      <td>1</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>
