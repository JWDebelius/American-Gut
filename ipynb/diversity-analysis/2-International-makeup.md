The goal of this notebook is to provide demographic summaries for participants in the American Gut and assoicated projects. We look at metadata, and summarize the avaliable information.

The information generated here will be used for table 1 the American Gut paper.

We'll start by importing the necessary libraries.

```python
>>> import os
>>> from functools
>>> import partial
>>> 
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
>>> md = pd.read_csv(map_fp, sep='\t', dtype=str, na_values=['NA', 'no_data', 'unknown', 'Unspecified', 'Unknown', 'None'])
>>> md.set_index('#SampleID', inplace=True)
```

We're also going to calculate the number of sequences per sample, and add that to our mapping file.

```python
>>> otu_summary = !biom summarize-table -i $otu_fp
>>> seq_depth = pd.DataFrame(np.array([l.split(': ') for l in otu_summary[15:]]), columns=['#SampleID', 'counts'])
>>> seq_depth.set_index('#SampleID', inplace=True)
>>> 
>>> md['count'] = seq_depth
```

We're going to start by dropping the blank samples. These are samples where the body habitat is not defined.


```python
>>> md = md[md['BODY_HABITAT'].notnull()]
```

Next, we're going to get a single sample for each person. The `HOST_SUBEJCT_ID` identfies the individual.


```python
>>> single = []
>>> for indv, ids in md.groupby('HOST_SUBJECT_ID').groups.iteritems():
...     single.append(ids[0])

>>> md['Participants'] = np.nan
>>> md['Samples'] = 1
>>> md.loc[single, 'Participants'] = 1
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
