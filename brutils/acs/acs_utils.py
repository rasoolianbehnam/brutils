from dataclasses import dataclass
from typing import List
from unittest.case import TestCase

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup

from brutils import config
from brutils.utility import cartesian_product_strict

VARIABLES_URL = "https://api.census.gov/data/{year}/acs/acs5/variables.html"
STATE_URL_TEMPLATE = "https://api.census.gov/data/{year}/acs/acs5?get=NAME,{variables}&for={on}"
ZIP_URL_TEMPLATE = "https://api.census.gov/data/{year}/acs/acs5?get=NAME,{variables}&for=zip%20code%20tabulation%20area"
OUTPUT_PATH = config.root + "acs_utils/"
acs_pop_funs = {}


def acs_pop_fun(*args):
    def decorator(fun):
        acs_pop_funs[tuple(args)] = fun
        return fun

    return decorator


def clean_age(x, max_age=120):
    a = x.replace('  and over', f"-{max_age}").replace('Under ', '1-').replace(' to ', '-').replace(' and ', '-')
    beg, *end = a.split('-')
    if len(end) == 0:
        end = beg
    else:
        end = end[0]
    return range(int(beg), int(end) + 1)


def pick_age_gender_only_data(data: pd.DataFrame):
    age_gender_condition = data['ts_description'].str.len() > 7
    data_age_gender = data[age_gender_condition].copy()
    return data_age_gender


def clean_age_gender_based_on_gracenote(us_pop_age_gender: pd.DataFrame, on: str):
    age_gender_mapping = pd.read_parquet(config.GR_AM_AGE_GENDER_MAP)[['gender', 'personAge', 'personGender']] \
        .drop_duplicates()
    age_gender_mapping[['min_age', 'max_age']] = age_gender_mapping['personAge'].str.replace('+', '-120') \
        .str.split('-', expand=True).astype('int')
    res = cartesian_product_strict(us_pop_age_gender, age_gender_mapping, extra_on=['gender'],
                                   suffixes=['_acs', ''])
    condition = (res['age_int'] >= res['min_age']) & (res['age_int'] <= res['max_age'])
    acs_us_pop_by_age_gender_state = res[condition].groupby([on, 'personAge', 'personGender'],
                                                            as_index=False)['population'].sum()
    return acs_us_pop_by_age_gender_state


def get_population_for_individual_ages(data: pd.DataFrame):
    data_age_gender = data.copy()
    data_age_gender[['gender', 'age']] = data_age_gender['ts_description'].str.split(':', expand=True)
    data_age_gender['gender'] = data_age_gender['gender'].str[:1]
    data_age_gender['age_int'] = data_age_gender['age'].map(clean_age)
    data_age_gender['population'] /= data_age_gender['age_int'].map(len)
    return data_age_gender.explode('age_int')


def get_acs_url(age_gender_variables, on, year):
    if on in {'state', 'county'}:
        age_gender_data_url = STATE_URL_TEMPLATE.format(on=on, year=year, variables=','.join(age_gender_variables))
    elif on == 'zip':
        age_gender_data_url = ZIP_URL_TEMPLATE.format(year=year, variables=','.join(age_gender_variables))
    else:
        raise ValueError('on must be in {`state`, `zip`, `county`}')
    return age_gender_data_url


@dataclass
class ACS:
    year: int
    variables: pd.DataFrame = None

    def __post_init__(self):
        response = requests.get(VARIABLES_URL.format(year=self.year))
        bs = BeautifulSoup(response.content)
        body = bs.find(name='tbody')
        rows = body.findAll('tr')
        table = [[col.text for col in row.findAll('td')] for row in rows]
        columns = """Name	Label	Concept	Required	Attributes	Limit	Predicate_Type	Group""".lower().split()
        if self.variables is None:
            variables = pd.DataFrame(table, columns=columns)
            self.variables = variables

    @acs_pop_fun('person', 'age', 'gender')
    def get_acs_persons_pop_by_age_gender(self, on: str):
        data = self.get_raw_person_population(on)

        data_age_gender = pick_age_gender_only_data(data)
        assert data_age_gender['population'].sum() == data[data['ts_description'].str.len() == 0]['population'].sum()

        us_pop_age_gender = get_population_for_individual_ages(data_age_gender)
        assert np.allclose(data_age_gender['population'].sum(), us_pop_age_gender['population'].sum())
        acs_us_pop_by_age_gender_state = clean_age_gender_based_on_gracenote(us_pop_age_gender, on)
        assert acs_us_pop_by_age_gender_state['population'].sum() / us_pop_age_gender['population'].sum() > .98
        return acs_us_pop_by_age_gender_state

    def get_raw_person_population(self, on):
        group_id = 'B01001'
        data = self.get_acs_population_by_group_id(on, group_id)
        data['ts_description'] = data['ts_description'].str.replace('years', '')
        return data

    def get_acs_population_by_group_id(self, on, group_id, rplc=''):
        age_gender_labels, age_gender_variables = self.get_variables_by_group_id(group_id)
        age_gender_data_url = get_acs_url(age_gender_variables, on, self.year)
        response = requests.get(age_gender_data_url, )
        data = pd.DataFrame(response.json()).iloc[:, :len(age_gender_labels) + 1]
        data.drop(0, inplace=True)
        data.columns = [on, *age_gender_labels]
        data = data.melt(id_vars=on, var_name='ts_description', value_name='population')
        data['population'] = data['population'].astype('int')
        data['ts_description'] = data['ts_description'].str.replace('Estimate!!Total:', '') \
            .str.replace('!', rplc)
        return data

    def get_variables_by_group_id(self, group_id, ):
        variables = self.variables
        age_gender_variables = variables[variables['group'] == group_id]['name'].values
        age_gender_labels = variables[variables['group'] == group_id]['label'].values
        return age_gender_labels, age_gender_variables

    @acs_pop_fun('household', 'hh_size')
    def get_acs_hh_population_by_hh_size(self, on: str):
        data = self.get_raw_hh_population(on)
        data_hh_size = self.pick_hh_size_only_data(data)
        assert data_hh_size['population'].sum() == data[data['ts_description'].str.len() == 0]['population'].sum()
        data_hh_size['ts_description'] = data_hh_size['ts_description'].str.replace('person household', '')
        out = self.clean_household_size(data_hh_size, on=on)
        return out

    def get_raw_hh_population(self, on):
        group_id = 'B11016'
        data = self.get_acs_population_by_group_id(on, group_id)
        return data

    def pick_hh_size_only_data(self, data: pd.DataFrame):
        condition = data['ts_description'].str.len() > 21
        return data[condition].copy()

    def clean_household_size(self, df: pd.DataFrame, on: str):
        data = df.copy()
        data['hh_size'] = data['ts_description'].str.split(':', expand=True)[1].str[:1].astype('int')
        out = data.groupby([on, 'hh_size'], as_index=False)['population'].sum()
        return out

    @acs_pop_fun('person', 'race')
    def get_acs_person_pop_by_race(self, on: str):
        data = self.get_acs_population_by_group_id(on=on, group_id='B02001')
        condition1 = data['ts_description'] == ''
        condition2 = data['ts_description'] == 'Two or more races:'
        data_cleaned = data.loc[~condition1 & ~condition2]
        assert data_cleaned['population'].sum() == data.loc[condition1, 'population'].sum()
        return data_cleaned

    @acs_pop_fun('household', 'race')
    def get_acs_hh_pop_by_race(self, on: str):
        data = self.get_acs_population_by_group_id(on=on, group_id='B25006')
        condition1 = data['ts_description'] == ''
        condition2 = data['ts_description'] == 'Householder who is Two or more races:'
        data_cleaned = data.loc[~condition1 & ~condition2]
        assert data_cleaned['population'].sum() == data.loc[condition1, 'population'].sum()
        return data_cleaned

    @acs_pop_fun('person', 'origin')
    def get_acs_person_pop_by_origin(self, on: str):
        data = self.get_acs_population_by_group_id(on=on, group_id='B03001')
        condition0 = data['ts_description'] == ''
        condition1 = data['ts_description'] == 'Not Hispanic or Latino'
        condition2 = data['ts_description'] == 'Hispanic or Latino:'
        data_cleaned = data.loc[condition1 | condition2]
        assert data_cleaned['population'].sum() == data.loc[condition0, 'population'].sum()
        return data_cleaned

    def GetAcsBroadbandByRace(acs, on: str):
        d_ = acs.variables[acs.variables.group.str.startswith('B28009')][['concept', 'group']].drop_duplicates()
        results = []
        for _, (concept, group) in d_.iterrows():
            out = acs.get_acs_population_by_group_id(on=on, group_id=group, rplc='-') \
                .query("ts_description=='--Has a computer:--With a broadband Internet subscription'") \
                .assign(concept=concept)
            results.append(out)
        broadband_by_race = pd.concat(results).drop('ts_description', 1)
        broadband_by_race['race'] = broadband_by_race.concept.str.replace(r'.*\((.*)\).*', r"\1", regex=True)
        return broadband_by_race


    def GetAcsPersonsEducationByBroadband(acs, on: str):
        tmp = acs.get_acs_population_by_group_id(on, 'B28006').query("ts_description != ''")
        c = tmp.ts_description.str.split(':', expand=True)
        tmp[['education', 'computer', 'internet']] = c
        tmp = tmp.query("internet.fillna('').str.contains('broadband')")
        return tmp


    def GetAcsPersonsEducation(acs, on: str):
        tmp = acs.get_acs_population_by_group_id(on, 'B15003') \
            .query("ts_description != ''") \
            .rename(columns={'ts_description': 'education'})
        return tmp


    def GetAcsPersonsEducationWithMapAndSeed(acs, on):
        tmp = acs.GetAcsPersonsEducation(on=on).assign(age2='25 years and over')
        mp = pd.DataFrame(np.array([
            '12-17', 'under 25 years',
            '18-24', 'under 25 years',
            '25-34', '25 years and over',
            '35-49', '25 years and over',
            '50-54', '25 years and over',
            '55-64', '25 years and over',
            '65+', '25 years and over',
            '2-5', 'under 25 years',
            '6-11', 'under 25 years',
        ]).reshape(-1, 2), columns=['personAge', 'age2'])
        seed = tmp[['education', 'age2']].drop_duplicates()
        seed.loc[0] = ['not applicable', 'under 25 years']
        return tmp, mp, seed


    def GetAcsHouseholdIncome(acs, on: str):
        tmp = acs.get_acs_population_by_group_id(on, 'B19001') \
            .query("ts_description != ''") \
            .rename(columns={'ts_description': 'income'})
        return tmp


    def GetAcsHouseholdIncomeBroadband(acs, on: str):
        tmp = acs.get_acs_population_by_group_id(on, 'B28004').query("ts_description != ''")
        c = tmp.ts_description.str.split(':', expand=True)
        tmp[['income', 'internet']] = c
        tmp = tmp.query("internet.fillna('').str.contains('broadband')")
        tmp = tmp.drop(['ts_description', 'internet'], 1)
        return tmp


    def GetAcsPersonsRaceByOrigin(acs, on: str):
        df = acs.get_acs_population_by_group_id(on=on, group_id='B03002')
        df.query("ts_description != ''").population.sum()
        x = df.ts_description.str.split(':', expand=True)
        df[['origin', 'pp_race', 'other']] = x
        person_race_origin = df.query("origin != '' and pp_race != '' and other != ''").drop(['other', 'ts_description'], 1)
        return person_race_origin


    def GetAcsPersonsBroadbandByAge(acs, on: str):
        tmp = acs.get_acs_population_by_group_id(on, 'B28005') \
            .query("ts_description.str.contains('broadband')") \
            .assign(age=lambda df: df.ts_description.str.split(":", expand=True)[0]) \
            .drop('ts_description', 1)
        return tmp

def get_dma_pop_from_county(us_pop_county: pd.DataFrame, demo: List[str]):
    """
    :param us_pop_county: must have `county` column which is in the format "{county}, {full state name}"
    :param demo: demographic columns such as ['personAge', 'personGender'] or ['household_size'] etc.
    :return:
    """
    T = TestCase()
    columns = list(us_pop_county.columns)
    if not set(demo).issubset(set(columns)):
        raise ValueError(f"demo ({demo}) must be in columns ({columns})")
    us_pop_county[['county', 'state']] = (
        us_pop_county['county'].str.split(', ', expand=True)
    )
    us_pop_county['county'] = (
        us_pop_county['county'].str.replace(
            r'( County)|( Parish)|( Municipio)|( City)|( Borough)|( city)'
            r'|( Municipality)|( Census Area)',
            '')
    )

    dma_county_map = pd.read_parquet(
        config.DMA_COUNTY_MAP_PATH)

    zip_county_map = pd.read_parquet(
        config.ZIP_COUNTY_MAP_PATH)
    zip_county_map.columns = zip_county_map.columns.str.lower()
    dma_county_map.columns = dma_county_map.columns.str.lower()
    state_abbr = pd.read_parquet(
        config.STATE_ABBR_MAP_PATH
    )
    state_abbr.columns = ['state_name', 'state']

    us_pop_county_with_dma = us_pop_county.merge(
        state_abbr, left_on='state', right_on='state_name', suffixes=['_x', '']) \
        .merge(dma_county_map, on=['state', 'county'], how='left')
    print("[*] Counties not matched:\n",
          us_pop_county_with_dma[us_pop_county_with_dma['dma'].isna()]['county'].unique())

    res2 = us_pop_county_with_dma[~us_pop_county_with_dma['dma'].isna()]
    print("[*] DMA/County Population ratio due to lack of mapping:", len(res2) / len(us_pop_county))
    T.assertGreater(len(res2) / len(us_pop_county), .95)
    us_pop_age_gender_dma = res2.groupby(
        ['dma', *demo], as_index=False)['population'].sum()
    return us_pop_age_gender_dma


