from functools import lru_cache

import pandas as pd


def zip_to_state(zipcode, ret='abbr'):
    # Ensure param is a string to prevent unpredictable parsing
    if not isinstance(zipcode, str):
        print('Please enter your ZIP code as a string.')
        return None

    # Don't parse codes that start with 0 as octal values
    try:
        zip = int(zipcode)
    except ValueError:
        return None

    # Code blocks alphabetized by state
    if 35000 <= zip <= 36999:
        abbr = 'AL'
        fullName = 'Alabama'
    elif 99500 <= zip <= 99999:
        abbr = 'AK'
        fullName = 'Alaska'
    elif 85000 <= zip <= 86999:
        abbr = 'AZ'
        fullName = 'Arizona'
    elif 71600 <= zip <= 72999:
        abbr = 'AR'
        fullName = 'Arkansas'

    elif 90000 <= zip <= 96699:
        abbr = 'CA'
        fullName = 'California'

    elif zip >= 80000 and zip <= 81999:
        abbr = 'CO'
        fullName = 'Colorado'

    elif (zip >= 6000 and zip <= 6999):
        abbr = 'CT'
        fullName = 'Connecticut'

    elif (zip >= 19700 and zip <= 19999):
        abbr = 'DE'
        fullName = 'Delaware'

    elif (zip >= 32000 and zip <= 34999):
        abbr = 'FL'
        fullName = 'Florida'

    elif (zip >= 30000 and zip <= 31999):
        abbr = 'GA'
        fullName = 'Georgia'

    elif (zip >= 96700 and zip <= 96999):
        abbr = 'HI'
        fullName = 'Hawaii'

    elif (zip >= 83200 and zip <= 83999):
        abbr = 'ID'
        fullName = 'Idaho'

    elif (zip >= 60000 and zip <= 62999):
        abbr = 'IL'
        fullName = 'Illinois'

    elif (zip >= 46000 and zip <= 47999):
        abbr = 'IN'
        fullName = 'Indiana'

    elif (zip >= 50000 and zip <= 52999):
        abbr = 'IA'
        fullName = 'Iowa'

    elif (zip >= 66000 and zip <= 67999):
        abbr = 'KS'
        fullName = 'Kansas'

    elif (zip >= 40000 and zip <= 42999):
        abbr = 'KY'
        fullName = 'Kentucky'

    elif (zip >= 70000 and zip <= 71599):
        abbr = 'LA'
        fullName = 'Louisiana'

    elif (zip >= 3900 and zip <= 4999):
        abbr = 'ME'
        fullName = 'Maine'

    elif (zip >= 20600 and zip <= 21999):
        abbr = 'MD'
        fullName = 'Maryland'

    elif (zip >= 1000 and zip <= 2799):
        abbr = 'MA'
        fullName = 'Massachusetts'

    elif (zip >= 48000 and zip <= 49999):
        abbr = 'MI'
        fullName = 'Michigan'

    elif (zip >= 55000 and zip <= 56999):
        abbr = 'MN'
        fullName = 'Minnesota'

    elif (zip >= 38600 and zip <= 39999):
        abbr = 'MS'
        fullName = 'Mississippi'

    elif (zip >= 63000 and zip <= 65999):
        abbr = 'MO'
        fullName = 'Missouri'

    elif (zip >= 59000 and zip <= 59999):
        abbr = 'MT'
        fullName = 'Montana'

    elif (zip >= 27000 and zip <= 28999):
        abbr = 'NC'
        fullName = 'North Carolina'

    elif (zip >= 58000 and zip <= 58999):
        abbr = 'ND'
        fullName = 'North Dakota'

    elif (zip >= 68000 and zip <= 69999):
        abbr = 'NE'
        fullName = 'Nebraska'

    elif (zip >= 88900 and zip <= 89999):
        abbr = 'NV'
        fullName = 'Nevada'

    elif (zip >= 3000 and zip <= 3899):
        abbr = 'NH'
        fullName = 'New Hampshire'

    elif (zip >= 7000 and zip <= 8999):
        abbr = 'NJ'
        fullName = 'New Jersey'

    elif (zip >= 87000 and zip <= 88499):
        abbr = 'NM'
        fullName = 'New Mexico'

    elif (zip >= 10000 and zip <= 14999):
        abbr = 'NY'
        fullName = 'New York'

    elif (zip >= 43000 and zip <= 45999):
        abbr = 'OH'
        fullName = 'Ohio'

    elif (zip >= 73000 and zip <= 74999):
        abbr = 'OK'
        fullName = 'Oklahoma'

    elif (zip >= 97000 and zip <= 97999):
        abbr = 'OR'
        fullName = 'Oregon'

    elif (zip >= 15000 and zip <= 19699):
        abbr = 'PA'
        fullName = 'Pennsylvania'

    elif (zip >= 300 and zip <= 999):
        abbr = 'PR'
        fullName = 'Puerto Rico'

    elif (zip >= 2800 and zip <= 2999):
        abbr = 'RI'
        fullName = 'Rhode Island'

    elif (zip >= 29000 and zip <= 29999):
        abbr = 'SC'
        fullName = 'South Carolina'

    elif (zip >= 57000 and zip <= 57999):
        abbr = 'SD'
        fullName = 'South Dakota'

    elif (zip >= 37000 and zip <= 38599):
        abbr = 'TN'
        fullName = 'Tennessee'

    elif ((zip >= 75000 and zip <= 79999) or (zip >= 88500 and zip <= 88599)):
        abbr = 'TX'
        fullName = 'Texas'

    elif (zip >= 84000 and zip <= 84999):
        abbr = 'UT'
        fullName = 'Utah'

    elif (zip >= 5000 and zip <= 5999):
        abbr = 'VT'
        fullName = 'Vermont'

    elif (zip >= 22000 and zip <= 24699):
        abbr = 'VA'
        fullName = 'Virgina'

    elif (zip >= 20000 and zip <= 20599):
        abbr = 'DC'
        fullName = 'Washington DC'

    elif (zip >= 98000 and zip <= 99499):
        abbr = 'WA'
        fullName = 'Washington'

    elif (zip >= 24700 and zip <= 26999):
        abbr = 'WV'
        fullName = 'West Virginia'

    elif 53000 <= zip <= 54999:
        abbr = 'WI'
        fullName = 'Wisconsin'

    elif 82000 <= zip <= 83199:
        abbr = 'WY'
        fullName = 'Wyoming'

    else:
        abbr = 'none'
        fullName = 'none'

    ret = ret.lower()
    if ret in ('abbr', 'abbreviation'):
        return abbr
    if ret in ("full_name", 'full name', 'full', 'name'):
        return fullName
    return abbr, fullName


@lru_cache(5)
def zip_to_dma_df():
    zip_to_dma = pd.read_csv(
        "https://gist.githubusercontent.com/clarkenheim/023882f8d77741f4d5347f80d95bc259/raw/f9f3424dbe4fb58b3dac65dced4c1c3a0f0db27a/Zip%2520Codes%2520to%2520DMAs",
        delimiter="\t"
    )
    return zip_to_dma


@lru_cache(5)
def zip_to_dma_dict():
    df = zip_to_dma_df()
    return {k: tuple(v) for k, v in df.set_index('zip_code').iterrows()}


def zip_to_dma(zipcode, ret='name'):
    try:
        zipcode = int(zipcode)
    except ValueError:
        return None
    ret = ret.lower()
    d = zip_to_dma_dict()
    if zipcode not in d:
        return None
    if ret in ('name', 'full name', 'dma name', 'dma_name'):
        return d[zipcode][1]
    elif ret in ('code', 'dma_code'):
        return d[zipcode][0]
    return d[zipcode]



