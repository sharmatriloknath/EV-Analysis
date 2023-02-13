# Import Necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob
from typing import List

# DATA FILE
BASE_PATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
DATA_PATH = os.path.join(BASE_PATH,"Data")
FILE_NAME = glob.glob(f"{DATA_PATH}/*.csv")[0]

# READ DATA FOR CHARGING POINTS
def _clean_and_filter_charging_data()->pd.DataFrame:
    # Read Data
    df_ = pd.read_csv(FILE_NAME)
    return (
        df_
            .assign(
                **{c:lambda df_,c=c:df_[c].astype('category') for c in ['region','category','parameter','mode','powertrain','unit']},
                year = lambda df_:df_['year'].astype("int16")
            )
            .query("category=='Historical' and unit=='charging points'")
        )

def _get_european_countries(df_:pd.DataFrame)->pd.DataFrame:
    EUROPEAN_COUNTRIES = ['Europe','Other Europe','Russia','Germany', 
                          'United Kingdom', 'France', 'Italy', 'Spain', 'Ukraine',
                           'Poland', 'Romania', 'Netherlands', 'Belgium',
                           'Czech Republic (Czechia)', 'Greece', 'Portugal', 'Sweden',
                           'Hungary', 'Belarus', 'Austria', 'Serbia', 'Switzerland',
                           'Bulgaria', 'Denmark', 'Finland', 'Slovakia', 'Norway', 'Ireland',
                           'Croatia', 'Moldova', 'Bosnia and Herzegovina', 'Albania',
                           'Lithuania', 'North Macedonia', 'Slovenia', 'Latvia', 'Estonia',
                           'Montenegro', 'Luxembourg', 'Malta', 'Iceland', 'Andorra',
                           'Monaco', 'Liechtenstein', 'San Marino', 'Holy See']
    
    conditions = [
       df_['region'].isin(EUROPEAN_COUNTRIES),
    ]

    choices = [
        'Europe'
    ]

    df_['region_class'] = np.select(conditions, choices, default=df_['region'].values)
    return df_


def get_charger_type_count_by_country(df_:pd.DataFrame, countries: List[str], charger_type:str)->pd.DataFrame:
    return (df_
            .pipe(_get_european_countries)
            .query(f"region_class in {countries} and powertrain=='Publicly available {charger_type}'")
            .assign(
                    powertrain = lambda df_:df_["powertrain"].to_list(),
            )
            .groupby(['region_class','year','powertrain'])['value']
            .sum()
            .reset_index()
            .assign(
                value= lambda df_:df_["value"]/1000
            )
            .sort_values("year")
            .reset_index(drop = True)
           )

def _get_ratio_of_ev_per_cv(df_:pd.DataFrame)->pd.DataFrame:

    # Electrical Vehicles Stock
    ev_stock = (df_
    .query("region not in ['Europe','Other Europe','World','Rest of the world']\
        and unit=='stock' and category=='Historical' and year==2021")
    .groupby("region")['value']
    .sum()
    .reset_index()
    .sort_values("value",ascending=False)
    .set_index("region")
    )

    # Charging points stock
    ev_charging_points = (df_
    .query("region not in ['Europe','Other Europe','World','Rest of the world']\
           and unit=='charging points' and category=='Historical' and year==2021")
    .groupby("region")['value']
    .sum()
    .reset_index()
    .sort_values("value",ascending=False)
    .set_index("region")
    )

    # Merged Two DataFrame
    merged_df = ev_stock.merge(ev_charging_points,left_index=True,right_index=True,suffixes=("_stock","_charging"))

    # Get ratio of charging points per ev
    ev_per_cp = (merged_df
                 .assign(
                    ev_per_charging_points = lambda df_: round(df_['value_stock']/df_['value_charging'],2)
                )
                 .sort_values("ev_per_charging_points", ascending=False)
            )

    return ev_per_cp


def _get_ratio_by_year(df_:pd.DataFrame, year:int)->float:
    
    ev_stock_yrs = (df_
    .query("region not in ['Europe','Other Europe','World','Rest of the world']\
        and unit=='stock' and category=='Historical' and year != 2021")
    .groupby(["region","year"])['value']
    .sum()
    .reset_index()
    .sort_values("value",ascending=False)
    .set_index("year")
    ).loc[year,:]
    
    ev_stock_chrg = (df_
    .query("region not in ['Europe','Other Europe','World','Rest of the world']\
        and unit=='charging points' and category=='Historical' and year != 2021")
    .groupby(["region","year"])['value']
    .sum()
    .reset_index()
    .sort_values("value",ascending=False)
    .set_index("year")
    ).loc[year,:]
    
    return ev_stock_yrs.merge(ev_stock_chrg,left_index=True,right_index=True,suffixes=("_stock","_charging"))


def _get_charging_ratio_of_countries_by_year(df_:pd.DataFrame)->pd.DataFrame:
    dfs = []
    for year in df_.year.unique():
        if year >2020: continue
        data = _get_ratio_by_year(df_,year)
        final_data = (data
                        [data["region_stock"]==data["region_charging"]]
                        .assign(
                            ev_per_charging_points = lambda df_: round(df_['value_stock']/df_['value_charging'],2),
                            )
                         .drop(["region_charging","value_stock","value_charging"],axis=1)
                         .reset_index(drop=True)
                         .set_index("region_stock")
                         .sort_values("ev_per_charging_points", ascending=False)
                         .transpose()
                         .reset_index(drop=True)
                          .assign(year = year)
                    )
        dfs.append(final_data)
    return pd.concat(dfs).set_index("year").sort_index(ascending=True)

if __name__ == '__main__':
    df = _clean_and_filter_charging_data()
    data_fast = get_charger_type_count_by_country(df,['China','India','USA','Europe'],'fast')

    # print(data_fast.head())

    import plotly.express as px
    fig = px.bar(data_fast, x='year', y='value',color="region_class",
                title="Fast publicly available chargers, 2014-2021",
                labels=dict(year="", value="Fast Charger Public Stock (Thousands)", region_class="Countries"))
    fig.show()