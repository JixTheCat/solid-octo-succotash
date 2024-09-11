"""This file contains basic reading and transformation functions
for teh SWA dataset.

"""
from carbon_converter import *
from conn import SWA
from os import path

import logging
import numpy as np
import pandas as pd

# This function is derived from the data prep done in paper 1
def prep():
    """This is the data preparation for exploratory analysis.

    It requires the SWA data libraries used in this project to function.
    """
    # You will require the df.feather output file from the SWA-Data pipeline.
    df = pd.read_feather("df.feather")

    # Lets get rid of some errors!
    # $1 sale price is ludicruous - There are other ludicruous prices like $50 etc as well. You can go harder at removing these.
    df.loc[1848, "average_per_tonne"] = np.nan
    df.loc[5493, "average_per_tonne"] = np.nan
    df = df.drop(df[df["giregion"]=="0"].index)
    # We fill in the average prices per region to flesh out the data:

    df["average_per_tonne"] = df["average_per_tonne"].replace({0: np.nan})
    avg_prices = df[
        df["average_per_tonne"].isnull()].apply(
        lambda x: mean_sale_price(x["giregion"],
                                  x["data_year_id"]),
        axis=1)
    df["average_per_tonne"] = df["average_per_tonne"].replace({0: np.nan})
    df["average_per_tonne"] = df["average_per_tonne"].combine_first(avg_prices)

    # Less than 1ha we can write off.
    # These were not such a problem?
    #df = df.drop(index=df[df['area_harvested']<1].index)

    ###################################
    # We insert values for emissions
    #
    # We convert the use of fuels into CO2E
    # This includes both scope1 and scope 2 in the totals, as well as separate columns for comparison.
    df["total_irrigation_electricity"] = df["total_irrigation_electricity"].apply(scope2)
    df["irrigation_energy_diesel"] = df["irrigation_energy_diesel"].div(1000).apply(diesel_irrigation)
    df["irrigation_energy_electricity"] = df["irrigation_energy_electricity"].div(1000).apply(scope2)
    df["irrigation_energy_pressure"] = df["irrigation_energy_pressure"].div(1000).apply(scope2)
    df["diesel_vineyard"] = df["diesel_vineyard"].div(1000).apply(diesel_vehicle)
    df["electricity_vineyard"] = df["electricity_vineyard"].div(1000).apply(scope2)
    df["petrol_vineyard"] = df["petrol_vineyard"].div(1000).apply(petrol_vehicle)
    df["lpg_vineyard"] = df["lpg_vineyard"].div(1000).apply(lpg_vehicle)
    df["biodiesel_vineyard"] = df["biodiesel_vineyard"].div(1000).apply(biodiesel_vehicle)

    df["scope2"] = df["total_irrigation_electricity"]\
        + df["irrigation_energy_electricity"]\
        + df["irrigation_energy_pressure"]\
        + df["electricity_vineyard"]

    df["scope1"] = df["irrigation_energy_diesel"]\
        + df["diesel_vineyard"]\
        + df["petrol_vineyard"]\
        + df["lpg_vineyard"]\
        + df["biodiesel_vineyard"]

    df["total_emissions"] = df["scope1"] + df["scope2"]

    ################################
    # We add total fertiliser used
    #
    df['nitrogen_applied'] = df['synthetic_nitrogen_applied']+df['organic_nitrogen_applied']
    df['fertiliser_applied'] = df['synthetic_fertiliser_applied']+df['organic_fertiliser_applied']

    df["fertiliser"] = df['fertiliser_applied'] + df['nitrogen_applied']

    ######################
    # We add in climates
    #
    df["rain"] = df["giregion"].apply(rain)
    df["temp"] = df["giregion"].apply(temp)
    df.loc[df["rain"]=="Unknown Climate", "rain"] = np.nan
    df.loc[df["temp"]=="Unknown Climate", "temp"] = np.nan
    df["climate"] = df["temp"] + " " + df["rain"]
    df.loc[df["climate"]=="Unknown Climate Unknown Climate", "climate"] = np.nan

    ########################
    # We change units
    #
    # changing ha to m2 for area to remove 1s and fractions for the log transform
    df["area_harvested"] = df["area_harvested"]*1000

    #create a flag for disease/fire/frost
    df["not_harvested"] = 0
    df.loc[df["area_not_harvested"]>0, "not_harvested"] = 1
    # dropiing not harvested area due to
    #df = df.drop(df[df["not_harvested"]==1].index)


    # red grapes
    df["red_grapes"] = 0
    df.loc[df["vineyard_area_red_grapes"]>0, "red_grapes"] = 1

    # include value
    df["value"] = df["area_harvested"]*df["average_per_tonne"]
    # include ratios
    #df["tha"] = df["tonnes_grapes_harvested"].div(df["area_harvested"])
    #df["vtha"] = df["tha"]*df["average_per_tonne"]

    # The variables are logarithmically transformed and then centrered.
    df_floats = df.select_dtypes(float)
    df_o = df.select_dtypes("O")
    df_int = df.select_dtypes(int)

    # We remove negative values
    for col in df_floats.columns:
        #We make an exception for gross margin as people can lose money
        if col == "gross_margin": 
            continue
        df_floats.loc[df_floats[col]<1, col] = np.nan 

    # we create variables scaled by area
    for col in df_floats.columns:
        # We don't want to divide area by itself.
        if col == "area_harvested": 
            continue
        df_floats["ha_" + col] = df_floats[col].div(df_floats["area_harvested"]).copy()

    # We spit out a copy that is not transformed for making maps and other summary
    # stats we want that make sense.
    pd.concat([df_floats, df_o, df_int], axis=1).to_csv("no_trans.csv")

    df_floats = df_floats.apply(np.log)

    # we create variables scaled by area
    for col in df_floats.columns:
        # We don't want to divide area by itself.
        if col == "area_harvested": 
            continue
        df_floats["ha_" + col] = df_floats[col].div(df_floats["area_harvested"]).copy()


    # Be sure to remove any values that become infinite, or are Null due to the transform or are missing/invalid values.
    df_floats = df_floats.replace({np.inf: np.nan, -np.inf: np.nan})

    df_floats = df_floats.loc[df_floats[df_floats["tonnes_grapes_harvested"].notnull()].index].copy()
    df_floats = df_floats.drop(df_floats[df_floats["water_used"].isnull()].index)
    df_floats = df_floats.drop(df_floats[df_floats["area_harvested"].isnull()].index)
    df_floats = df_floats.drop(df_floats[df_floats["total_emissions"].isnull()].index)

    # Scale and centre
    df_floats = df_floats - df_floats.mean()
    df_floats = df_floats/df_floats.std()

    return pd.concat([df_floats, df_o, df_int], axis=1)


def centre_scale(df: pd.DataFrame()):
    df_floats = df.select_dtypes(float)
    df_floats = df_floats - df_floats.mean()
    df_floats = df_floats/df_floats.std()
    df_o = df.select_dtypes("O")
    return pd.concat([df_floats, df_o], axis=1)


def fill_missing_value(df: pd.DataFrame, subset: str):
    """Fill missing values using median of a given subset in a df.

    Args:
        df: A given dataframe with missing values.
        subset: A column that can be used to subset rows for
        calculating the median.

    Returns:
        df: a dataframe with missing values filled using medians of
         associated rows subsets.
    """
    medians = df.groupby(by=subset).median()
    for col in df.columns:
        if col==subset:
            continue
        for item in df[subset].unique():
            df.loc[(df[col].isna()) & (df[subset] == item), col] = medians.loc[item][col]

def log_area_ratio(df: pd.DataFrame, feature: str):
    """Return a pandas series transformed as the log area ratio for a
        given dataframe variable

    Args:
        df: A dataframe as output from a SWA object. That contains an
            "area_harvested" column.
        feature: the name of the column within the given dataframe to
            transform.

    Returns:
        A log transformed pandas series of the given column divided by
            the log of the vineyards area.
    """
    return df[feature].apply(np.log)\
        .div(df["area_harvested"].apply(np.log))


def read_in(file: str, sheet="Vineyard",
            index="Membership Number") -> pd.DataFrame:
    """Read in a file from a .xlxs

    The read in Excel sheet is cleaned of NaN indexed rows and sets
    misinterpreted int types back to int.

    Args:
        file: The name of the file to be read in from the data
            subfolder.
        sheet: The sheet in the Excel file to read from.
        index: The name of the id column to use as an index

    Returns:
         Pandas dataframe with a cleaned index.
    """
    try:
        df = pd.read_excel(file, sheet_name=sheet)
    except PermissionError:
        raise PermissionError("You need to close the associated Excel"
                              " spreadsheet before continuing.\n\
        Please Try again...")
    # We need to drop the nans that are picked up erroneously from the
    # Excel read.
    df = df[df[index].notna()]
    if df[index].dtype == object:
        df[index] = df[index].str.extract(r'(\d+)')
        df = df[df[index].notna()]
        df[index] = df[index].astype(int)
    elif df[index].dtype == float:
        df[index] = df[index].astype(int)
    return df.set_index(index)


def remove_radicals(df: pd.DataFrame):
    """Remove radicals from df.feather as an input pandas dataframe.

    Args:
        df: a pandas dataframe as input in a format of the transformed
            database read out from the swa object.

    Returns:
            Pandas dataframe without radical values or bottom and top
            1% of values.
    """
    # Negative vineyard area harvested or farms that are too small
    for index in df[df['area_harvested'] < 0.1].index:
        df = df.drop(index=index)
    # Remove radical outliers - caused mostly by incorrect units.
    for index in df[df['water_used'] > 1000000].index:
        df = df.drop(index=index)

    ratio = df["tonnes_grapes_harvested"].div(
        df["area_harvested"]).apply(np.log)

    ratio = ratio.replace({np.inf: np.nan})
    ratio = ratio.replace({-np.inf: np.nan})
    ratio = ratio.replace({0: np.nan})
    ratio = ratio.replace({'0': np.nan})
    ratio.dropna(how='any', inplace=True)

    # We remove the bottom .1% and top .1% of values. This was
    # established as a sweet spot of the minimum amount of values
    # required to create a GLM with normal residuals.
    ids = ratio[~ratio.isnull()].sort_values().iloc[
        round(len(ratio) * .001):round(len(ratio) - len(ratio) * .001)
    ].index

    return df.loc[ids]


def temp(region: str):
    if region in ['Macedon Ranges',
                  'Mornington Peninsula',
                  'Orange', 'Canberra District', 'Yarra Valley',
                  'Beechworth',  'Upper Goulburn',
                  'Strathbogie Ranges',
                  'Southern Fleurieu', 'Adelaide Hills']:
        return 'Cool'
    if region in ['Mount Gambier', 'Henty', 'Grampians',
                  'Kangaroo Island', 'Sunbury', 'Wrattonbully',
                  'Coonawarra', 'Robe', 'Mount Benson']:
        return 'Cool'
    if region in ['Geelong', 'Pyrenees']:
        return 'Cool'
    if region in ['Alpine Valleys', 'Pemberton', 'Tumbarumba']:
        return 'Mild'
    if region in ['Blackwood Valley',
                  'Eden Valley',
                  'Manjimup',
                  'Granite Belt',
                  'Currency Creek',
                  'Padthaway',
                  'Heathcote']:
        return 'Mild'
    if region in ['Great Southern', 'McLaren Vale', 'Bendigo']:
        return 'Mild'
    if region in ['Geographe', 'New England Australia',
                  'Shoalhaven Coast', 'Southern Highlands',
                  'Margaret River']:
        return 'Warm'
    if region in ['Peel', 'Gundagai', 'Glenrowan',
                  'Pericoota','Clare Valley', 'Mudgee']:
        return 'Warm'
    if region in ['Rutherglen', 'Goulburn Valley', 'Langhorne Creek',
                  'Barossa Valley']:
        return 'Warm'
    if region in ['Hunter Valley', 'Hastings River']:
        return 'Hot'
    if region in ['Perth Hills', 'Swan District' 'Southern Flinders Ranges']:
        return 'Hot'
    if region in ['South Burnett',
                  'Cowra',
                  'Riverina',
                  'Adelaide Plains',
                  'Riverland',
                  'Swan Hill',
                  'Murray Darling']:
        return 'Hot'
    return 'Unknown Climate'


def rain(region: str):
    if region in ['Macedon Ranges',
                  'Mornington Peninsula',
                  'Orange', 'Canberra District', 'Yarra Valley',
                  'Beechworth',  'Upper Goulburn',
                  'Strathbogie Ranges',
                  'Southern Fleurieu', 'Adelaide Hills']:
        return 'Damp'
    if region in ['Mount Gambier', 'Henty', 'Grampians',
                  'Kangaroo Island', 'Sunbury', 'Wrattonbully',
                  'Coonawarra', 'Robe', 'Mount Benson']:
        return 'Dry'
    if region in ['Geelong', 'Pyrenees']:
        return 'Very Dry'
    if region in ['Alpine Valleys', 'Pemberton', 'Tumbarumba']:
        return 'Damp'
    if region in ['Blackwood Valley',
                  'Eden Valley',
                  'Manjimup',
                  'Granite Belt',
                  'Currency Creek',
                  'Padthaway',
                  'Heathcote']:
        return 'Dry'
    if region in ['Great Southern', 'McLaren Vale', 'Bendigo']:
        return 'Very Dry'
    if region in ['Geographe', 'New England Australia',
                  'Shoalhaven Coast', 'Southern Highlands',
                  'Margaret River']:
        return 'Damp'
    if region in ['Peel', 'Gundagai', 'Glenrowan',
                  'Pericoota','Clare Valley', 'Mudgee']:
        return 'Dry'
    if region in ['Rutherglen', 'Goulburn Valley', 'Langhorne Creek',
                  'Barossa Valley']:
        return 'Very Dry'
    if region in ['Hunter Valley', 'Hastings River']:
        return 'Damp'
    if region in ['Perth Hills', 'Swan District' 'Southern Flinders Ranges']:
        return 'Dry'
    if region in ['South Burnett',
                  'Cowra',
                  'Riverina',
                  'Adelaide Plains',
                  'Riverland',
                  'Swan Hill',
                  'Murray Darling']:
        return 'Very Dry'
    return 'Unknown Climate'


def mean_sale_price(giregion: str, year: str):
    """Returns the average price of sale per tonnes for a region."""
    df = pd.read_csv("wine_prices.csv", index_col=0)
    if giregion in df.index:
        if year in df.columns:
            return df.loc[giregion, year]
    return None


class DataDictionary:
    def __init__(self, pwd: str):
        """Initializes a data dictionary object."""
        # the data dictioanry is read in - this contains information
        # about different columns and their attributes.
        self.dd = read_in("Dictionary.xlsx", "Data Dictionary", "ID")

        # A connection the SWA database is initialised. It checks
        # if data is already downloaded and attempts to construct
        # a dataset that will be used for cleaning and transformations
        self.swa = SWA(pwd, populate=True)

        if path.exists(path.join('swa', 'answers.feather')):
            self.answers = pd.read_feather(
                path.join('swa', 'answers.feather')
            )
            logging.info('\tloaded answers')
        else:
            self.swa.create()
            self.answers = pd.read_feather(
                path.join('swa', 'answers.feather'))

    def clean(self, year: int, df: pd.DataFrame) -> pd.DataFrame:
        """Remove identified errors and deprecated columns. Rename
        columns to standardised names across for all  years.

        Args:
            year: The corresponding financial year to the dataframe in
             the format XXYY. for example 1998-1999 financial
                year would be 9899

            df: The input SWA data as a Pandas Dataframe.

        Returns:
            A data frame with changed columns that are standardised
            between each of the financial years.
        """
        # TODO the year argument can be replaced by automation
        #  - potentially
        for column in df.columns:
            col_name = self.dd[self.dd[year] == column][
                "Column names"].values
            if len(col_name):
                if \
                        self.dd[self.dd[year] == column][
                            "Deprecated"].values[
                            0]:
                    df = df.drop(columns=[column])
                else:
                    df = df.rename(columns={
                        column: col_name[0]})
            else:
                # We drop undefined columns
                logging.warning(
                    "\"{}\" column is not accounted in the data "
                    "dictionary.\n".format(
                        column))
                df = df.drop(columns=[column])
        return df

    def remove_admin(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove admin and  from an input dataframe

        Args:
            df: a dataframe from the SWA dataset defined in the
                created data dictionary object.

        Returns:
            The input dataframe [df] with admin columns removed.
        """
        for column in df.columns:
            try:
                if self.dd[self.dd["Column names"] == column]\
                        ["Admin"].values[0]:
                    df = df.drop(columns=[column])
            except IndexError:
                logging.warning(
                    "Column {} Could not be found!!!".format(column))
        return df

    def id_to_fertiliser(self, id: int):
        """Convert an id into a fertiliser name

        id: A fertiliser id.
        """
        if id == '0' or id == 0:
            return 0
        try:
            return self.swa.fertilisers \
                [self.swa.fertilisers['id'] == int(id)] \
                ['label'].values[0]
        except ValueError:
            return id

    def clean_answers(self, use_median=False):
        """Takes self.answers dataframe from conn and cleans it.

        """
        # administrative data is listed as None in the year column
        # To preserve it we need to set it to a value
        self.answers['data_year_id'] = np.where(
            self.answers['data_year_id'].isnull(),
            'admin',
            self.answers['data_year_id'])

        # We remove unsued columns:
        self.answers = self.answers.drop(['created_at'], axis=1)

        # we collect all the columns defined in the dictionary.
        self.answers = self.answers[
            self.answers['field_id'].isin(self.dd.index)]

        # We remove deleted rows and then the corresponding column
        self.answers = self.answers[
            self.answers['deleted_at'].isnull()]
        self.answers = self.answers.drop(['deleted_at'], axis=1)

        # Admin values such as region do not have an entered
        # year id, so we change these to "admin" otherwise they
        # are automatically removed.

        self.answers.loc[
            self.answers[
                "data_year_id"].isnull(),
            "data_year_id"] = "admin"

        """
        gi_regions = self.answers\
            [self.answers["field"] == 'giregion']\
            .pivot(index="member_id", columns="field", values="value")
        """
        # We sort by the last entry so that it is the most up-to-date
        # data when select using aggfunc first
        self.answers = self.answers.sort_values('updated_at',
                                                ascending=False)
        self.answers = self.answers.pivot_table(
            index=['member_id', 'data_year_id'],
            columns='field', values='value',
            aggfunc='first')

        # TODO why does this functio neven exist? It is dangerous and
        #  I can only imagine to be useful in edge cases.
        if use_median:
            fill_missing_value(self.answers, "giregion")
        else:
            # This is potentially a dangerously slow way to implement
            # this
            self.answers = self.answers.replace({np.nan: 0})

        #######################################################
        # this is the most garbage way to catch these errors  #
        #######################################################
        #
        # Report these vaariables to the deve team next time.
        #

        # self.answers.loc[762].loc['2014/2015', 'vineyard_area'] = 28
        # # The original value is a string of '28 ha'
        # # it is the only occurrence of this though weird.
        # # The original value of 579's entries were marked as 'nill'
        self.answers.loc[579].loc['2017/2018',
                                  'vineyard_area_red_grapes'] = 0
        self.answers.loc[579].loc['2017/2018',
                                  'vineyard_area_white_grapes'] = 0
        # # Below had the value '80 cubic mtrs'
        self.answers.loc[
            self.answers['organic_applied_1'] == '80 cubic mtrs',
            'organic_applied_1'] = 80
        # almost identical to above but '80 cu mt.' instead
        self.answers.loc[
            self.answers['organic_applied_1'] == '80 cu mt.',
            'organic_applied_1'] = 80

        # We need to type the columns because they are not
        # strictly typed.
        self.answers['vineyard_area_red_grapes'] = self.answers[
            'vineyard_area_red_grapes'].astype(float)
        self.answers['vineyard_area_white_grapes'] = self.answers[
            'vineyard_area_white_grapes'].astype(float)
        self.answers['vineyard_area'] = self.answers[
            'vineyard_area'].astype(float)
        self.answers['tonnes_grapes_harvested'] = self.answers[
            'tonnes_grapes_harvested'].astype(float)

        self.answers['organic_fertiliser_applied'] = 0
        self.answers['synthetic_fertiliser_applied'] = 0

        for i in range(1, 6):
            self.answers['fertilizer_o' + str(i)] = \
                self.answers['fertilizer_o' + str(i)].apply(
                    self.id_to_fertiliser)
            self.answers['fertilizer_o' + str(i)] = \
                self.answers['fertilizer_o' + str(i)].astype(str)

            self.answers['fertilizer_s' + str(i)] = \
                self.answers['fertilizer_s' + str(i)].apply(
                    self.id_to_fertiliser)
            self.answers['fertilizer_s' + str(i)] = \
                self.answers['fertilizer_s' + str(i)].astype(str)

            self.answers['applied_o' + str(i)] = \
                self.answers['applied_o' + str(i)].astype(float)
            self.answers['organic_fertiliser_applied'] += \
                self.answers['applied_o' + str(i)]

            self.answers['applied_s' + str(i)] = \
                self.answers['applied_s' + str(i)].astype(float)
            self.answers['synthetic_fertiliser_applied'] += \
                self.answers['applied_s' + str(i)]

        # The administrative values need to be populated for each year
        # so that we can properly slice the data into wanted subsets.
        ids = self.answers.droplevel('data_year_id').index.unique()
        for id in ids:
            if len(self.answers.loc[id]) == 1:
                self.answers.drop(index=(id, 'admin'), inplace=True)
            elif int(self.answers.loc[id].loc['admin', 'giregion']) == 0:
                self.answers.loc[id].loc['admin', 'giregion'] = "Winery"
            else:
                try:
                    self.answers.loc[id].loc['admin', 'giregion'] = \
                        self.swa.gi_regions[
                            self.swa.gi_regions['id'] == \
                            int(self.answers.loc[id].loc["admin", 'giregion'])]\
                            ['label'].values[0]
                    for year in self.answers.loc[id].index:
                        self.answers.loc[id].loc[year, 'giregion'] = \
                            self.answers.loc[id].loc['admin', 'giregion']
                except IndexError:
                    print(self.answers.loc[id].loc["admin", 'giregion'])
                    print(id)
                    raise IndexError

        # We set vineyard_area to the sum of red and white grape areas
        # for areas that don't already have a total area.
        # we ignore vineyards that do not have an area but also sum
        # to zero for red and white areas.
        self.answers.loc[
            (0 != (self.answers['vineyard_area_red_grapes'] +
                   self.answers[
                       'vineyard_area_white_grapes'])) & \
            ((self.answers['vineyard_area'] !=
              (self.answers['vineyard_area_red_grapes'] +
               self.answers[
                   'vineyard_area_white_grapes']))),
            'vineyard_area'] = \
            self.answers['vineyard_area_red_grapes'] + self.answers[
                'vineyard_area_white_grapes']

        # we remove vineyards with no area
        # note that this removes the admin rows as well.
        self.answers.drop(
            index=self.answers[
                self.answers['vineyard_area'] == 0].index,
            inplace=True)

        # features need to be created such as area from red and white
        # grapes and the total fertiliser/water etc.

        self.answers['synthetic_nitrogen_applied'] = 0
        self.answers['organic_nitrogen_applied'] = 0

        for i in range(1, 4):
            self.answers['synthetic_applied_' + str(i)] = \
                self.answers['synthetic_applied_' + str(i)].astype(
                    float)
            self.answers['synthetic_nitrogen_applied'] += \
                self.answers['synthetic_applied_' + str(i)]

            self.answers['organic_applied_' + str(i)] = \
                self.answers['organic_applied_' + str(i)].astype(
                    float)
            self.answers['organic_nitrogen_applied'] += \
                self.answers['organic_applied_' + str(i)]

        # We need to properly type all columns or converting between
        # formats fails.
        self.answers['river_water'] = \
            self.answers['river_water'].astype(float)
        self.answers['groundwater'] = \
            self.answers['groundwater'].astype(float)
        self.answers['surface_water_dam'] = \
            self.answers['surface_water_dam'].astype(float)
        self.answers['recycled_water_from_other_source'] = \
            self.answers['recycled_water_from_other_source'].astype(
                float)
        self.answers['mains_water'] = \
            self.answers['mains_water'].astype(float)
        self.answers['other_water'] = \
            self.answers['other_water'].astype(float)
        self.answers['water_applied_for_frost_control'] = \
            self.answers['water_applied_for_frost_control'].astype(
                float)

        self.answers['irrigation_energy_diesel'] = \
            self.answers['irrigation_energy_diesel'].astype(float)
        self.answers['irrigation_energy_electricity'] = \
            self.answers['irrigation_energy_electricity'].astype(
                float)
        self.answers['irrigation_energy_pressure'] = \
            self.answers['irrigation_energy_pressure'].astype(float)
        self.answers['irrigation_energy_solar'] = \
            self.answers['irrigation_energy_solar'].astype(float)

        self.answers['irrigation_type_dripper'] = \
            self.answers['irrigation_type_dripper'].astype(float)
        self.answers['irrigation_type_flood'] = \
            self.answers['irrigation_type_flood'].astype(float)
        self.answers['irrigation_type_non_irrigated'] = \
            self.answers['irrigation_type_non_irrigated'].astype(
                float)
        self.answers['irrigation_type_overhead_sprinkler'] = \
            self.answers['irrigation_type_overhead_sprinkler'].astype(
                float)
        self.answers['irrigation_type_undervine_sprinkler'] = \
            self.answers[
                'irrigation_type_undervine_sprinkler'].astype(float)

        self.answers['diesel_vineyard'] = \
            self.answers['diesel_vineyard'].astype(float)
        self.answers['electricity_vineyard'] = \
            self.answers['electricity_vineyard'].astype(float)
        self.answers['petrol_vineyard'] = \
            self.answers['petrol_vineyard'].astype(float)
        self.answers['vineyard_solar'] = \
            self.answers['vineyard_solar'].astype(float)
        self.answers['vineyard_wind'] = \
            self.answers['vineyard_wind'].astype(float)
        self.answers['lpg_vineyard'] = \
            self.answers['lpg_vineyard'].astype(float)
        self.answers['lpg_vineyard'] = \
            self.answers['lpg_vineyard'].astype(float)
        self.answers['biodiesel_vineyard'] = \
            self.answers['biodiesel_vineyard'].astype(float)

        self.answers['slashing_number_of_times_passes_per_year'] = \
            self.answers['slashing_number_of_times_passes_' +
                         'per_year'].astype(float)
        self.answers[
            'fungicide_spraying_number_of_times_passes_' +
            'per_year'] = \
            self.answers['fungicide_spraying_number_of_times_' +
                         'passes_per_year'].astype(float)
        self.answers[
            'herbicide_spraying_number_of_times_passes_' +
            'per_year'] = \
            self.answers['herbicide_spraying_number_of_times_' +
                         'passes_per_year'].astype(float)
        self.answers[
            'insecticide_spraying_number_of_times_passes_' +
            'per_year'] = \
            self.answers['insecticide_spraying_number_of_' +
                         'times_passes_per_year'].astype(float)

        self.answers['nh_disease'] = \
            self.answers['nh_disease'].astype(float)
        self.answers['nh_frost'] = \
            self.answers['nh_frost'].astype(float)
        self.answers['nh_new_development'] = \
            self.answers['nh_new_development'].astype(float)
        self.answers['nh_non_sale'] = \
            self.answers['nh_non_sale'].astype(float)

        self.answers['bare_soil'] = \
            self.answers['bare_soil'].astype(float)
        self.answers['annual_cover_crop'] = \
            self.answers['annual_cover_crop'].astype(float)
        self.answers['permanent_cover_crop_native'] = \
            self.answers['permanent_cover_crop_native'].astype(float)
        self.answers['permanent_cover_crop_non_native'] = \
            self.answers['permanent_cover_crop_non_' +
                         'native'].astype(float)
        self.answers['permanent_cover_crop_volunteer_sward'] = \
            self.answers['permanent_cover_crop_volunteer' +
                         '_sward'].astype(float)

        self.answers['off_farm_income'] = self.answers[
            'off_farm_income'].astype(float)
        self.answers['highest_per_tonne'] = self.answers[
            'highest_per_tonne'].astype(float)
        self.answers['lowest_per_tonne'] = self.answers[
            'lowest_per_tonne'].astype(float)
        self.answers['average_per_tonne'] = self.answers[
            'average_per_tonne'] .astype(float)
        self.answers['total_grape_revenue'] = self.answers[
            'total_grape_revenue'].astype(float)
        self.answers['total_operating_costs'] = self.answers[
            'total_operating_costs'].astype(float)
        self.answers['cost_of_debt_servicing'] = self.answers[
            'cost_of_debt_servicing'].astype(float)
        self.answers['gross_margin'] = self.answers[
            'gross_margin'].astype(float)
        self.answers['operating_cost_per_ha'] = self.answers[
            'operating_cost_per_ha'].astype(float)
        self.answers['operating_cost_per_t'] = self.answers[
            'operating_cost_per_t'].astype(float)

        # Below are calculated columns - this is done before resetting
        # it back to using nans
        self.answers['water_used'] = \
            self.answers['river_water'] \
            + self.answers['groundwater'] \
            + self.answers['surface_water_dam'] \
            + self.answers['recycled_water_from_other_source'] \
            + self.answers['mains_water'] \
            + self.answers['other_water'] \
            + self.answers['water_applied_for_frost_control']

        self.answers['total_irrigation_electricity'] = \
            + self.answers['irrigation_energy_electricity'] \
            + self.answers['irrigation_energy_solar']
        # + self.answers['irrigation_energy_pressure']
        # ^ is not included as it is measured in hectares.

        self.answers['total_irrigation_fuel'] = \
            self.answers['irrigation_energy_diesel'] \
        # + self.answers['irrigation_energy_pressure']
        # ^ is not included as it is measured in hectares.

        self.answers['total_irrigation_area'] = \
            self.answers['irrigation_type_dripper'] \
            + self.answers['irrigation_type_flood'] \
            + self.answers['irrigation_type_non_irrigated'] \
            + self.answers['irrigation_type_overhead_sprinkler'] \
            + self.answers['irrigation_type_undervine_sprinkler']

        self.answers['total_vineyard_fuel'] = \
            self.answers['diesel_vineyard'] \
            + self.answers['petrol_vineyard'] \
            + self.answers['lpg_vineyard'] \
            + self.answers['biodiesel_vineyard']

        self.answers['total_vineyard_electricity'] = \
            + self.answers['electricity_vineyard'] \
            + self.answers['vineyard_solar'] \
            + self.answers['vineyard_wind'] \

        self.answers['total_tractor_passes'] = \
            self.answers['slashing_number_of_times_passes_per_year'] \
            + self.answers[
                'fungicide_spraying_number_of_times_passes_' +
                'per_year'] \
            + self.answers[
                'herbicide_spraying_number_of_times_passes_' +
                'per_year'] \
            + self.answers[
                'insecticide_spraying_number_of_times_passes_' +
                'per_year']

        self.answers['area_not_harvested'] = \
            self.answers['nh_disease'] \
            + self.answers['nh_frost'] \
            + self.answers['nh_new_development'] \
            + self.answers['nh_non_sale']

        self.answers['area_harvested'] = \
            self.answers['vineyard_area'] - \
            self.answers['area_not_harvested']

        self.answers['tonnes_per_hectare'] = \
            self.answers['tonnes_grapes_harvested'] / \
            self.answers['area_harvested']

        self.answers['water_used_per_hectare'] = \
            self.answers['water_used'] / \
            self.answers['area_harvested']

        self.answers['water_used_per_tonne_harvested'] = \
            self.answers['water_used'] / \
            self.answers['tonnes_grapes_harvested']

        self.answers = self.answers[
            [
                # derived columns
                'vineyard_area'
                , 'area_harvested'
                , 'tonnes_grapes_harvested'
                , 'water_used'
                , 'total_tractor_passes'
                , 'total_vineyard_fuel'
                , 'total_vineyard_electricity'
                , 'total_irrigation_area'
                , 'synthetic_nitrogen_applied'
                , 'organic_nitrogen_applied'
                , 'synthetic_fertiliser_applied'
                , 'organic_fertiliser_applied'
                , 'area_not_harvested'
                , 'total_irrigation_electricity'
                , 'total_irrigation_fuel'
                # properties
                , 'giregion'
                , 'fertilizer_s1'
                , 'fertilizer_s2'
                , 'fertilizer_s3'
                , 'fertilizer_s4'
                , 'fertilizer_s5'
                , 'fertilizer_o1'
                , 'fertilizer_o2'
                , 'fertilizer_o3'
                , 'fertilizer_o4'
                , 'fertilizer_o5'
                # area
                , 'vineyard_area_white_grapes'
                , 'vineyard_area_red_grapes'
                # water
                , 'river_water'
                , 'groundwater'
                , 'surface_water_dam'
                , 'recycled_water_from_other_source'
                , 'mains_water'
                , 'other_water'
                , 'water_applied_for_frost_control'
                # cover crops
                , 'bare_soil'
                , 'annual_cover_crop'
                , 'permanent_cover_crop_native'
                , 'permanent_cover_crop_non_native'
                , 'permanent_cover_crop_volunteer_sward'
                # irrigation energy
                , 'irrigation_energy_diesel'
                , 'irrigation_energy_electricity'
                , 'irrigation_energy_pressure'
                , 'irrigation_energy_solar'
                # irrigation area
                , 'irrigation_type_dripper'
                , 'irrigation_type_flood'
                , 'irrigation_type_non_irrigated'
                , 'irrigation_type_overhead_sprinkler'
                , 'irrigation_type_undervine_sprinkler'
                # vineyard energy
                , 'diesel_vineyard'
                , 'electricity_vineyard'
                , 'petrol_vineyard'
                , 'vineyard_solar'
                , 'vineyard_wind'
                , 'lpg_vineyard'
                , 'biodiesel_vineyard'
                # tractor passes
                , 'slashing_number_of_times_passes_per_year'
                , 'fungicide_spraying_number_of_times_passes_' +
                  'per_year'
                , 'herbicide_spraying_number_of_times_passes_' +
                  'per_year'
                , 'insecticide_spraying_number_of_times_passes_' +
                  'per_year'
                # area not harvested
                , 'nh_disease'
                , 'nh_frost'
                , 'nh_new_development'
                , 'nh_non_sale'
                # economics
                , 'off_farm_income'
                , 'highest_per_tonne'
                , 'lowest_per_tonne'
                , 'average_per_tonne'
                , 'total_grape_revenue'
                , 'total_operating_costs'
                , 'cost_of_debt_servicing'
                , 'gross_margin'
                , 'operating_cost_per_ha'
                , 'operating_cost_per_t'

            ]
        ]

        return self.answers.replace({0: np.nan})


def nan_to_str(array: np.array, replace_with: str) -> np.array:
    """Return a str array with replaced NaN's.

    Args:
        array: A Numpy array of dtype string.
        replace_with: a str to replace NaN values with

    Returns:
        A Numpy array of dtype string with no NaNs.
    """
    array = array.astype(str)
    array = np.where(
        array == "nan",
        replace_with,
        array)
    return array


def bin_col(array: np.array, str_true="yes") -> pd.Categorical:
    """Return a binary array for false and true.

    The default is to convert yes/no to 1/0 and to convert an array
    of dtype float to 1/0 for values above zero and equal to zero,
    respectively.

    Args:
        array: A numpy array
        str_true: The string value that will be converted to 1, indicating a TRUE value.

    Returns:
        A numpy array of dtype categorical.
    """
    if array.dtype == float:
        return pd.Categorical(np.where(array > 0, 1, 0).tolist())
    else:
        return pd.Categorical(
            np.where(array.str.lower() == str_true.lower(), 1,
                     0).tolist())


# TODO This function needs a better name...
def viticulture_clean(df: pd.DataFrame) -> pd.DataFrame:
    """This function consists of fixes that are specific to viticulture variables.

    This function fills out several variables from other such as

    Args:
        df: A pandas dataframe to be transformed.

    Returns:
        A data frame of Viticulture variables.
    """
    ##############
    # Dropped Rows
    df = df[~np.isnan(df["tonnes grapes harvested"])]

    ##################
    # Dropped Columns:
    df = df.drop(columns=["tonnes crushed if winery",
                          "vineyard area metric"])

    ###########
    # Fill NaNs
    df["certified organic"] = nan_to_str(
        df["certified organic"], "NO")
    df["vineyard biodynamic"] = nan_to_str(
        df["vineyard biodynamic"], "NO")
    df["not harvested yes no"] = nan_to_str(
        df["not harvested yes no"], "No")

    for column in df.columns:
        # floats are broadly dealt with by setting all NaNs to 0
        # This is potentially problematic and may need to be addressed
        # but the risk is unknown.
        if df[column].dtype == float:
            df[column] = np.where(np.isnan(df[column]), 0, df[column])

    #####################
    # Calculated Columns:
    df["total vineyard area"] = df["vineyard area white grapes"] + df[
        "vineyard area red grapes"]

    df["total water used"] = df["river water"] + df["groundwater"] + \
                             df["surface water dam"] + \
                             df["recycled water from winery"] + df[
                                 "recycled water from other source"] + \
                             df["mains water"] + df["other water"] + \
                             df["water applied for frost control"]

    df["total crop cover"] = df["annual cover crop"] + df[
        "permanent cover crop non native"] + \
                             df[
                                 "permanent cover crop volunteer sward"] + \
                             df["permanent cover crop native"]

    df["total irrigation energy"] = df["irrigation energy pressure"] + \
                                    df["irrigation energy diesel"] + \
                                    df[
                                        "irrigation energy electricity"] + \
                                    df["irrigation energy solar"]

    df["total irrigation area"] = df["irrigation type dripper"] + df[
        "irrigation type undervine sprinkler"] + \
                                  df[
                                      "irrigation type overhead sprinkler"] + \
                                  df["irrigation type flood"] + df[
                                      "irrigation type non irrigated"]

    df["total fertilizer"] = df["fertilizer area mulch"] + df[
        "fertilizer area compost"] + \
                             df["fertilizer area manure"] + df[
                                 "applied s2"] + df["applied s1"] + \
                             df["applied s3"] + df["applied s4"] + \
                             df["applied s5"] + df["applied o1"] + df[
                                 "applied o2"] + df["applied o4"] + \
                             df["applied o3"] + \
                             df["applied o5"]

    df["total nitrogen"] = df["nitrogen"] + df["organic nitrogen"] + \
                           df["synthetic applied 1"] + \
                           df["synthetic applied 2"] + df[
                               "synthetic applied 3"] + df[
                               "organic applied 1"] + df[
                               "organic applied 2"] + \
                           df["organic applied 3"]

    ################
    # Typed Columns:
    df["certified organic"] = bin_col(df["certified organic"])
    df["not harvested yes no"] = bin_col(df["not harvested yes no"])
    df["vineyard biodynamic"] = bin_col(df["vineyard biodynamic"])
    df["vineyard community biodiversity"] = bin_col(
        df["vineyard community biodiversity"])
    df["off farm income"] = bin_col(df["off farm income"])
    #
    #   Below pertain to contractors performing duties
    df["giregion"] = pd.Categorical(df["giregion"])
    df["gizone"] = pd.Categorical(df["gizone"])
    df["harvesting"] = pd.Categorical(df["harvesting"])
    df["pruning"] = pd.Categorical(df["pruning"])
    df["slashing"] = pd.Categorical(df["slashing"])
    df["fungacide spraying"] = pd.Categorical(
        df["fungacide spraying"])
    df["insecticide spraying"] = pd.Categorical(
        df["insecticide spraying"])
    df["herbicide spraying"] = pd.Categorical(
        df["herbicide spraying"])

    ###########################################
    # Repurposed columns, float to categorical:
    #
    #   Grape type - summed to ["total vineyard"]
    df["vineyard area white grapes"] = bin_col(
        df["vineyard area white grapes"])
    df = df.rename(
        columns={"vineyard area white grapes": "white grapes"})
    df["vineyard area red grapes"] = bin_col(
        df["vineyard area red grapes"])
    df = df.rename(
        columns={"vineyard area red grapes": "red grapes"})
    #
    #   Cover crops - summed to ["total crop cover"]
    df["annual cover crop"] = bin_col(df["annual cover crop"])
    df["permanent cover crop non native"] = bin_col(
        df["permanent cover crop non native"])
    df["permanent cover crop volunteer sward"] = bin_col(
        df["permanent cover crop volunteer sward"])
    df["permanent cover crop native"] = bin_col(
        df["permanent cover crop native"])
    # not included in sum as it is the absence
    df["bare soil"] = bin_col(df["bare soil"])
    #
    #   Water - summed to ["total water used"]
    df["river water"] = bin_col(df["river water"])
    df["groundwater"] = bin_col(df["groundwater"])
    df["surface water dam"] = bin_col(df["surface water dam"])
    df["recycled water from winery"] = bin_col(
        df["recycled water from winery"])
    df["recycled water from other source"] = bin_col(
        df["recycled water from other source"])
    df["mains water"] = bin_col(df["mains water"])
    df["other water"] = bin_col(df["other water"])
    df["water applied for frost control"] = bin_col(
        df["water applied for frost control"])
    #
    #   Irrigation energy - summed to ["total irrigation energy"]
    df["irrigation energy pressure"] = bin_col(
        df["irrigation energy pressure"])
    df["irrigation energy diesel"] = bin_col(
        df["irrigation energy diesel"])
    df["irrigation energy electricity"] = bin_col(
        df["irrigation energy electricity"])
    df["irrigation energy solar"] = bin_col(
        df["irrigation energy solar"])
    #
    #   Irrigation type - summed to ["total irrigation area"]
    df["irrigation type dripper"] = bin_col(
        df["irrigation type dripper"])
    df["irrigation type undervine sprinkler"] = bin_col(
        df["irrigation type undervine sprinkler"])
    df["irrigation type overhead sprinkler"] = bin_col(
        df["irrigation type overhead sprinkler"])
    df["irrigation type flood"] = bin_col(df["irrigation type flood"])
    df["irrigation type non irrigated"] = bin_col(
        df["irrigation type non irrigated"])
    #
    #   Fertiliser used - summed to ["total fertilizer"]
    df["fertilizer area mulch"] = bin_col(df["fertilizer area mulch"])
    df = df.rename(
        columns={"fertilizer area mulch": "mulch"})
    df["fertilizer area compost"] = bin_col(
        df["fertilizer area compost"])
    df = df.rename(
        columns={"fertilizer area compost": "compost"})
    df["fertilizer area manure"] = bin_col(
        df["fertilizer area manure"])
    df = df.rename(
        columns={"fertilizer area manure": "manure"})

    ################################################
    # Combination Categorical (i.e Other categories)
    df["fertilizer other"] = df["applied s2"] + df["applied s1"] + df[
        "applied s3"] + df["applied s4"] + \
                             df["applied s5"] + df["applied o1"] + df[
                                 "applied o2"] + df["applied o4"] + \
                             df["applied o3"] + \
                             df["applied o5"]
    df["fertilizer other"] = bin_col(df["fertilizer other"])

    #   Nitrogen - summed to ["total nitrogen"]
    df["nitrogen synthetic"] = df["nitrogen"] + df[
        "synthetic applied 1"] + df["synthetic applied 2"] + \
                               df["synthetic applied 3"]
    df["nitrogen synthetic"] = bin_col(df["nitrogen synthetic"])

    df["nitrogen organic"] = df["organic nitrogen"] + df[
        "organic applied 1"] + df["organic applied 2"] + \
                             df["organic applied 3"]
    df["nitrogen organic"] = bin_col(df["nitrogen organic"])

    # The below are contained within electricity sourced from the grid and are only descriptors of what type of
    # renewable energy is being sourced from the grid. Many entries are conflicting or non-dscript such as recording
    # renewable energy being sourced then not specifying which one. Or specifying the same amount for each, with
    # these values being identical to the total sourced. To overcome this they are combined into a renewable energy
    # categorical variable.
    df["renewable grid source"] = df[
                                      "vineyard renewable energy sourced from the grid"] + \
                                  df["vineyard solar"] + \
                                  df["vineyard wind"] + df[
                                      "vineyard other renewable"]
    df["renewable grid source"] = bin_col(df["renewable grid source"])

    df = df.drop(columns=[
        "vineyard generated renewable electricity exported to the grid",
        "biodiesel vineyard",
        "cost of debt servicing",
        "gross margin",
        "operating cost per ha",
        "operating cost per t",
        # The below are the names of alternative or decidedly listed products. We use the amount column associated
        # with the names below to populate the 'Other' column and throw these away:
        "fertilizer s1",
        "fertilizer s2",
        "fertilizer s3",
        "fertilizer s4",
        "fertilizer s5",
        "fertilizer o1",
        "fertilizer o2",
        "fertilizer o3",
        "fertilizer o4",
        "fertilizer o5",
        "synthetic nitrogen percent 1",
        "synthetic nitrogen percent 2",
        "synthetic nitrogen percent 3",
        "organic nitrogen percent 1",
        "organic nitrogen percent 2",
        "organic nitrogen percent 3",
        "urea percent 1",
        "urea percent 2",
        "urea percent 3",
        "applied s2",
        "applied s1",
        "applied s3",
        "applied s4",
        "applied s5",
        "applied o1",
        "applied o2",
        "applied o4",
        "applied o3",
        "applied o5",
        "nitrogen",
        "synthetic applied 1",
        "synthetic applied 2",
        "synthetic applied 3",
        "organic nitrogen",
        "organic applied 1",
        "organic applied 2",
        "organic applied 3",
        "vineyard renewable energy sourced from the grid",
        "vineyard solar",
        "vineyard wind",
        "vineyard other renewable",
        # The below are linear transforms of other columns:
        "average per tonne",
    ])

    return df


def nan_row_summary(df: pd.DataFrame):
    """Return a summary of the number of rows that are empty in a given dataframe.

    Args:
        df: A pandas dataframe to be summarised.
    """
    df_len = len(df.columns)
    df = df.transpose()

    df = df.isna().sum()
    df = pd.DataFrame(df, )
    print("Below are the proportions of NaN values in a row:")
    df[0] = df[0] / df_len
    print(df[0].describe())


def nan_col_summary(df: pd.DataFrame):
    """Return a summary of the number of columns  that are empty in a given dataframe.

    Args:
        df: A pandas dataframe to be summarised."""
    df_len = len(df)

    df = df.isna().sum()
    df = pd.DataFrame(df, )
    df = df / df_len
    df.columns = ['Proportion of NaNs']
    print("Below is a summary of the proportion of NaNs by column:\n")
    print(df.describe())
    df = df.transpose()
    print("\nBelow are the proportion of NaNs in each column:\n")
    for col in df:
        print("{}: {}%".format(col, round(df[col][0] * 100, 2)))


def pivot_answers(answers: pd.DataFrame):
    """
    This pivots the self.answers df from a saved feather state.

    Args:
        self.answers: a pandas dataframe as generated by a SWA data object.

    Returns:
        a pivoted dataframe for use in analysis.
    """
    piv = pd.pivot_table(
        answers.sort_values('updated_at'), values='value',
        index=['member_id', 'data_year_id'], columns='field_id',
        aggfunc='first')

    piv = piv.convert_dtypes()

    return piv.apply(pd.to_numeric, errors='ignore')

def fuel_cost(fuel: float, year: str, type: str):
    """


    """
