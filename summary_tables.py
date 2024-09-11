"""This is a quick script to make some summary supplementary tables."""

from os import listdir
from os.path import isfile, join
import pandas as pd

import re

df = pd.read_csv("dfb.csv")

df.loc[df["herbicide_spraying_number_of_times_passes_per_year"]<1, "herbicide_spraying_number_of_times_passes_per_year"] = 0
df.loc[df["insecticide_spraying_number_of_times_passes_per_year"]<1, "insecticide_spraying_number_of_times_passes_per_year"] = 0
df.loc[df["herbicide_spraying_number_of_times_passes_per_year"]<1, "herbicide_spraying_number_of_times_passes_per_year"] = 0
for col in df.loc[:, df.dtypes == np.float64].columns:
    df.loc[df[col]<0, col] = np.nan
df[df.loc[:, df.dtypes == np.float64].columns].min()

cols = ["tonnes_grapes_harvested"
    , "area_harvested"
    , "water_used"
    # , "total_tractor_passes"
    , "total_fertiliser"
    # , "synthetic_nitrogen_applied"
    # , "organic_nitrogen_applied"
    # , "synthetic_fertiliser_applied"
    # , "organic_fertiliser_applied"
    , "giregion"
    , "data_year_id"
    , "river_water"
    , "groundwater"
    , "surface_water_dam"
    , "recycled_water_from_other_source"
    , "mains_water"
    , "other_water"
    , "water_applied_for_frost_control"
    , "bare_soil"
    , "annual_cover_crop"
    , "permanent_cover_crop_native"
    , "permanent_cover_crop_non_native"
    , "permanent_cover_crop_volunteer_sward"
    , "diesel_vineyard"
    , "electricity_vineyard"
    , "petrol_vineyard"
    , "vineyard_solar"
    # , "vineyard_wind"
    , "lpg_vineyard"
    , "biodiesel_vineyard"
    , "slashing_number_of_times_passes_per_year"
    , "fungicide_spraying_number_of_times_passes_per_year"
    , "herbicide_spraying_number_of_times_passes_per_year"
    , "insecticide_spraying_number_of_times_passes_per_year"
    , "irrigation_energy_diesel"
    , "irrigation_energy_electricity"
    , "irrigation_energy_pressure"
    , "irrigation_energy_solar"
    , "irrigation_type_dripper"
    , "irrigation_type_flood"
    , "irrigation_type_non_irrigated"
    , "irrigation_type_overhead_sprinkler"
    , "irrigation_type_undervine_sprinkler"
    , "nh_disease"
    # , "nh_frost"
]

# These are the columns that will be classes!

cat_cols = [ # These are binary
    "bare_soil"
    , "annual_cover_crop"
    , "permanent_cover_crop_native"
    , "permanent_cover_crop_non_native"
    , "permanent_cover_crop_volunteer_sward"
    , "irrigation_energy_diesel"
    , "irrigation_energy_electricity"
    , "irrigation_energy_pressure"
    , "irrigation_energy_solar"
    , "irrigation_type_dripper"
    , "irrigation_type_flood"
    , "irrigation_type_non_irrigated"
    , "irrigation_type_overhead_sprinkler"
    , "irrigation_type_undervine_sprinkler"
    , 'river_water'
    , 'groundwater'
    , 'surface_water_dam'
    , 'recycled_water_from_other_source'
    , 'mains_water'
    , 'other_water'
    , 'water_applied_for_frost_control'
    , "nh_disease"
    , "data_year_id" # These are one hot encoded
    , "giregion"
]

# We need to declare each column that is categorical
# as a categorical column!
#
# later these need to be one hot encoded.
for column in cat_cols:
    df[column] = pd.Categorical(df[column])

# We are going to change some binary columns into
# multiclass columns!

irrigation_type = [
    "irrigation_type_dripper"
    , "irrigation_type_flood"
    , "irrigation_type_non_irrigated"
    , "irrigation_type_overhead_sprinkler"
    , "irrigation_type_undervine_sprinkler"
]

for col in irrigation_type:
    df[col] = df[col].cat.rename_categories({0: "", 1: "{} ".format(col[16:])}).copy()
df["irrigation_type"] = pd.Categorical(df[irrigation_type].astype(str).sum(axis=1))
cols = list(set(cols) - set(irrigation_type))
cols.append("irrigation_type")

####################

irrigation_energy = [
    "irrigation_energy_diesel"
    , "irrigation_energy_electricity"
    , "irrigation_energy_pressure"
    , "irrigation_energy_solar"
]

for col in irrigation_energy:
    df[col] = df[col].cat.rename_categories({0: "", 1: "{} ".format(col[18:])}).copy()
df["irrigation_energy"] = pd.Categorical(df[irrigation_energy].astype(str).sum(axis=1))
cols = list(set(cols) - set(irrigation_energy))
cols.append("irrigation_energy")

####################

cover_crops = [
    "bare_soil"
    , "annual_cover_crop"
    , "permanent_cover_crop_native"
    , "permanent_cover_crop_non_native"
    , "permanent_cover_crop_volunteer_sward"]

for col in cover_crops:
    df[col] = df[col].cat.rename_categories({0: "", 1: "{} ".format(col)}).copy()
df["cover_crops"] = pd.Categorical(df[cover_crops].astype(str).sum(axis=1))
cols = list(set(cols) - set(cover_crops))
cols.append("cover_crops")

####################

water_type = [
    'river_water'
    , 'groundwater'
    , 'surface_water_dam'
    , 'recycled_water_from_other_source'
    , 'mains_water'
    , 'other_water'
    , 'water_applied_for_frost_control']

for col in water_type:
    df[col] = df[col].cat.rename_categories({0: "", 1: "{} ".format(col)}).copy()
df["water_type"] = pd.Categorical(df[water_type].astype(str).sum(axis=1))
cols = list(set(cols) - set(water_type))
cols.append("water_type")

####################

files = [f for f in listdir("./") if isfile(join("./", f))]
r = re.compile(".*_loss.csv")
files = list(filter(r.match, files))
files = [file[:-9] for file in files]
files = list(set(cols) - set(files))

cat_cols = [ # These are binary
    "water_type"
    , "cover_crops"
    , "irrigation_type"
    , "irrigation_energy"
    , "data_year_id" # These are one hot encoded
    , "giregion"
]

df = df.replace({0: np.nan})
for col in cols:
    print(df[col].describe())
    print("\n\n\n\n")

print(df["water_type"].value_counts().to_string())
print(df["cover_crops"].value_counts().to_string())
print(df["irrigation_type"].value_counts().to_string())
print(df["irrigation_energy"].value_counts().to_string())
print(df["data_year_id"].value_counts().to_string())
print(df["giregion"].value_counts().to_string())