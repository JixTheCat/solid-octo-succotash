from os import listdir
from os.path import isfile, join

import pandas as pd
import re

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
    , "nh_frost"
    , "nh_disease"
    , "data_year_id" # These are one hot encoded
    , "giregion"
]

files = [f for f in listdir("./") if isfile(join("./", f))]

r = re.compile(".*imp")
files = list(filter(r.match, files))

df = pd.read_csv(files[0], index_col=0, names=[files[0][:-9]], header=0)

for filename in files[1:]:
    df[filename[:-9]] = pd.read_csv(files[0], index_col=0, header=0)

df = df[df.index[:3]].iloc[:3]

for col in list(set(cat_cols) - set(df.index)):
    r = re.compile("{}.*".format(col))
    rows = list(filter(r.match, list(df.index)))
    df.loc[col] = df.loc[rows].mean()
    df = df.drop(rows).copy()

df = df.stack().reset_index()
df.columns = ["source", "links", "value"]

df["value"] = round(df["value"]*1000000000).apply(int)

