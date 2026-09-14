import pandas as pd
import numpy as np 
from tabulate import tabulate
from shapely.geometry import Point
import geopandas as gpd

# Then write a simple function to clean up the data and perform simple normalisation of text
def clean_data(df):

  new_df = df.copy()

  # Process MONTH to RESALE_YEAR and RESALE_MONTH
  new_df['MONTH'] = pd.to_datetime(new_df['MONTH'], format='%Y-%m')
  new_df['RESALE_YEAR'] = new_df['MONTH'].dt.year
  new_df['RESALE_MONTH'] = new_df['MONTH'].dt.month

  # Standardise FLAT_TYPE labels
  new_df.loc[:, 'FLAT_TYPE'] = new_df['FLAT_TYPE'].str.replace('-', ' ').str.strip().str.lower()

  # Standardise STREET labels
  new_df.loc[:, 'STREET'] = new_df['STREET'].str.lower()

  # Remove ECO_CATEGORY feature
  if 'ECO_CATEGORY' in new_df.columns:
    new_df = new_df.drop(columns=['ECO_CATEGORY'])

  return new_df

def get_nearest_amenity(hdb_df, amenity_df, amenity_colname='DIST_NEAREST', additional_cols=[]):
  # amenity_df: Desired amenity to find (E.g. hawker centers/schools/mrt etc)
  # amenity_colname: What to name the column containing the distance. 
  # additional_cols: Takes in a list, and returns all the additional ones in the amenity_df or hdb_df that you want.
  #
  # Returns a dataframe containing the primary key of the hdb (TOWN, BLOCK, ADDRESS) and the nearest distance, and all the additional cols desired.
  JOIN_KEY = ['TOWN','BLOCK','ADDRESS', 'POSTAL_CODE']
  hdb_gdf = gpd.GeoDataFrame(hdb_df, geometry=gpd.points_from_xy(hdb_df.LONGITUDE, hdb_df.LATITUDE), crs="EPSG:4326")
  amenity_gdf = gpd.GeoDataFrame(amenity_df, geometry=gpd.points_from_xy(amenity_df.LONGITUDE, amenity_df.LATITUDE), crs="EPSG:4326")

  # Convert lat/lon to meters (Singapore is small, use EPSG:3414 - SVY21 / Singapore TM)
  hdb_gdf = hdb_gdf.to_crs(epsg=3414)
  amenity_gdf = amenity_gdf.to_crs(epsg=3414)

  # Get nearest amemnity
  res = hdb_gdf.sjoin_nearest(amenity_gdf, distance_col=amenity_colname, lsuffix=None)
  res = res[JOIN_KEY + [amenity_colname] + additional_cols]
  # Remove duplicates when the dist is exactly the same.
  # It actually happens btw, probably due to the dataset having incorrect data for lat and lon
  res = res[~res[JOIN_KEY].duplicated()]
  
  assert hdb_df.shape[0] == res.shape[0], f"Num rows of df has changed! Before: {hdb_df.shape[0]}, After: {res.shape[0]}"

  return res

def get_amenities_within_dist(hdb_df, amenity_df, min_dist=0, max_dist=0, amenity_colname='NUM', return_agg_counts=False):
  # amenity_df: Desired amenity to find (E.g. hawker centers/schools/mrt etc)
  # min_dist: min distance to amenity in metres
  # max_dist: max distance to amenity in metres
  # amenity_colname: What to name the column containing the counts. Only used if return_agg_counts=True
  # return_agg_counts: If true, aggregates all the counts within the distance. Otherwise returns the df with the raw joined result instead.
  #
  # Returns a dataframe containing the primary key of the hdb (TOWN, BLOCK, ADDRESS) and the count of amneities within dist.
  JOIN_KEY = ['TOWN','BLOCK','ADDRESS', 'POSTAL_CODE']

  # Convert to GeoDataFrame
  hdb_gdf = gpd.GeoDataFrame(hdb_df, geometry=gpd.points_from_xy(hdb_df.LONGITUDE, hdb_df.LATITUDE), crs="EPSG:4326")
  amenity_gdf = gpd.GeoDataFrame(amenity_df, geometry=gpd.points_from_xy(amenity_df.LONGITUDE, amenity_df.LATITUDE), crs="EPSG:4326")

  # Convert lat/lon to meters (Singapore is small, use EPSG:3414 - SVY21 / Singapore TM)
  hdb_gdf = hdb_gdf.to_crs(epsg=3414)
  amenity_gdf = amenity_gdf.to_crs(epsg=3414)

  amnity_count_within_dists = []
  raw_df_result = []
  for dist in (min_dist, max_dist):
    # Spatial join (find amenities within buffer)
    amenity_gdf["buffer"] = amenity_gdf.buffer(dist)
    joined = gpd.sjoin(hdb_gdf, amenity_gdf.set_geometry("buffer"), how="inner", predicate="within", lsuffix=None)
    raw_df_result.append(joined)

    # Rename the column containing the count. Default name is '0'
    non_zero_amenity_counts = joined.groupby(JOIN_KEY).size().to_frame()

    # Debug code pls ignore
    # with pd.option_context('display.max_columns', 999):
    #   debug_df = joined.groupby(JOIN_KEY).apply(lambda x:x)

    # When we join we only get results where there is at least 1 amenity. This adds back hdb rows with 0 closeby amenities.
    all_amenity_counts = hdb_gdf[JOIN_KEY].merge(non_zero_amenity_counts, how='left', on=JOIN_KEY)
    all_amenity_counts[all_amenity_counts.isna()] = 0
    all_amenity_counts = all_amenity_counts.astype({0: np.int32})

    amnity_count_within_dists.append(all_amenity_counts)

  if not return_agg_counts:
    results_with_dist = pd.concat(raw_df_result).drop_duplicates(keep=False)
    return results_with_dist

  within_min_df, within_max_df = amnity_count_within_dists
  bounded_amenity_counts = within_min_df.merge(within_max_df, on=JOIN_KEY)
  # Deduct amenities within max_dist by amenity within min_dist to get difference
  bounded_amenity_counts[amenity_colname] = bounded_amenity_counts['0_y'] - bounded_amenity_counts['0_x']
  bounded_amenity_counts.drop(columns=['0_x', '0_y'], inplace=True)

  assert hdb_df.shape[0] == bounded_amenity_counts.shape[0], f'Result df row count changed! Expected {hdb_df.shape[0]} rows, but got: {bounded_amenity_counts.shape[0]}'
  return bounded_amenity_counts

def compute_num_mrt_lines(hdb_df, mrt_df, min_dist=0, max_dist=0, amenity_colname='NUM'):
  JOIN_KEY = ['TOWN','BLOCK','ADDRESS', 'POSTAL_CODE']

  #####
  # Compute NUM_MRT_LINES. Since these depend on the proximity of the hdb block, we need the hdb data, and cannot do it in process_mrt_df.
  #####
  hdb_lines_df = get_amenities_within_dist(hdb_df, mrt_df, min_dist, max_dist)[JOIN_KEY + ['CODE', 'STATUS']]
  # Map to only mrt code E.g. NE12 -> NE
  hdb_lines_df['CODE'] = hdb_lines_df['CODE'].map(lambda x: x[:2])
  # Remove duplicates formed by the previous step, hence getting unique unique tuples of JOIN_KEY, CODE, STATUS.
  # Aka Unique mrt lines and their states for a housing entry.
  hdb_lines_df = hdb_lines_df.drop_duplicates()

  hdb_lines_df_open = hdb_lines_df[hdb_lines_df['STATUS'] == 'open']
  non_zero_mrt_lines_open = hdb_lines_df_open.groupby(JOIN_KEY).count().reset_index()
  # Remove all rows except the join key and the num mrt lines count for the amenity joining later
  non_zero_mrt_lines_open = non_zero_mrt_lines_open.rename({'CODE': f'{amenity_colname}_OPEN'}, axis=1)[JOIN_KEY + [f'{amenity_colname}_OPEN']]

  # When we join we only get results where there is at least 1 amenity. This adds back hdb rows with 0 closeby amenities.
  no_mrt_lines_open = pd.concat([hdb_df[JOIN_KEY], non_zero_mrt_lines_open[JOIN_KEY]]).drop_duplicates(keep=False)
  no_mrt_lines_open[f'{amenity_colname}_OPEN'] = 0
  num_mrt_lines_open = pd.concat([non_zero_mrt_lines_open, no_mrt_lines_open])

  # Now we do for plan (Which is more painful)

  # For each housing, we left join the planned hdb lines with opened hdb lines, based on the mrt code.
  # If a mrt line of a given code is unable to find a opened mrt line with the same code. That means the line is defintely not opened within
  # the specified dist.
  hdb_lines_df_planned = hdb_lines_df[hdb_lines_df['STATUS'] == 'planned']
  non_zero_mrt_lines_plan = hdb_lines_df_planned.merge(hdb_lines_df_open, how='left', on=JOIN_KEY + ['CODE'])
  non_zero_mrt_lines_plan = non_zero_mrt_lines_plan[non_zero_mrt_lines_plan['STATUS_y'].isna()]
  non_zero_mrt_lines_plan = non_zero_mrt_lines_plan.groupby(JOIN_KEY).count().reset_index()
  # Remove all rows except the join key and the num mrt lines count for the amenity joining later
  non_zero_mrt_lines_plan = non_zero_mrt_lines_plan.rename({'CODE': f'{amenity_colname}_PLAN'}, axis=1)[JOIN_KEY + [f'{amenity_colname}_PLAN']]

  # When we join we only get results where there is at least 1 amenity. This adds back hdb rows with 0 closeby amenities.
  no_mrt_lines_plan = pd.concat([hdb_df[JOIN_KEY], non_zero_mrt_lines_plan[JOIN_KEY]]).drop_duplicates(keep=False)
  no_mrt_lines_plan[f'{amenity_colname}_PLAN'] = 0
  num_mrt_lines_plan = pd.concat([non_zero_mrt_lines_plan, no_mrt_lines_plan])

  return num_mrt_lines_open, num_mrt_lines_plan
  #####

def get_dist_to(hdb_df, longitude, latitude):
  hdb_gdf = gpd.GeoDataFrame(hdb_df, geometry=gpd.points_from_xy(hdb_df.LONGITUDE, hdb_df.LATITUDE), crs="EPSG:4326")
  hdb_gdf = hdb_gdf.to_crs(epsg=3857)
  ref_point = gpd.GeoSeries([Point(longitude, latitude)], crs="EPSG:4326")
  ref_point = ref_point.to_crs(epsg=3857)

  dists = hdb_gdf.distance(ref_point[0])
  return dists


def process_mrt_df(mrt_df):
  open_mrt_stops_df = mrt_df.copy()
  open_mrt_stops_df = open_mrt_stops_df[open_mrt_stops_df['STATUS'] == 'open']

  # planned_mrt_stops_df = mrt_df.copy()
  # planned_mrt_stops_df = planned_mrt_stops_df[planned_mrt_stops_df['STATUS'] == 'planned']

  open_mrt_int_df = mrt_df.copy()
  open_mrt_int_df = mrt_df[mrt_df['STATUS'] == 'open'].groupby('NAME').filter(lambda x: len(x) > 1)[['NAME', 'LATITUDE', 'LONGITUDE']].drop_duplicates()

  # planned_mrt_int_df = mrt_df.copy()
  # planned_mrt_int_df = mrt_df.groupby('NAME').filter(lambda x: len(x) > 1)[~mrt_df['NAME'].isin(open_mrt_int_df['NAME'])][['NAME', 'LATITUDE', 'LONGITUDE']].drop_duplicates()

  open_mrt_stn_df = mrt_df.copy()
  open_mrt_stn_df = open_mrt_stn_df[open_mrt_stn_df['STATUS'] == 'open'][['NAME', 'LATITUDE', 'LONGITUDE']].drop_duplicates()

  # planned_mrt_stn_df = mrt_df.copy()
  # planned_mrt_stn_df = planned_mrt_stn_df[~planned_mrt_stn_df['NAME'].isin(open_mrt_stn_df['NAME'])][['NAME', 'LATITUDE', 'LONGITUDE']].drop_duplicates()

  mrt_df_dict = {
      'mrt_df': mrt_df.copy(),
      'open_mrt_stops_df': open_mrt_stops_df,
      # 'planned_mrt_stops_df': planned_mrt_stops_df,
      'open_mrt_int_df': open_mrt_int_df,
      # 'planned_mrt_int_df': planned_mrt_int_df,
      'open_mrt_stn_df': open_mrt_stn_df,
      # 'planned_mrt_stn_df': planned_mrt_stn_df,
  }

  return mrt_df_dict

def process_mall_df(mall_df):
  new_df = mall_df.copy()
  return new_df

def process_hawker_df(hawker_df):
  new_df = hawker_df.copy()
  return new_df

def process_sch_df(sch_df):
  new_df = sch_df.copy()
  return new_df

def process_walk_dist_df(walk_dist_df):
  new_df = walk_dist_df.copy()
  return new_df

def process_hdb_df(hdb_df, mrt_df_dict, mall_df, hawker_df, pri_sch_df, sec_sch_df, walking_dist_df):
  new_df = hdb_df.copy()
  # Merge walking_dist_df with hdb_df. Right now its redundant as hdb_df is a subset of walking_dist_df
  new_df = new_df.merge(walking_dist_df, on=['TOWN', 'BLOCK', 'ADDRESS', 'POSTAL_CODE', 'LATITUDE', 'LONGITUDE', 'MAX_FLOOR', 'SUBZONE', 'PLANNING_AREA', 'REGION'], validate='1:1')

  num_amenities_df = [
     get_nearest_amenity(hdb_df, mrt_df_dict['open_mrt_stn_df'], 'DIST_NEAREST_MRT_OPEN'),
     # get_nearest_amenity(hdb_df, mrt_df_dict['planned_mrt_stn_df'], 'DIST_NEAREST_MRT_PLAN'),

     # get_nearest_amenity(hdb_df, hawker_df, 'DIST_NEAREST_HAWKER', additional_cols=['NUMBER_OF_STALLS']).rename(columns={'NUMBER_OF_STALLS': 'NEAREST_HAWKER_STALLS'}),

     get_amenities_within_dist(hdb_df, mrt_df_dict['open_mrt_stops_df'], 0, 1000, 'NUM_MRT_STOPS_1KM_OPEN', return_agg_counts=True),
     get_amenities_within_dist(hdb_df, mrt_df_dict['open_mrt_stops_df'], 1000, 2000, 'NUM_MRT_STOPS_2KM_OPEN', return_agg_counts=True),

     # get_amenities_within_dist(hdb_df, mrt_df_dict['planned_mrt_stops_df'], 0, 1000, 'NUM_MRT_STOPS_1KM_PLAN', return_agg_counts=True),
     # get_amenities_within_dist(hdb_df, mrt_df_dict['planned_mrt_stops_df'], 1000, 2000, 'NUM_MRT_STOPS_2KM_PLAN', return_agg_counts=True),

     get_amenities_within_dist(hdb_df, mrt_df_dict['open_mrt_int_df'], 0, 1000, 'NUM_MRT_INT_1KM_OPEN', return_agg_counts=True),
     get_amenities_within_dist(hdb_df, mrt_df_dict['open_mrt_int_df'], 1000, 2000, 'NUM_MRT_INT_2KM_OPEN', return_agg_counts=True),

     # get_amenities_within_dist(hdb_df, mrt_df_dict['planned_mrt_int_df'], 0, 1000, 'NUM_MRT_INT_1KM_PLAN', return_agg_counts=True),
     # get_amenities_within_dist(hdb_df, mrt_df_dict['planned_mrt_int_df'], 1000, 2000, 'NUM_MRT_INT_2KM_PLAN', return_agg_counts=True),

     get_amenities_within_dist(hdb_df, mrt_df_dict['open_mrt_stn_df'], 0, 1000, 'NUM_STN_1KM_OPEN', return_agg_counts=True),
     get_amenities_within_dist(hdb_df, mrt_df_dict['open_mrt_stn_df'], 1000, 2000, 'NUM_STN_2KM_OPEN', return_agg_counts=True),

     # get_amenities_within_dist(hdb_df, mrt_df_dict['planned_mrt_stn_df'], 0, 1000, 'NUM_STN_1KM_PLAN', return_agg_counts=True),
     # get_amenities_within_dist(hdb_df, mrt_df_dict['planned_mrt_stn_df'], 1000, 2000, 'NUM_STN_2KM_PLAN', return_agg_counts=True),

     get_amenities_within_dist(hdb_df, mall_df, 0, 1000, 'NUM_MALL_1KM', return_agg_counts=True),
     get_amenities_within_dist(hdb_df, mall_df, 1000, 2000, 'NUM_MALL_2KM', return_agg_counts=True),

     get_amenities_within_dist(hdb_df, hawker_df, 0, 1000, 'NUM_HAKWER_1KM', return_agg_counts=True),
     get_amenities_within_dist(hdb_df, hawker_df, 1000, 2000, 'NUM_HAWKER_2KM', return_agg_counts=True),

     get_amenities_within_dist(hdb_df, pri_sch_df, 0, 1000, 'NUM_PRI_SCH_1KM', return_agg_counts=True),
     get_amenities_within_dist(hdb_df, pri_sch_df, 1000, 2000, 'NUM_PRI_SCH_2KM', return_agg_counts=True),

     get_amenities_within_dist(hdb_df, sec_sch_df, 0, 1000, 'NUM_SEC_SCH_1KM', return_agg_counts=True),
     get_amenities_within_dist(hdb_df, sec_sch_df, 1000, 2000, 'NUM_SEC_SCH_2KM', return_agg_counts=True),
  ]

  JOIN_KEY = ['TOWN','BLOCK','ADDRESS', 'POSTAL_CODE']

  num_mrt_lines_1km_open, num_mrt_lines_1km_plan = compute_num_mrt_lines(hdb_df, mrt_df_dict['mrt_df'], min_dist=0, max_dist=1000, amenity_colname='NUM_MRT_LINES_1KM')
  num_amenities_df.append(num_mrt_lines_1km_open) # We don't add num_mrt_lines_1km_plan because we decided not to use planned mrt stuff in the model.

  for amenity_df in num_amenities_df:
    new_df = new_df.merge(amenity_df, on=JOIN_KEY)
  
  # CBD coords obtained from https://www.findlatitudeandlongitude.com/l/central+business+district%2C+singapore/424476/
  CBD_LON, CBD_LAT = 103.819836, 1.352083
  dists_to_cbd = get_dist_to(new_df, CBD_LON, CBD_LAT)
  new_df = new_df.join(dists_to_cbd.rename('DIST_TO_CBD'))

  # Coordinates obtained from https://sg.pagenation.com/sin/Singapore%20Geographival%20Origin_103.8333_1.3667.map and converted into lon and lat
  SG_CENTER_LON, SG_CENTER_LAT = 103.83306, 1.36667
  dists_to_center = get_dist_to(new_df, SG_CENTER_LON, SG_CENTER_LAT)
  new_df = new_df.join(dists_to_center.rename('DIST_TO_CENTER'))

  assert(hdb_df.shape[0] == new_df.shape[0]), f"process_hdb_df changed num rows, Before: {hdb_df.shape[0]}, After: {new_df.shape[0]}"
  return new_df


def integrate_aux_data(df, hdb_df):
  # Combines (Joins) train/test with all the auxilary data (Which should all be combined into the hdb_df at this point)

  joined_df = df.copy()
  joined_df['STREET'] = joined_df['STREET'].str.lower() # Standardise street and address convention
  joined_df = joined_df.merge(hdb_df, left_on=['TOWN', 'BLOCK', 'STREET'], right_on=['TOWN', 'BLOCK', 'ADDRESS']).drop('ADDRESS', axis=1)
  assert(joined_df.shape[0] == df.shape[0]), f'Joined row count not equal; Main df shape before: {df.shape}, after: {joined_df.shape}' # Ensure num rows the same

  return joined_df

def map_flat_type(x):
  first = str(x)[0].upper()
  if first.isdigit():
      return int(first)
  elif first == "E":
      return 6
  elif first == "M":
      return 7
  else:
      return None
    
# Create data processing function to encode features and create new useful features
def process_data(df):
  new_df = df.copy()

  # Compute TRIG_MONTH (MONTH_SIN/COS2 removed)
  new_df["MONTH_SIN1"] = np.sin(2 * np.pi * new_df["RESALE_MONTH"] / 12)
  new_df["MONTH_COS1"] = np.cos(2 * np.pi * new_df["RESALE_MONTH"] / 12)
  #new_df["MONTH_SIN2"] = np.sin(4 * np.pi * new_df["RESALE_MONTH"] / 12)
  #new_df["MONTH_COS2"] = np.cos(4 * np.pi * new_df["RESALE_MONTH"] / 12)

  # Compute REMAINING_LEASE
  new_df['REMAINING_LEASE'] = 99 - (new_df['RESALE_YEAR'] - new_df['LEASE_COMMENCE_DATA'])

  '''
  # Create auspicious/unauspicious block numbers
  new_df["BLOCK_NUM_8"] = new_df['BLOCK'].str.count("8")
  new_df["BLOCK_NUM_4"] = new_df['BLOCK'].str.count("4")
  '''

  # Create FLOOR_MID variable ###
  def floor_midpoint(floor_range):
      start, end = map(int, floor_range.split(' to '))
      return (start + end) // 2  # integer midpoint
  new_df['FLOOR_MID'] = new_df['FLOOR_RANGE'].apply(floor_midpoint)

  # Ordinal encoding for FLAT_TYPE
  new_df["ROOM_QTY"] = new_df["FLAT_TYPE"].apply(map_flat_type)

  '''
  # One-hot encoding for FLAT_MODEL
  encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
  FLAT_MODEL_OHE = encoder.fit_transform(new_df[['FLAT_MODEL']])
  FLAT_MODEL_OHE_df = pd.DataFrame(
      FLAT_MODEL_OHE,
      columns=encoder.get_feature_names_out(['FLAT_MODEL']),
      index=new_df.index
  )
  new_df = pd.concat([new_df, FLAT_MODEL_OHE_df], axis=1)
  '''
  return new_df

# Create data processing function to encode features and create new useful features
def add_new_features(df, is_test=False):
  path = '/content/drive/My Drive/CS5228 KDDM Project (TG14)/CS5228-14ANCHovYs/'
  auxiliary_data_path = path + 'data/auxiliary_data/'
  rental_df = pd.read_csv(auxiliary_data_path + 'RentingOutofFlats2025.csv')
  population_df = pd.read_csv(auxiliary_data_path + 'ResidentPopulationbyPlanningAreaSubzoneofResidenceEthnicGroupandSexCensusofPopulation2020.csv')
  accommsCPI_df = pd.read_csv(auxiliary_data_path + 'ConsumerPriceIndexAccommodation2024.csv')
  RPI_df = pd.read_csv(auxiliary_data_path + 'ResalePriceIndex2025Q3.csv')
  new_df = df.copy()

  # Rental - Yiwei
  ABBREV_MAP = {
    "ave":"avenue","aye":"ayer rajah expressway","bke":"bukit timah expressway",
    "bldg":"building","blk":"block","blvd":"boulevard","bo":"branch office",
    "br":"branch","bt":"bukit","cbd":"central business district","cc":"community centre",
    "ch":"church","cl":"close","clubhse":"clubhouse","condo":"condominium","cp":"carpark",
    "cplx":"complex","cres":"crescent","ct":"court","cte":"central expressway",
    "ctr":"centre","ctrl":"central","c'wealth":"commonwealth","dr":"drive",
    "ecp":"east coast expressway","est":"estate","fty":"factory","gdn":"garden",
    "gdns":"gardens","govt":"government","gr":"grove","hosp":"hospital","hq":"headquarters",
    "hse":"house","hts":"heights","ind":"industrial","inst":"institute","intl":"international",
    "jc":"junior college","jln":"jalan","kg":"kampong","kje":"kranji expressway",
    "kpe":"kallang paya lebar expressway","lib":"library","lk":"link","lor":"lorong",
    "mai":"maisonette","mce":"marina coastal expressway","met":"metropolitan",
    "mjd":"masjid","mkt":"market","mt":"mount","natl":"national","nth":"north",
    "pie":"pan island expressway","pk":"park","pl":"place","poly":"polyclinic",
    "pri":"primary","rd":"road","sch":"school","sec":"secondary","sle":"seletar expressway",
    "sq":"square","st":"street","sth":"south","stn":"station","tc":"town council",
    "tech":"technival","ter":"terrace","tg":"tanjong","tpe":"tampines expressway",
    "upp":"upper","voc":"vocational","warehse":"warehouse"
  }

  def build_key(s: pd.Series) -> pd.Series:
    x = s.astype(str).str.lower()
    for abbr, full in ABBREV_MAP.items():
      x = x.str.replace(rf"\b{abbr}\b", full, regex=True)
    x = x.str.replace(r"\d+", "", regex=True)
    x = x.str.replace(r"[^a-z\s]", " ", regex=True)
    x = x.str.replace(r"\s+", " ", regex=True).str.strip()
    return x

  def format_rental_df(rental: pd.DataFrame) -> pd.DataFrame:
    rental = rental.rename(columns={
      "street_name": "STREET",
      "street": "STREET",
      "block": "BLOCK",
      "flat_type": "FLAT_TYPE",
      "month": "MONTH",
      "monthly_rent": "RENTAL",
      "rent": "RENTAL",
    }).copy()
    rental["ROOM_QTY"] = rental["FLAT_TYPE"].map(map_flat_type)
    rental["STREET"] = rental["STREET"].astype(str)
    rental["STREET_KEY"]  = build_key(rental["STREET"])
    return rental
    
  def join_rental(prepped: pd.DataFrame, rental: pd.DataFrame) -> pd.DataFrame:
    prepped = prepped.copy()
    prepped["ROOM_QTY"] = prepped["ROOM_QTY"].replace({7: 6})
    prepped["STREET"] = prepped["STREET"].astype(str)
    prepped["STREET_KEY"] = build_key(prepped["STREET"])
    
    rent_mean = (
      rental
      .dropna(subset=["STREET_KEY", "ROOM_QTY", "RENTAL"])
      .groupby(["STREET_KEY", "ROOM_QTY"], as_index=False)["RENTAL"]
      .mean()
    )
    
    out = prepped.merge(
      rent_mean,
      on=["STREET_KEY", "ROOM_QTY"],
      how="left",
      validate="m:1"
    )
    return out

  rental_df = format_rental_df(rental_df)
  new_df = join_rental(new_df, rental_df)
  mean_dict = rental_df.groupby("ROOM_QTY")["RENTAL"].mean().to_dict()
  new_df["RENTAL"] = new_df["RENTAL"].fillna(
    new_df["ROOM_QTY"].map(mean_dict)
  )

  '''
  # Resident Population by Planning Area (Yiwei)
  pop = population_df.copy()
  pop["TOWN"] = (
    pop["Number"]
    .str.lower()
    .str.replace(r" - total$", "", regex=True)
    .str.replace("central subzone", "central area")
    .str.replace(r"(kallang|whampoa)", "kallang/whampoa", case=False, regex=True)
  )
  pop = pop[["TOWN","Total_Total"]].rename(columns={"Total_Total": "TOWN_POPULATION"})
  pop = (
    pop
    .sort_values("TOWN_POPULATION", ascending=False)
    .drop_duplicates(subset=["TOWN"], keep="first"))
  new_df = new_df.merge(
    pop,
    on = ["TOWN"],
    how = "left",
    validate="m:1"
  )
  new_df["TOWN_POPULATION"] = new_df["TOWN_POPULATION"].astype(int)
  '''


  '''
  # Housing CPI data - CHARISSA (some NaN values, will need to see if useful then maybe can project)
  accommsCPI_lookup = accommsCPI_df.set_index('Year')['Index'].to_dict()
  new_df['ANNUAL_ACCOMMS_CPI'] = new_df['RESALE_YEAR'].map(accommsCPI_lookup)
  '''

  # HDB resale price trends - CHARISSA
  new_df['RESALE_QUARTER'] = np.ceil(new_df['RESALE_MONTH'] / 3).astype(int)
  rpi_lookup = RPI_df.set_index(['Year','Quarter'])['Index'].to_dict()

  def get_rpi(row):
    year = row['RESALE_YEAR']
    month = row['RESALE_MONTH']

    # Handle July 2025 and later
    if year == 2025 and month >= 10:
      year = 2025
      month = 9  # force Sep 2025

    quarter = int(np.ceil(month / 3))
    return rpi_lookup.get((year, quarter), np.nan)

  new_df['RESALE_QUARTER_RPI'] = new_df.apply(get_rpi, axis=1)

  return new_df

# Function that adjusts all price-related variables w.r.t. FLOOR_AREA_SQM and/or RPI
def finalise_data(df, is_train=False, is_test=False, adj=None, model_means=None, typemodel_means=None, town_means=None, subzone_means=None, planning_area_means=None, region_means=None):
  new_df = df.copy()

  if adj=='sqm':
    new_df['RENTAL_PER_SQM'] = new_df['RENTAL'] / new_df['FLOOR_AREA_SQM']
    if not is_test:
      new_df['RESALE_PRICE_PER_SQM'] = new_df['RESALE_PRICE'] / new_df['FLOOR_AREA_SQM']
      target = 'RESALE_PRICE_PER_SQM'
  elif adj=='rpi':
    new_df['RENTAL_RPI_ADJ'] = new_df['RENTAL'] / new_df['RESALE_QUARTER_RPI'] * 100
    if not is_test:
      new_df['RESALE_PRICE_RPI_ADJ'] = new_df['RESALE_PRICE'] / new_df['RESALE_QUARTER_RPI'] * 100
      target = 'RESALE_PRICE_RPI_ADJ'
  elif adj=='both':
    new_df['RENTAL_PER_SQM_RPI_ADJ'] = new_df['RENTAL'] / new_df['FLOOR_AREA_SQM'] / new_df['RESALE_QUARTER_RPI'] * 100
    if not is_test:
      new_df['RESALE_PRICE_PER_SQM_RPI_ADJ'] = new_df['RESALE_PRICE'] / new_df['FLOOR_AREA_SQM'] / new_df['RESALE_QUARTER_RPI'] * 100
      target = 'RESALE_PRICE_PER_SQM_RPI_ADJ'
  else:
      target = 'RESALE_PRICE'

  # Target encoding for FLAT_MODEL
  if is_train:
    model_means = new_df.groupby("FLAT_MODEL")[target].mean().to_dict()
    global_model_mean = new_df[target].mean()
    new_df['MODEL_TE'] = new_df['FLAT_MODEL'].map(model_means).fillna(global_model_mean)
  else:
    if model_means is None:
      raise ValueError("model_means must be provided when is_train=False")
    global_model_mean = np.mean(list(model_means.values()))
    new_df['MODEL_TE'] = new_df['FLAT_MODEL'].map(model_means).fillna(global_model_mean)

  # Target encoding of FLAT_TYPE + FLAT_MODEL combined
  if is_train:
    new_df['TYPE_MODEL'] = new_df['FLAT_TYPE'].astype(str) + "_" + new_df['FLAT_MODEL'].astype(str)
    typemodel_means = new_df.groupby("TYPE_MODEL")[target].mean().to_dict()
    global_typemodel_mean = new_df[target].mean()
    new_df['TYPEMODEL_TE'] = new_df['TYPE_MODEL'].map(typemodel_means).fillna(global_typemodel_mean)
  else:
    if typemodel_means is None:
      raise ValueError("typemodel_means must be provided when is_train=False")
    global_typemodel_mean = np.mean(list(typemodel_means.values()))
    new_df['TYPE_MODEL'] = new_df['FLAT_TYPE'].astype(str) + "_" + new_df['FLAT_MODEL'].astype(str)
    new_df['TYPEMODEL_TE'] = new_df['TYPE_MODEL'].map(typemodel_means).fillna(global_typemodel_mean)

  # Target encoding of TOWN
  if is_train:
    town_means = new_df.groupby("TOWN")[target].mean().to_dict()
    global_town_mean = new_df[target].mean()
    new_df['TOWN_TE'] = new_df['TOWN'].map(town_means).fillna(global_town_mean)
  else:
    if town_means is None:
      raise ValueError("town_means must be provided when is_train=False")
    global_town_mean = np.mean(list(town_means.values()))
    new_df['TOWN_TE'] = new_df['TOWN'].map(town_means).fillna(global_town_mean)

  # Target encoding of SUBZONE
  if is_train:
    subzone_means = new_df.groupby("SUBZONE")[target].mean().to_dict()
    global_subzone_mean = new_df[target].mean()
    new_df['SUBZONE_TE'] = new_df['SUBZONE'].map(subzone_means).fillna(global_subzone_mean)
  else:
    if subzone_means is None:
      raise ValueError("subzone_means must be provided when is_train=False")
    global_subzone_mean = np.mean(list(subzone_means.values()))
    new_df['SUBZONE_TE'] = new_df['SUBZONE'].map(subzone_means).fillna(global_subzone_mean)

  # Target encoding of PLANNING_AREA
  if is_train:
    planning_area_means = new_df.groupby("PLANNING_AREA")[target].mean().to_dict()
    global_planning_area_mean = new_df[target].mean()
    new_df['PLANNING_AREA_TE'] = new_df['PLANNING_AREA'].map(planning_area_means).fillna(global_planning_area_mean)
  else:
    if planning_area_means is None:
      raise ValueError("planning_area_means must be provided when is_train=False")
    global_planning_area_mean = np.mean(list(planning_area_means.values()))
    new_df['PLANNING_AREA_TE'] = new_df['PLANNING_AREA'].map(planning_area_means).fillna(global_planning_area_mean)

  # Target encoding of REGION
  if is_train:
    region_means = new_df.groupby("REGION")[target].mean().to_dict()
    global_region_mean = new_df[target].mean()
    new_df['REGION_TE'] = new_df['REGION'].map(region_means).fillna(global_region_mean)
  else:
    if region_means is None:
      raise ValueError("region_means must be provided when is_train=False")
    global_region_mean = np.mean(list(region_means.values()))
    new_df['REGION_TE'] = new_df['REGION'].map(region_means).fillna(global_region_mean)

  new_order = ['MONTH', 'RESALE_YEAR', 'RESALE_QUARTER', 'RESALE_MONTH', 'LEASE_COMMENCE_DATA',
              'REMAINING_LEASE', 'MONTH_SIN1', 'MONTH_COS1',
              'POSTAL_CODE', 'LATITUDE', 'LONGITUDE', 'TOWN', 'TOWN_TE', 'REGION', 'REGION_TE',
                'PLANNING_AREA', 'PLANNING_AREA_TE', 'SUBZONE', 'SUBZONE_TE',
                'BLOCK', 'STREET', 'MAX_FLOOR', 'FLOOR_RANGE', 'FLOOR_MID',
                'FLAT_TYPE', 'ROOM_QTY', 'FLAT_MODEL', 'MODEL_TE', 'TYPEMODEL_TE'] + \
                [col for col in new_df.columns if col.startswith("FLAT_MODEL_")] + \
                [col for col in new_df.columns if col.startswith("NUM_")] + \
                [col for col in new_df.columns if col.startswith("DIST_")] + \
                  ['FLOOR_AREA_SQM', 'RENTAL', 'RESALE_QUARTER_RPI']
                  # 'MONTH_SIN2', 'MONTH_COS2', 'BLOCK_NUM_4', 'BLOCK_NUM_8', "TOWN_POPULATION", #'ANNUAL_ACCOMMS_CPI'
  if adj=='sqm':
    new_order.append('RENTAL_PER_SQM')
  elif adj=='rpi':
    new_order.append('RENTAL_RPI_ADJ')
  elif adj=='both':
    new_order.append('RENTAL_PER_SQM_RPI_ADJ')
  if not is_test:
    new_order.append('RESALE_PRICE')
    if target != 'RESALE_PRICE':
      new_order.append(target)
  new_df = new_df[new_order]

  if is_train:
    return new_df, model_means, typemodel_means, town_means, subzone_means, planning_area_means, region_means
  else:
    return new_df
  
def df_basic_info(df, table_name = None):
    title = f"=== {table_name} summary ==="
    data = [
    ["Number of records", len(df)],
    ["Number of features", len(df.columns)],
    ["Number of NaN rows", df.isna().sum().sum()],
    ["Number of duplicated rows", df.duplicated(keep=False).sum()]]

    print(f"\n{title}")
    print(tabulate(data, tablefmt="fancy_grid"))

def col_basic_info(df, col_name, table_name = None):
    title = f"=== {table_name}:  {col_name} summary ==="
    target_col = df[col_name]
    
    data = [["Data type", target_col.dtype],
            ["Number of unique values", len(target_col.unique())],
            ["Mode", target_col.mode()[0]]]
    print(title)
    print(tabulate(data, tablefmt="fancy_grid"))


def column_display(df, ncols, table_name = None):
    title = f"=== {table_name} columns display ===" if table_name else "=== Columns ==="
    print(f"\n{title}")
    
    if ncols:  
        rows = math.ceil(len(df.columns) / ncols)
        data = [df.columns[i*ncols:(i+1)*ncols] for i in range(rows)]
        print(tabulate(data, tablefmt="fancy_grid"))
    else: 
        data = [[i+1, col, df[col].dtype] for i, col in enumerate(df.columns)]
        print(tabulate(data, headers=["No.", "Column", "Dtype"], tablefmt="fancy_grid"))


# ===========================================================
# Attempted- Outlier Removal & Feature Interaction Utilities
# ===========================================================
# These utilities can be toggled on/off using the flags below.
# - Outlier removal uses Z-Score method (|Z| > 3)
# - Feature interaction generation uses sklearn's PolynomialFeatures
# ===========================================================

# --- Toggles ---
RUN_OUTLIER_REMOVAL = False      # Set True to enable outlier removal (Z-score > 3)
RUN_FEATURE_INTERACTIONS = False # Set True to enable interaction feature creation

import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures


# ===========================================================
# Attempted- Outlier Detection (Z-score Method)
# ===========================================================
def detect_univariate_outliers(df, columns=None, z_thresh=3):
    """
    Detects univariate outliers using Z-score threshold (default |Z| > 3).
    Returns the count of outlier features per row.
    """
    if not RUN_OUTLIER_REMOVAL:
        print("Outlier detection skipped (RUN_OUTLIER_REMOVAL = False).")
        return pd.Series([0]*len(df), index=df.index)

    if columns is None:
        columns = df.select_dtypes(include=np.number).columns

    z_scores = np.abs((df[columns] - df[columns].mean()) / df[columns].std())
    return (z_scores > z_thresh).sum(axis=1)


# ===========================================================
# Attempted-Outlier Removal (Z-score > 3)
# ===========================================================
def remove_outliers(df, target_col, z_thresh=3):
    """
    Removes rows that have more than 3 features flagged as outliers (|Z| > 3).
    Only applied if RUN_OUTLIER_REMOVAL = True.
    """
    if not RUN_OUTLIER_REMOVAL:
        print("Outlier removal skipped (RUN_OUTLIER_REMOVAL = False).")
        return df

    X = df.drop(columns=[target_col])
    y = df[target_col]
    outlier_counts = detect_univariate_outliers(X, z_thresh=z_thresh)
    mask = outlier_counts < 3
    cleaned_df = pd.concat([X[mask], y[mask]], axis=1).reset_index(drop=True)

    print(f"Removed {len(df) - len(cleaned_df)} outliers "
          f"({100*(1 - len(cleaned_df)/len(df)):.2f}% of rows)")
    return cleaned_df


# ===========================================================
# Attempted- Feature Interaction Generation (PolynomialFeatures)
# ===========================================================
def add_interaction_terms(X_train, X_val=None, degree=2, interaction_only=True):
    """
    Adds pairwise interaction terms using sklearn's PolynomialFeatures.
    Only applied if RUN_FEATURE_INTERACTIONS = True.
    """
    if not RUN_FEATURE_INTERACTIONS:
        print("Feature interaction generation skipped (RUN_FEATURE_INTERACTIONS = False).")
        if X_val is not None:
            return X_train, X_val, X_train.columns
        return X_train, X_train.columns

    poly = PolynomialFeatures(degree=degree, interaction_only=interaction_only, include_bias=False)
    X_train_poly = poly.fit_transform(X_train)
    feature_names = poly.get_feature_names_out(X_train.columns)
    X_train_df = pd.DataFrame(X_train_poly, columns=feature_names, index=X_train.index)

    if X_val is not None:
        X_val_poly = poly.transform(X_val)
        X_val_df = pd.DataFrame(X_val_poly, columns=feature_names, index=X_val.index)
        print(f"Added interaction features: {X_train_df.shape[1]} total features")
        return X_train_df, X_val_df, feature_names

    print(f"Added interaction features: {X_train_df.shape[1]} total features")
    return X_train_df, feature_names
