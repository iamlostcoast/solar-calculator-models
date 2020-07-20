import time
from datetime import datetime
import random
import pandas as pd
import numpy as np
import sqlite3
import requests

"""
This script will query the NREL API for a list of zip/lat+lon pairs, find the average values for a bunch
of solar related data from that area.  It will return csvs that have saved all of that data.
It should be able to be run on AWS on a 24 chronjob, and automatically pick up from the last
zipcode that was successfully grabbed.
"""

# loading in the csv file that has all the zipcodes with their corresponding lat/lon pairs
good_zips = pd.read_csv("./good_zips_lat_lon.csv")

# making series of the lats, lons, and zipcodes
lats = good_zips["LAT"].values
lons = good_zips["LNG"].values
zips = good_zips["zipcode"].values

# Defining all our API values that we'll use later in the API queries.
# removed api_key for privacy
api_key = 'removed'
attributes = 'ghi,dhi,dni,wind_speed_10m_nwp,surface_air_temperature_nwp,solar_zenith_angle'
year = '2015'
leap_year = 'false'
interval = '60'
utc = 'false'
your_name = 'first+last'
reason_for_use = 'beta+testing'
your_affiliation = 'Test'
your_email = 'my_email_address@gmail.com'
mailing_list='true'


# Connecting to the database where we'll store failed zipcodes and our last_success zipcode.
conn = sqlite3.connect("./solar_logs.sqlite")

# This will try to find a csv that holds our last successfully found zipcode, if it doesn't
# exist, we'll just set it to 0 and start from the beginning.
starting_zipcode = pd.read_sql("SELECT * FROM last_success", con=conn)["zipcode"][0]
print "starting on: ", starting_zipcode

# This will set our top level for our high level iterator, based on our last success
top_of_loop = starting_zipcode + 2000

# Setting up a placeholder variable, we'll iteratively update this, and then at the end write
# it to our database so we can load it up the next time this script runs.
last_success = pd.read_sql("SELECT * FROM last_success", con=conn)["zipcode"][0]

# creating a failed_zips list that we'll later add to our database.
failed_zips = []

def query_api(fails, last_success):
    """
    This function queries the NREL API, find the average for the current lat lon, year, etc.,
    aggregates all the returned data and appends it to a dictionary defined outside the function.

    parameters:
        fails: defines the number of times the api has returned a 429 response.
                Used to stop the code if API quota has been exhausted/API is not working.
    """
    try:
        url = 'http://developer.nrel.gov/api/solar/nsrdb_0512_download.csv?wkt=POINT({lon}%20{lat})&names={year}&leap_day={leap}&interval={interval}&utc={utc}&full_name={name}&email={email}&affiliation={affiliation}&mailing_list={mailing_list}&reason={reason}&api_key={api}&attributes={attr}'.format(year=year, lat=lat, lon=lon, leap=leap_year, interval=interval, utc=utc, name=your_name, email=your_email, mailing_list=mailing_list, affiliation=your_affiliation, reason=reason_for_use, api=api_key, attr=attributes)
        # Return all but first 2 lines of csv to get data:
        info = pd.read_csv(url, nrows=2)
        df = pd.read_csv(url, skiprows=2)
        # creating a pivot table of our data, so that we get the average values for the whole year.
        pivot = df.pivot_table(["GHI","DHI","DNI","Wind Speed","Temperature","Solar Zenith Angle"], index="Year")
        for c in pivot.columns:
            zipcode_values[c].append(pivot[c].values[0])
        zipcode_values["Zipcode"].append(zips[bottom:top][i])
        print 'going to next zipcode'
        wait_time = random.randint(3,6)
        time.sleep(wait_time)
        # Updating last_success, it should be the top level iteration + lower level iteration.
        last_success = iteration + i
        # reset our successive_fails variable to 0, because we've had a success and want to stop counting them.
        fails = 0
    except:
        url = 'http://developer.nrel.gov/api/solar/nsrdb_0512_download.csv?wkt=POINT({lon}%20{lat})&names={year}&leap_day={leap}&interval={interval}&utc={utc}&full_name={name}&email={email}&affiliation={affiliation}&mailing_list={mailing_list}&reason={reason}&api_key={api}&attributes={attr}'.format(year=year, lat=lat, lon=lon, leap=leap_year, interval=interval, utc=utc, name=your_name, email=your_email, mailing_list=mailing_list, affiliation=your_affiliation, reason=reason_for_use, api=api_key, attr=attributes)
        # Add 1 to our successive_fails variable, which will iterate for every fail on the same zipcode.
        fails += 1
        resp = requests.get(url)
        response_code = resp.status_code
        print response_code
        if response_code == 429 and fails <= 5:
            wait_time = random.randint(6,8)
            time.sleep(wait_time)
            fails, last_success = query_api(fails, last_success)
        else:
            print "zipcode failed, going to next zipcode"
            # add this zipcode to the failed zips list.
            failed_zips.append(iteration + i)
            wait_time = random.randint(3,6)
            time.sleep(wait_time)

    # function spits out updated "successive fails" number and what the last success was
    return fails, last_success

# Setting normal iteration size to 500.  this will only change if the starting zip+2000 is greater than the lengeth of the csv we're drawing from.
iteration_size = 500

# Setting it up so if we're towards the end of the list, it'll just make one final csv.
if top_of_loop > len(zips):
    top_of_loop = len(zips)
    iteration_size = top_of_loop - starting_zipcode

# Setting up a higher level loop that will spit out a csv for every x number of zipcodes
for iteration in range(starting_zipcode, top_of_loop, iteration_size):

    zipcode_values = {
        "Zipcode": [],
        "DHI": [],
        "DNI": [],
        "GHI": [],
        "Solar Zenith Angle": [],
        "Temperature": [],
        "Wind Speed": []
    }

    bottom = int(iteration)
    top = int(iteration) + iteration_size
    # setting our successive_fails variable, used by the query_api function.
    successive_fails = 0

    for i, (lat, lon) in enumerate(zip(lats[bottom:top], lons[bottom:top])):
        print lat, lon, i

        # calling our query_api function, which should take in all the current variables
        # we have set and append data to the zipcode_values dictionary.
        # it should also be able to handle 429 responses.

        #if we get 15 successive 429 responses the query function will stop.
        if successive_fails <= 15:
            successive_fails, last_success = query_api(successive_fails, last_success)
            print "successive fails: ", successive_fails
        else:
            break

    # only export to csv and to the db if there's actually some data.
    if len(zipcode_values['Zipcode']) > 0:
        zipcode_df = pd.DataFrame(zipcode_values)
        # exporting the data as a csv.
        zip_filepath = "./zipcode_data_%s.csv" % str(bottom)
        zipcode_df.to_csv(zip_filepath)
        # add this to our database as well.
        zipcode_df.to_sql("zipcode_data", con=conn, if_exists='append')
        print "exporting csv for iteration range: %s : %s" % (bottom, top)

# change last_success table in db
last_success_df = pd.DataFrame({'zipcode': [last_success]})
last_success_df.to_sql("last_success", con=conn, if_exists='replace')

# adding the failed zips to the failed_zips table in our db
failed_zips_df = pd.DataFrame({'zip_index':failed_zips})
failed_zips_df.to_sql("failed_zips", if_exists='append', con=conn)

print datetime.now().time()
