import csv
import pandas as pd
import numpy as np
import wbdata

def readdata():

  wbdata.get_source()
  wbdata.get_indicator(source=2)
  
  countries = [i['id'] for i in wbdata.get_country(incomelevel="HIC", display=False)]

  indicator1 = {"NY.GDP.PCAP.PP.KD": "gdppc", "IP.JRN.ARTC.SC": "Scientific_and_technical_journal_articles"}
  indicator2 = {"NY.GDP.PCAP.PP.KD": "gdppc", "SE.ADT.1524.LT.FM.ZS": "Literacyrate"}

  df = wbdata.get_dataframe(indicator1, country=countries, convert_date=True)
  df2=  wbdata.get_dataframe(indicator2, country=countries, convert_date=True)

  df = df.dropna()
  df2 = df2.dropna()

  df["uniqid"]=(df.country +df.date)
  df2["uniqid"]=(df2.country +df2.date)

  dfmerge = df.merge(df2,on="uniqid")
  dfmerge = dfmerge.drop(['country_y','date_y','gdppc_y'], axis=1)
  dfmerge = dfmerge.rename(columns = {"uniqid":"Unique_ID","date_x": "Date","country_x": "Country","gdppc_x": "GDPpc","Scientific_and_technical_journal_articles": "Scientific_And_Technical_Journal_Articles","Literacyrate": "Literacy_Rate"})

  dfmerge.to_csv("dfmerge.csv")
