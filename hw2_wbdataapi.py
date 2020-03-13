import datetime
import pandas as pd
import wbdata

def readdata():

  wbdata.get_source()
  wbdata.get_indicator(source=2)
  data_date = datetime.datetime(2015, 1, 1)
  countries = [i['id'] for i in wbdata.get_country(incomelevel = "MIC", display=False)]
  
  x1 = wbdata.get_data("IP.JRN.ARTC.SC", data_date = data_date, country = countries , pandas = True)
  x2 = wbdata.get_data("EG.ELC.ACCS.RU.ZS", data_date = data_date, country = countries, pandas = True)
  y = wbdata.get_data("NY.GDP.PCAP.PP.KD", data_date = data_date, country = countries, pandas = True)
  data = pd.concat([x1, x2, y], axis = 1)
  data = data.dropna(axis=0, how='all')
  
  X = pd.concat([x1,x2], axis = 1)
  X = X.dropna(axis=0, how='all')

  data.columns = ["Scientific and technical journal articles", "Access to electricity, rural","GDPpc"]
  X.columns = ["Scientific and technical journal articles", "Access to electricity, rural","GDPpc"]

  x1 = data["Scientific and technical journal articles"].tolist()
  x2 = data["Access to electricity, rural"].tolist()
  y = data["GDPpc"].tolist()
  
  data.to_csv('data.csv')
  
  return (X,x1,x2,y)
