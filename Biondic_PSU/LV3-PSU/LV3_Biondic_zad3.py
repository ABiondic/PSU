import urllib.request as ur
import pandas as pd
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt

# url koji sadrzi xml datoteku s mjerenjima:
url = 'http://iszz.azo.hr/iskzl/rs/podatak/export/xml?postaja=160&polutant=5&tipPodatka=0&vrijemeOd=01.01.2017&vrijemeDo=31.12.2017'

airQualityHR = ur.urlopen(url).read()
root = ET.fromstring(airQualityHR)

df = pd.DataFrame(columns=('mjerenje', 'vrijeme'))

i = 0
while True:
    try:
        obj = root[i]
    except:
        break
    
    row = dict(zip(['mjerenje', 'vrijeme'], [obj[0].text, obj[2].text]))
    row_s = pd.Series(row)
    row_s.name = i
    
  
    df = pd.concat([df, row_s.to_frame().T], ignore_index=True)
    
   
    df.loc[i, 'mjerenje'] = float(df.loc[i, 'mjerenje'])
    i = i + 1


df.vrijeme = pd.to_datetime(df.vrijeme, utc=True)


df.plot(y='mjerenje', x='vrijeme')
plt.show()

najveci = df.sort_values(by='mjerenje', ascending=False).head(3)

print("Tri datuma s najvecom koncentracijom PM10:")
print(najveci[['vrijeme', 'mjerenje']])