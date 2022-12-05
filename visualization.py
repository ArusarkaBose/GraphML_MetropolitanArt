import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

import warnings
warnings.filterwarnings('ignore')

data = pd.read_csv('MetObjects.csv', encoding='utf8')
metobjects_nonan = metobjects.copy()

#Drawing Visualization

#Visualization 2
count_artist = pd.DataFrame(data['Artist Display Name'].value_counts())
count_artist.columns = ['Count']
count_artist['Artist Name'] = count_artist.index.tolist()
count_artist.sort_values(by="Count", ascending=False)
count_artist = count_artist.reset_index(drop=True)

fig, ax = plt.subplots(figsize=(20, 10), subplot_kw=dict(aspect="equal"))

temp = count_artist[count_artist['Count'] >= 1000]

Culture = temp['Artist Name']
Count = temp['Count']

def func(pct, allvals):
    absolute = int(pct/100.*np.sum(allvals))
    return "{:.1f}%\n({:d} g)".format(pct, absolute)

wedges, texts, autotexts = ax.pie(Count, autopct=lambda pct: func(pct, Count),
                                  textprops=dict(color="w"))

ax.legend(wedges, Culture,
          title="Ingredients",
          loc="center left",
          bbox_to_anchor=(1, 0, 0.1
          ,1))

plt.setp(autotexts, size=8, weight="bold")
plt.show()

#Visualization 1
count_dept = pd.DataFrame(data['Department'].value_counts())
count_dept.columns = ['Count']
count_dept['Department Name'] = count_dept.index.tolist()
count_dept.sort_values(by="Count", ascending=False)
count_dept = count_dept.reset_index(drop=True)

fig, ax = plt.subplots(figsize=(20, 10), subplot_kw=dict(aspect="equal"))

temp = count_dept[count_dept['Count'] >= 1000]

Culture = temp['Department Name']
Count = temp['Count']

def func(pct, allvals):
    absolute = int(pct/100.*np.sum(allvals))
    return "{:.1f}%\n({:d} g)".format(pct, absolute)

wedges, texts, autotexts = ax.pie(Count, autopct=lambda pct: func(pct, Count),
                                  textprops=dict(color="w"))

ax.legend(wedges, Culture,
          title="Ingredients",
          loc="center left",
          bbox_to_anchor=(1, 0, 0.1
          ,1))

plt.setp(autotexts, size=8, weight="bold")
plt.show()

#Visualization 3
count_country = pd.DataFrame(data['Country'].value_counts())
count_country.columns = ['Count']
count_country['Country Name'] = count_country.index.tolist()
count_country.sort_values(by="Count", ascending=False)
count_country = count_country.reset_index(drop=True)

fig, ax = plt.subplots(figsize=(20, 10), subplot_kw=dict(aspect="equal"))

temp = count_country[count_country['Count'] >= 1000]

Culture = temp['Country Name']
Count = temp['Count']

def func(pct, allvals):
    absolute = int(pct/100.*np.sum(allvals))
    return "{:.1f}%\n({:d} g)".format(pct, absolute)

wedges, texts, autotexts = ax.pie(Count, autopct=lambda pct: func(pct, Count),
                                  textprops=dict(color="w"))

ax.legend(wedges, Culture,
          title="Ingredients",
          loc="center left",
          bbox_to_anchor=(1, 0, 0.1
          ,1))

plt.setp(autotexts, size=8, weight="bold")
plt.show()

#Visualization 4
data['Culture'].value_counts()[:10].plot(kind = 'bar')


