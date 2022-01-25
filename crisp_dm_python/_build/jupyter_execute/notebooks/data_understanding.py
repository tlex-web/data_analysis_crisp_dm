#!/usr/bin/env python
# coding: utf-8

# # Data Understanding

# In[1]:


import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

import plotly.graph_objects as go
from plotly.missing_ipywidgets import FigureWidget
from plotly.subplots import make_subplots
import plotly.express as px
import plotly.figure_factory as ff
import plotly.offline as py
from plotly.offline import download_plotlyjs, init_notebook_mode, iplot, plot

import seaborn as sns


# In[2]:


def read_and_set_df(filepath: str, train: bool) -> pd.DataFrame:

    # Datensatz einlesen
    df = pd.read_csv(filepath, sep='$',
                     decimal=".", engine='python') 

    # Spaltennamen alle kleingeschrieben
    df.columns = df.columns.str.lower()

    # Die Spaltennamen waren verschoben - In diesem Schritt werden sie richtig zugeordnet
    df.rename(columns={
        'unnamed: 0': 'id',
        'id': 'gender',
        'gender': 'age',
        'age': 'driving_license',
        'driving_license': 'region_code',
        'region_code': 'previously_insured',
        'previously_insured': 'vehicle_age',
        'vehicle_age': 'vehicle_damage',
        'vehicle_damage': 'annual_premium',
        'annual_premium': 'policy_sales_channel',
        'policy_sales_channel': 'vintage',
        'vintage': 'response',
        'response': 'nan'
    },
        inplace=True)

    # Letzte Spalte besteht nur aus nan und kann somit gelöscht werden
    if not train:
        del df['nan']

    return df


def set_datatypes(df: pd.DataFrame) -> pd.DataFrame:

    # Numerische Variablen
    # Nullable Interger

    # Variable Age
    df["age"] = df["age"].astype(str)
    df["age"] = df["age"].str.rstrip('.')
    df['age'] = df["age"].replace('nan', np.nan)
    df['age'] = pd.to_numeric(df['age'], errors='raise')
    df["age"] = df["age"].astype('Int64')

    # Annual Premium
    df['annual_premium'] = df['annual_premium'].astype(str)
    df['annual_premium'] = df['annual_premium'].str.rstrip('.')
    df["annual_premium"] = pd.to_numeric(df["annual_premium"], errors='raise')
    df["annual_premium"] = df["annual_premium"].astype('Int64')

    # Vintage
    df['vintage'] = df['vintage'].astype(str)
    df['vintage'] = df['vintage'].str.rstrip('##')
    df['vintage'] = df["vintage"].replace('nan', np.nan)
    df["vintage"] = pd.to_numeric(df["vintage"], errors='raise')
    df["vintage"] = df["vintage"].astype('Int64')

    # Region Code
    df['region_code'] = df['region_code'].astype(str)
    df['region_code'] = df['region_code'].str.rstrip('#')
    df["region_code"] = pd.to_numeric(df["region_code"], errors='raise')
    df['region_code'] = df['region_code'].astype('category')

    # Policy Sales Channel
    df['policy_sales_channel'] = df['policy_sales_channel'].astype(str)
    df['policy_sales_channel'] = df['policy_sales_channel'].str.rstrip('##')
    df["policy_sales_channel"] = pd.to_numeric(
        df["policy_sales_channel"], errors='raise')
    df["policy_sales_channel"] = df["policy_sales_channel"].astype('Int64')

    # Kategorische Variablen
    df['gender'] = df['gender'].astype('category')
    df['driving_license'] = df['driving_license'].astype('category')
    df['previously_insured'] = df['previously_insured'].astype('category')
    df['vehicle_damage'] = df['vehicle_damage'].astype('category')
    df['vehicle_age'] = df['vehicle_age'].astype('category')

    # Response
    df.response.replace(
        {'0': 'no', '1': 'yes', 1: 'yes', 0: 'no'}, inplace=True)
    df['response'] = df['response'].astype('category')

    return df


# Nun wenden wir die zuvor definierte Funktion auf dem train.csv Datensatz an.

# In[3]:


df = read_and_set_df('../data/train.csv', train=False)

# Anwenden der zuvor definierten Funktion auf df
set_datatypes(df)


# Um einen ersten Überblick über die Variablen zu erahlten, werden zunächst folgende statistische Kennzahlen verwendet.

# In[4]:


# transpose = Tabelle transponieren für eine bessere Ansicht
df.describe(include='all').transpose()


# - `id`: min=1, max=380.999 -> es liegt eine fortlaufende Nummerierung der ID vor
# - `gender`: zwei Ausprägungen von Male und Female, in der Variable sind zudem Missing Values enthalten
# - `age`: min = 20, max = 205 -> extrem hoher max Wert. Deutet auf Ausreißer hin, enhält Missing Values
# - `driving_license`: enthält ebenfalls Missing Values, zwei Ausprägungen
# - `region_code`: keine Missing Values, 53 verschiedene Region Codes
# - `previously_insured`: enhält Missing Values, insgesamt zwei Ausprägungen
# - `vehicle_age`: enhält Missing Values, drei Ausprägungen
# - `vehicle_damage`: enhält Missing Values, zwei Ausprägungen
# - `annual_premium`: keine Missing Values, min = -9997, max=540.165, negative Werte erscheinen hier nicht logisch, Hohe Differenz zwischen Mean und Max deuten auf Ausreißer hin
# - `policy_sales_channel`: keine Missing Values, Channels von 1 bis 163
# - `vintage`: enthält Missing Values, min=10, max=299
# - `response`: Zielvarible enthält keine Missing Values, zwei Ausrägungen, top-Ausprägung ist no

# In[5]:


# Absolute Anzahl der Missing Values je Variable ausgeben lassen
pd.isna(df).sum()


# In diesem Schritt lassen wir uns die absolute Anzahl der Missing Values je Varialbe ausgeben
# 
# - Es wird ersichtlich, dass die meisten Missing Values in der Variable `age` vorliegen.
# - In der Variable `gender` liegen 1051 Missing Values.
# - In den Variablen `driving_license`, `previously_insured`, `vehicle_age`, `vehicle_damage` und `vintage` sind jeweils 51 Missing Values. 
# - In den restlichen Variablen sind keine Missing Values.

# # Grafische Datenanalyse
# 
# ## Geschlechtsverteilung
# 

# In[6]:


# Absolute Anzahl von Male und Female in dff abspeichern
dff = df['gender'].value_counts()[:10] 

# Label für den Pie chart festlegen
label = dff.index

# Summe der Ausprägungen je Label
size = dff.values

# Farben definieren und pie chart erzeugen
colors = ['rgb(20,29,67)', 'rgb(18,116,117)']
trace = go.Pie(labels=label, values=size, marker=dict(colors=colors), hole=.2)

data = [trace]

# Titel hinzufügen
layout = go.Layout(
    title='Geschlechtsverteilung'
)

# Plot erzeugen mit den zuvor definierten Spezifikationen
fig = go.Figure(data=data, layout=layout)
py.iplot(fig)


# In dem Datensatz befinden sich zu 45,9% Frauen und zu 54,1% Männer. Somit besteht ein sehr ausgewogenes Verhältnis zwischen Frauen und Männern in dem Datensatz!

# ## Vehicle Age
# 

# In[7]:


# Prozentuale Verteilung je Kategorie in vehicle age absteigend nach 
# den höchsten Werten in einem df ablegen
df_vehicle_age = round(df['vehicle_age'].value_counts(normalize=True).to_frame().sort_index(), 4)

# Den gleichen DataFrame erstellen wir respektive für Male und female
male_to_vehicle_age = round(df[df['gender'] == 'Male']['vehicle_age'].value_counts(normalize=True).to_frame().sort_index(), 4)
female_to_vehicle_age = round(df[df['gender'] == 'Female']['vehicle_age'].value_counts(normalize=True).to_frame().sort_index(), 4)

# Hier spezifizieren wir für gesamt, male und female jeweils einen Bar Chart
trace = [
    #Bar Chart für gesamt
    go.Bar(x=df_vehicle_age.index,
           y=df_vehicle_age['vehicle_age'],
           opacity=0.8,
           name="total",
           hoverinfo="y",
           marker=dict(
               color=df_vehicle_age['vehicle_age'],
               colorscale='ice',
               reversescale=True,
               showscale=True)
           ),
    # Bar Chart für Male
    go.Bar(x=male_to_vehicle_age.index,
           y=male_to_vehicle_age['vehicle_age'],
           visible=False,
           opacity=0.8,
           name="male",
           hoverinfo="y",
           marker=dict(
               color=male_to_vehicle_age['vehicle_age'],
               colorscale='ice',
               reversescale=True,
               showscale=True)
           ),
    # Bar Chart für female
    go.Bar(x=female_to_vehicle_age.index,
           y=female_to_vehicle_age['vehicle_age'],
           visible=False,
           opacity=0.8,
           name="female",
           hoverinfo="y",
           marker=dict(
               color=female_to_vehicle_age['vehicle_age'],
               colorscale='ice',
               reversescale=True,
               showscale=True)
           )
]

# Layout konfigurieren
layout = go.Layout(title=dict(text='Vehicle Age', y=.95),
                   plot_bgcolor='rgb(240, 240, 240)',
                   autosize=True,
                   xaxis=dict(title="vehicle_age",
                              titlefont=dict(size=15),
                              tickmode="linear"),
                   yaxis=dict(title="%",
                              titlefont=dict(size=20)),
                   )

# Attribute für das DropDown-Menü definieren
updatemenus = list([
    dict(
        buttons=list([
            dict(
                args=[{'visible': [True, False, False, False, False, False]}],
                label="Total",
                method='update',
            ),
            dict(
                args=[{'visible': [False, True, False, False, False, False]}],
                label="Male",
                method='update',
            ),
            dict(
                args=[{'visible': [False, False, True, False, False, False]}],
                label="Female",
                method='update',
            ),

        ]),
        direction="down",
        pad={'r': 10, "t": 10},
        x=0.1,
        y=1.25,
        yanchor='top',
    ),
])
layout['updatemenus'] = updatemenus

# Plot erzeugen
fig = dict(data=trace, layout=layout)
py.iplot(fig)


# __Gesamt:__ 
# - In dem oberen Plot kann man erkennen, dass 52,56% der Kunden ein Auto besitzen, das zwischen einem und zwei Jahre alt ist.  
# - 43,24% der Kunden besitzen ein Auto, das nicht älter als ein Jahr ist.
# - Lediglich 4,2% der gesamten Kunden in dem Datensatz besitzen ein Auto, das älter als zwei Jahre ist. 
# 
# __Male:__
# - Von den männlichen Kunden in dem Datensatz besitzen ebefalls mehr als die Hälte ein Auto, das zwischen einem und zwei Jahren alt ist. 
# - Darüber hinaus besitzen 35,65% ein Auto das nicht älter als ein Jahr ist und 5% ein Auto, das älter als zwei Jahre ist.
# 
# __Female:__
# - Bei den weiblichen Kunden unterscheidet sich die Verteilung. Hier besitzt die Mehrheit ein Auto, das noch kein Jahr alt ist. 
# - 44,56% besitzen ein Auto, dass zwischen einem und zwei Jahre alt ist. Ein Auto, das älter als zwei Jahre ist, besitzen lediglich 3,26% der weiblichen Kunden. 
# 
# 

# In[8]:


# Prozentuale Verteilung je Kategorie in region_code absteigend nach 
# den höchsten Werten in einem df ablegen
df_region_code = round(df['region_code'].value_counts(normalize=True).to_frame().sort_index(), 4)

# Den gleichen DataFrame erstellen wir respektive für Male und female
male_to_region_code = round(df[df['gender'] == 'Male']['region_code'].value_counts(normalize=True).to_frame().sort_index(), 4)
female_to_region_code = round(df[df['gender'] == 'Female']['region_code'].value_counts(normalize=True).to_frame().sort_index(), 4)

trace = [
    # Barchart für die Verteilung in der Variable region_code insgesamt erstellen
    go.Bar(x=df_region_code.index,
           y=df_region_code['region_code'],
           opacity=0.8,
           name="total",
           hoverinfo="y",
           marker=dict(
               color=df_region_code['region_code'],
               colorscale='ice',
               reversescale=True,
               showscale=True)
           ),

    # Barchart für die Verteilung bei male
    go.Bar(x=male_to_region_code.index,
           y=male_to_region_code['region_code'],
           visible=False,
           opacity=0.8,
           name="male",
           hoverinfo="y",
           marker=dict(
               color=male_to_region_code['region_code'],
               colorscale='ice',
               reversescale=True,
               showscale=True)
           ),

    # Barchart für die Verteilung bei female
    go.Bar(x=female_to_region_code.index,
           y=female_to_region_code['region_code'],
           visible=False,
           opacity=0.8,
           name="female",
           hoverinfo="y",
           marker=dict(
               color=female_to_region_code['region_code'],
               colorscale='ice',
               reversescale=True,
               showscale=True)
           )
]

# Layout des Plots definieren
layout = go.Layout(title=dict(text='Region Code Verteilung', y=.95),
                   plot_bgcolor='rgb(240, 240, 240)',
                   autosize=True,
                   xaxis=dict(title="Region Code",
                              titlefont=dict(size=16),
                              tickmode="linear"),
                   yaxis=dict(title="%",
                              titlefont=dict(size=17)),
                   )

# Attribute für das DropDown-Menü definieren
updatemenus = list([
    dict(
        buttons=list([
            dict(
                args=[{'visible': [True, False, False, False, False, False]}],
                label="Total",
                method='update',
            ),
            dict(
                args=[{'visible': [False, True, False, False, False, False]}],
                label="Male",
                method='update',
            ),
            dict(
                args=[{'visible': [False, False, True, False, False, False]}],
                label="Female",
                method='update',
            ),

        ]),
        direction="down",
        pad={'r': 10, "t": 10},
        x=0.1,
        y=1.25,
        yanchor='top',
    ),
])
layout['updatemenus'] = updatemenus

# Plot erstellen
fig = dict(data=trace, layout=layout)
py.iplot(fig)


# Wenn man sich die relativen Auspägungen der einzelnen Region Codes anschaut, fällt direkt auf, dass fast 28% der Kunden dem Region Code 28 zugeordnet sind. Des Weiteren sind ca. 9% der Kunden mit dem Region Code 8 klassifziert. Hierauf folgen Code 41 und 46 mit ca. 5%. Die restlichen Region Codes weisen einen relativen Anteil von unter 5% auf.

# In[9]:


# Prozentuale Verteilung je Kategorie in policy_sales_channel absteigend nach 
# den höchsten Werten in einem df ablegen
df_policy_sales_channel = round(df['policy_sales_channel'].value_counts(normalize=True).to_frame().sort_index(), 4)

# Den gleichen DataFrame erstellen wir respektive für Male und female
male_to_policy_sales_channel = round(df[df['gender'] == 'Male']['policy_sales_channel'].value_counts(normalize=True).to_frame().sort_index(), 4)
female_to_policy_sales_channel = round(df[df['gender'] == 'Female']['policy_sales_channel'].value_counts(normalize=True).to_frame().sort_index(), 4)

trace = [
    # Barchart für die Verteilung insgesamt
    go.Bar(x=df_policy_sales_channel.index,
           y=df_policy_sales_channel['policy_sales_channel'],
           opacity=0.8,
           name="total",
           hoverinfo="y",
           marker=dict(
               color=df_policy_sales_channel['policy_sales_channel'],
               colorscale='ice',
               reversescale=True,
               showscale=True)
           ),

    # Barchart für die Verteilung bei male
    go.Bar(x=male_to_policy_sales_channel.index,
           y=male_to_policy_sales_channel['policy_sales_channel'],
           visible=False,
           opacity=0.8,
           name="male",
           hoverinfo="y",
           marker=dict(
               color=male_to_policy_sales_channel['policy_sales_channel'],
               colorscale='ice',
               reversescale=True,
               showscale=True)
           ),


    # Barchart für die Verteilung bei female
    go.Bar(x=female_to_policy_sales_channel.index,
           y=female_to_policy_sales_channel['policy_sales_channel'],
           visible=False,
           opacity=0.8,
           name="female",
           hoverinfo="y",
           marker=dict(
               color=female_to_policy_sales_channel['policy_sales_channel'],
               colorscale='ice',
               reversescale=True,
               showscale=True)
           )
]

# Layout des Plots definieren
layout = go.Layout(title=dict(text='Policy Sales Channel', y=.95),
                   plot_bgcolor='rgb(240, 240, 240)',
                   autosize=True,
                   xaxis=dict(title="Policy Sales Channel",
                              titlefont=dict(size=16),
                              tickmode="linear"),
                   yaxis=dict(title="%",
                              titlefont=dict(size=17)),
                   )

# Attribute für das DropDown-Menü definieren
updatemenus = list([
    dict(
        buttons=list([
            dict(
                args=[{'visible': [True, False, False, False, False, False]}],
                label="Total",
                method='update',
            ),
            dict(
                args=[{'visible': [False, True, False, False, False, False]}],
                label="Male",
                method='update',
            ),
            dict(
                args=[{'visible': [False, False, True, False, False, False]}],
                label="Female",
                method='update',
            ),

        ]),
        direction="down",
        pad={'r': 10, "t": 10},
        x=0.1,
        y=1.25,
        yanchor='top',
    ),
])
layout['updatemenus'] = updatemenus

# Plot erstellen
fig = dict(data=trace, layout=layout)
py.iplot(fig)


# In der oberen Grafik wird deutlich, dass fast 80% der gesamten Kunden in dem Datensatz den Policy Sales Channels 26, 124 und 152 zugeordnet sind. 
# 
# Auffäligkeiten: Es gibt extrem viele Vertriebskanäle, über die nur sehr wenige bis keine Kunden angesprochen werden. 

# # Kategorische Variablen in Relation zur Zielvariable
# 
# ### Gender zur Zielvariable
# 

# In[10]:


# DataFrame von gender im Verhältnis zu response 
df_g = df.groupby(['gender', 'response']).size().reset_index()

# Die relativen prozentualen Anteile der Variable response im Verhältnis
# zur Variable gender als neue Spalte speichern
df_g['percentage'] = df.groupby(['gender', 'response']).size().groupby(level=0).apply(lambda x: 100 * x / float(x.sum())).values

# Variablennamen übergeben
df_g.columns = ['gender', 'response', 'Counts', 'Percentage']

# Bar Chart definieren
fig = px.bar(df_g, x='gender', y=['Counts'], color='response',
             text=df_g['Percentage'].apply(lambda x: '{0:1.2f}%'.format(x)),
             color_discrete_map={
    'yes': 'rgb(18,116,117)',
    'no': 'rgb(20,29,67)'
})

# Layout erstellen
fig.update_layout(title='Gender in Bezug zur Zielvariable',
                  xaxis_title='Gender',
                  yaxis_title='Count')

# Plot erzeugen                  
fig.show()


# Es gibt mehr Männer als Frauen in dem Datensatz. Darüber hinaus wird ersichtlich, dass relativ betrachtet, mehr Männer als Frauen eine Autoversicherung abgeschlossen haben.

# ### Driving_License zur Zielvariable
# 

# In[11]:


# DataFrame von driving_license im Verhältnis zu response
df_dl = df.groupby(['driving_license', 'response']).size().reset_index()

# Die relativen prozentualen Anteile der Variable response im Verhältnis
# zur Variable driving_license als neue Spalte speichern
df_dl['percentage'] = df.groupby(['driving_license', 'response']).size(
).groupby(level=0).apply(lambda x: 100 * x / float(x.sum())).values

# Variablennamen übergeben
df_dl.columns = ['driving_license', 'response', 'Counts', 'Percentage']

# Barplot für driving_license und response erstellen und das Layout definieren
fig = px.bar(df_dl, x='driving_license', y=['Counts'], color='response',
             text=df_dl['Percentage'].apply(lambda x: '{0:1.2f}%'.format(x)),
             color_discrete_map={
    'yes': 'rgb(18,116,117)',
    'no': 'rgb(20,29,67)'
})

# Achsenbeschriftung hinzufügen
fig.update_layout(title='Driving License in Bezug zur Zielvariable',
                  xaxis_title='Driving License',
                  yaxis_title='Count')

# Plot erstellen
fig.show()


# - Absolute Betrachtung: Es gibt mehr Kunden, die keinen Führerschein besitzen
# - Auffällig ist, dass fast ausschließlich Kunden ohne Führeschein eine Autoversicherung abgeschlossen haben. Diese Erkenntnis wirkt auf den esten Blick kontraintuitiv. Man sollte annehmen, dass nur Kunden, die einen Führerschein besitzen auch eine Autoversicherung abschließen.
# - Mögliche wäre, dass man für den Abschluss einer Versicherung keinen Nachweis über den Besitz eines Führerscheins vorlegen muss, und Eltern ihre Kinder als Versicherungsnehmer eintragen lassen. 

# ### Previously_Insured zur Zielvariable
# 

# In[12]:


# DataFrame von previously_insured zur Zielvariable
df_pi = df.groupby(['previously_insured', 'response']).size().reset_index()

# Die relativen prozentualen Anteile der Variable response im Verhältnis
# zur Variable previously_insured als neue Spalte speichern
df_pi['percentage'] = df.groupby(['previously_insured', 'response']).size(
).groupby(level=0).apply(lambda x: 100 * x / float(x.sum())).values
df_pi.columns = ['previously_insured', 'response', 'Counts', 'Percentage']

# Barplot für previously_insured und response erstellen und das Layout definieren
fig = px.bar(df_pi, x='previously_insured', y=['Counts'], color='response',
              text=df_pi['Percentage'].apply(lambda x: '{0:1.2f}%'.format(x)),
              color_discrete_map={
    'yes': 'rgb(18,116,117)',
    'no': 'rgb(20,29,67)'
})

# Achsenbeschriftung hinzufügen
fig.update_layout(title='previously_insured in Bezug zur Zielvariable',
                   xaxis_title='previously_insured',
                   yaxis_title='Count')

# Plot erstellen
fig.show()


# Es schließen fast ausschließlich Kunden eine Autoversicherung ab, die akutell noch keine Autoversicherung haben. 

# ### Vehicle_Age zur Zielvariable
# 

# In[13]:


# DataFrame von previously_insured zur Zielvariable
df_va = df.groupby(['vehicle_age', 'response']).size().reset_index()

# Die relativen prozentualen Anteile der Variable response im Verhältnis
# zur Variable vehicle_age als neue Spalte speichern
df_va['percentage'] = df.groupby(['vehicle_age', 'response']).size().groupby(
    level=0).apply(lambda x: 100 * x / float(x.sum())).values
df_va.columns = ['vehicle_age', 'response', 'Counts', 'Percentage']

# Barplot für vehicle_age und response erstellen und das Layout definieren
fig = px.bar(df_va, x='vehicle_age', y=['Counts'], color='response',
              text=df_va['Percentage'].apply(lambda x: '{0:1.2f}%'.format(x)),
              color_discrete_map={
    'yes': 'rgb(18,116,117)',
    'no': 'rgb(20,29,67)'
})

# Achsenbeschriftung hinzufügen
fig.update_layout(title='vehicle_age in Bezug zur Zielvariable',
                   xaxis_title='vehicle_age',
                   yaxis_title='Count')

# Plot erstellen
fig.show()


# Der Großteil der Kunden besitzt ein Auto, das nicht älter als 2 Jahre alt ist. Darüber hinaus fällt auf, dass Kunden mit einem Auto, dass älter als 1 Jahr alt ist, eine Autoversicherung abschließen (z.B. Wechsel der Versicherung bei Gebrauchtwagenkauf). 
# 
# - Man sollte annehmen, dass Kunden im Anschluss an den Kauf eine Autoversicherung abschließen (also in die Kategorie unter einem Jahr fallen).
# - Der absolute Anteil an Kunden, die eine Autoversicherung abschließen => besitzen ein Auto, das zwischen 1-2 Jahre alt ist. 
# - Somit lassen bei der NextGenInsurance GmbH mehr Kunden eine Gebrauchtwagen versichern als Neuwagen. 

# ### Vehicle_Damage zur Zielvariable
# 

# In[14]:


# DataFrame von previously_insured zur Zielvariable
df_vg = df.groupby(['vehicle_damage', 'response']).size().reset_index()

# Die relativen prozentualen Anteile der Variable response im Verhältnis
# zur Variable vehicle_damage als neue Spalte speichern
df_vg['percentage'] = df.groupby(['vehicle_damage', 'response']).size(
).groupby(level=0).apply(lambda x: 100 * x / float(x.sum())).values
df_vg.columns = ['vehicle_damage', 'response', 'Counts', 'Percentage']

# Barplot für vehicle_age und response erstellen und das Layout definieren
fig = px.bar(df_vg, x='vehicle_damage', y=['Counts'], color='response',
             text=df_vg['Percentage'].apply(lambda x: '{0:1.2f}%'.format(x)),
             color_discrete_map={
    'yes': 'rgb(18,116,117)',
    'no': 'rgb(20,29,67)'
})

# Achsenbeschriftung hinzufügen
fig.update_layout(title='Vehicle_Damage in Bezug zur Zielvariable',
                  xaxis_title='Vehicle_Damage',
                  yaxis_title='Count')

# Plot erstellen
fig.show()


# In dem Plot fällt auf, dass fast ausschließlich Kunden eine Autoversicherung abschließen, die bereits einen Schaden an ihrem Auto haben. Darüber hinaus ist der Anteil von Kunden mit und ohne einen Schaden an ihrem Auto sehr ausgeglichen. 

# ### Response
# 

# In[15]:


# DataFrame von der Variable response erzeugen
df_response = df.groupby(['response']).size().reset_index()

# Barplot der prozentualen Anteile in der Ausprägung von response erstellen
fig = px.bar(df_response, x='response', y=df['response'].value_counts(normalize=True), color='response',
             text=df['response'].value_counts(normalize=True).apply(
                 lambda x: '{0:1.2f}%'.format(x)),
             color_discrete_map={
    'yes': 'rgb(18,116,117)',
    'no': 'rgb(20,29,67)'
})

# Achsenbeschriftung hinzufügen
fig.update_layout(title='Relative Verteilung der Auprägungen von Response',
                  xaxis_title='Response',
                  yaxis_title='Count')

# Plot erstellen
fig.show()


# In der oberen Grafik wird die starke Unbalanciertheit der Zielvariable deutlich. Lediglich 12% der gesamten Kunden in dem Datensatz haben eine Autoversicherung abgeschlossen.

# ## Numerische Variablen darstellen
# 
# ### Histogram mit Mittelwerten der Variablen auf die Zielvariable
# 
# -   Da wir in der Variable `age` Missing Values vorfinden, können wir hier ohne Anpassung der NaN's keinen Histogram generieren. Da wir keinen Wert unter 20 Jahren haben setzen wir alle NaN's auf 0. Um eine Verzerrung bei der Berechnung vom Mittelwert zu vermeiden, haben wir hier alle Null Werte exkludiert.
# 
# -   Dasselbe haben wir bei der Variable `vintage` vollzogen. Auch hier gibt es keine Auspägungen, die 0 aufweisen. Aus diesem Grund haben wir hier auch die Missing Values auf 0 gesetzt.
# 
# ### Age
# 

# In[16]:


# DataFrame mit den Variablen age und response erstellen
df_age_response = df[['age', 'response']].interpolate(method='ffill')

# 
age_response_yes = df_age_response[df_age_response['response'] == 'yes']

# Age = response ist no
age_response_no = df_age_response[df_age_response['response'] == 'no']

dmax = 8
dmin = 0

# Histogram erstellen
fig = px.histogram(df_age_response, x="age", color="response",
                   marginal='box', histnorm='percent')
fig.update_layout(barmode='overlay')  # man kann auch stacked verwenden
fig.update_traces(opacity=0.65)

# Mean age von response = "yes" hinzufügen
fig.add_trace(go.Scatter(x=[np.mean(age_response_yes['age']), np.mean(age_response_yes['age'])],
                         y=[dmin, dmax],
                         mode='lines', opacity=.4,
                         line=dict(color='#1f77b4', width=2, dash='dash'),
                         name=f'Median: {np.round(np.mean(age_response_yes["age"]), 2)}')
              )

# Mean age von response = "no" hinzufügen
fig.add_trace(go.Scatter(x=[np.mean(age_response_no['age']), np.mean(age_response_no['age'])],
                         y=[dmin, dmax],
                         mode='lines',
                         line=dict(color='rgba(248, 118, 109, 0.5)',
                                   width=2, dash='dash'),
                         name=f'Median: {np.round(np.mean(age_response_no["age"]), 2)}')
              )

# Skalierung der Achsen anpassen
fig.update_layout(xaxis_type="linear", yaxis_type="log")

fig.update_layout(
    title='Normalisiertes Historgram der Variable Age nach der Zielvariable')

# Generate Plot
fig.show()


# - Minimun Alter  = 20 Jahre -> In Indien ist man mit 18 Jahren volljährig. 
# - Maximum Alter = 205 Jahre -> Erscheint nicht realistisch, da die durchschnittliche Lebenserwartungen in Indien unter 70 Jahren liegt. 
# 
# Response = Yes:
# - 25% der Kunden sind zwischen 20 und 34 Jahren alt
# - 50% der Kunden sind zwischen 34 und 50 Jahren alt
# - Der Median liegt bei 43 Jahren 
# - Der Mittelwert liegt aufgrund der Ausreißer bei ca. 43 Jahren
# - Der Älteste Kunde in der Klasse ist 198 Jahre alt
# 
# 
# Response = No:
# - 25% der Kunden sind zwischen 20 und 24 Jahren alt
# - 50% der Kunden sind zwischen 24 und 49 Jahren alt
# - Der Median liegt bei 32 Jahren 
# - Der Mittelwert liegt aufgrund der zahlreichen Ausreißer bei ca. 38 Jahren
# - Der Älteste Kunde in der Klasse ist 205 Jahre alt

# ### Annual Premium
# 

# In[17]:


# Data Frame mit den Beiden variablen erzeugen
df_annual_premium = df[['annual_premium', 'response']]

#df_annual_premium['annual_premium'] = np.log10(df_annual_premium[['annual_premium']])

# annual_premium = response ist yes
annual_premium_response_yes = df_annual_premium[df_annual_premium['response'] == 'yes']

# annual_premium = response ist no
annual_premium_response_no = df_annual_premium[df_annual_premium['response'] == 'no']

# Histogram definieren
fig = px.histogram(df_annual_premium, x="annual_premium",
                   color="response", marginal='box', histnorm='percent')
fig.update_layout(barmode='overlay')  # man kann auch stacked verwenden
fig.update_traces(opacity=0.65)

# Mittlewert age von response = "yes"
fig.add_trace(go.Scatter(x=[np.mean(annual_premium_response_yes['annual_premium']), np.mean(annual_premium_response_yes['annual_premium'])],
                         y=[0, 10],
                         mode='lines', opacity=.5,
                         line=dict(color='#1f77b4', width=2, dash='dash'),
                         name=f'Mean: {np.round(np.mean(annual_premium_response_yes["annual_premium"]), 2)}')
              )

# Mittlewert annual_premium von response = "no"
fig.add_trace(go.Scatter(x=[np.mean(annual_premium_response_no['annual_premium']), np.mean(annual_premium_response_no['annual_premium'])],
                         y=[0, 10],
                         mode='lines',
                         line=dict(color='rgba(248, 118, 109, 0.5)',
                                   width=2, dash='dash'),
                         name=f'Mean: {np.round(np.mean(annual_premium_response_no["annual_premium"]), 2)}')
              )

# Sklaierung der Achsen anpassen
fig.update_layout(xaxis_type="linear", yaxis_type="log")

# Überschrift hinzufügen
fig.update_layout(
    title='Normalisiertes Historgram der Variable Annual_Premium nach der Zielvariable')

# Plot erzeugen
fig.show()


# - Minimun Annual Premium  = -9.99k Rs -> negative Werte bei Annual Premium  
# - Maximum Annual Premium = 540.165k  Rs -> Extrem hoher Anteil an 
# - Auffällig ist die große Spannweite an Ausreißern zwischen 60 und 540.165k 
# 
# - Die Verteilung von Annual Premium je nach Ausprägung der Zielvariable ist sehr ähnlich
# - Darüber hinaus liegen 16,84% bei `response = no` zwischen 2500 und 2999 Rs
# - Darüber hinaus liegen 18,19% bei `response = yes` zwischen 2500 und 2999 Rs

# ## Vintage
# 

# In[18]:


# Data Frame mit den Variablen vintage und response erstellen und NaN mit 0 ersetzen
df_vintage_response = df[['vintage', 'response']].interpolate(method='ffill')
df_vintage_response['vintage'] = df_vintage_response['vintage']


# Vintage = response ist yes
vintage_response_yes = df_vintage_response[df_vintage_response['response'] == 'yes']

# Vintage = response ist no
vintage_response_no = df_vintage_response[df_vintage_response['response'] == 'no']

# Histogram
fig = px.histogram(df_vintage_response, x="vintage",
                   color="response",  histnorm='percent', marginal='box')
fig.update_layout(barmode='overlay')  # man kann auch stacked verwenden
fig.update_traces(opacity=0.65)

# Skalierung der Achsen anpassen
fig.update_layout(xaxis_type="linear", yaxis_type="linear")

# Überschrift hinzufügen
fig.update_layout(
    title='Normalisiertes Historgram der Variable Vintage nach der Zielvariable')

# Plot erzeugen
fig.show()


# Vintage:
# - min = 10 Tage 
# - max = 299 Tage 
# 
# - Auffällig ist hier, dass kein Kunde kürzer als 10 Tage und nicht länger als 299 Tage in einer Beziehung zu dem Unternehmen steht
# - Des Weiteren fällt auf, dass die Verteilung insgesamt normalverteilt ist

# ## Violin Plots
# 
# ### Customer Age - Sales Lead
# 

# In[19]:


# Hier müssen wir mal gucken  - ob es ok ist, dass wir die NaN's auf 0 gesetzt haben

fig = go.Figure()

options = ['yes', 'no']

for response in options:
    fig.add_trace(go.Violin(x=df_age_response['response'][df_age_response['response'] == response],
                            y=df_age_response['age'][df_age_response['response']
                                                     == response],
                            name=response,
                            box_visible=True,
                            meanline_visible=True))

fig.update_layout(title='Violin Plot',
                  xaxis_title='Sales Lead',
                  yaxis_title='Customer Age')
fig.show()


# ### Annual_Premium
# 

# In[20]:


fig = go.Figure()

options = ['yes', 'no']

for response in options:
    fig.add_trace(go.Violin(x=df_annual_premium['response'][df_annual_premium['response'] == response],
                            y=df_annual_premium['annual_premium'][df_annual_premium['response'] == response],
                            name=response,
                            box_visible=True,
                            meanline_visible=True))

fig.update_layout(title='Violin Plot',
                  xaxis_title='Sales Lead',
                  yaxis_title='Annual Premium')
fig.show()


# ### Vintage
# 

# In[21]:


fig = go.Figure()

options = ['yes', 'no']

for response in options:
    fig.add_trace(go.Violin(x=df_vintage_response['response'][df_vintage_response['response'] == response],
                            y=df_vintage_response['vintage'][df_vintage_response['response'] == response],
                            name=response,
                            box_visible=True,
                            meanline_visible=True))

fig.update_layout(title='Violin Plot',
                  xaxis_title='Response',
                  yaxis_title='Vintage')
fig.show()


# ## Korrelation

# In[22]:


# Create the correlation matrix
corr = df.corr(method='pearson')

# Generate a mask for the upper triangle; True = do NOT show
mask = np.zeros_like(corr, dtype=bool)
mask[np.triu_indices_from(mask)] = True

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))

# Generate a custom diverging colormap
cmap = sns.color_palette("crest", as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(
    corr,          # The data to plot
    mask=mask,     # Mask some cells
    cmap=cmap,     # What colors to plot the heatmap as
    annot=True,    # Should the values be plotted in the cells?
    vmax=.3,       # The maximum value of the legend. All higher vals will be same color
    vmin=-.3,      # The minimum value of the legend. All lower vals will be same color
    center=0,      # The center value of the legend. With divergent cmap, where white is
    square=True,   # Force cells to be square
    linewidths=.5,  # Width of lines that divide cells
    # Extra kwargs for the legend; in this case, shrink by 50%
    cbar_kws={"shrink": .5}
)


# -   In dem Plot werden lediglich die Variablen angezeigt, die keine Missing Values enthalten 
# -   Es wird ersichtlich, dass die Variable `annual_premium` und `age` die größte positive Korrelation mit 6,7% aufweisen.
# -   `policy_channel` und `age` weisen eine negative Korrelation von 57% auf.
# -   Die restlichen Variablen weisen keine nennenswerten linearen Abhängigkeiten auf.
# 

# In[23]:


# Create the correlation matrix
corr = df.corr(method='spearman')

# Generate a mask for the upper triangle; True = do NOT show
mask = np.zeros_like(corr, dtype=bool)
mask[np.triu_indices_from(mask)] = True

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))

# Generate a custom diverging colormap
cmap = sns.color_palette("crest", as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(
    corr,          # The data to plot
    mask=mask,     # Mask some cells
    cmap=cmap,     # What colors to plot the heatmap as
    annot=True,    # Should the values be plotted in the cells?
    vmax=.3,       # The maximum value of the legend. All higher vals will be same color
    vmin=-.3,      # The minimum value of the legend. All lower vals will be same color
    center=0,      # The center value of the legend. With divergent cmap, where white is
    square=True,   # Force cells to be square
    linewidths=.5,  # Width of lines that divide cells
    # Extra kwargs for the legend; in this case, shrink by 50%
    cbar_kws={"shrink": .5}
)


# Unterschied Pearson und Spearman:
# 
# - Die Korrelationskoeffizienten nach `Pearson` und `Spearman` können Werte zwischen −1 und +1 annehmen. Wenn der Korrelationskoeffizient nach Pearson +1 ist, gilt: Wenn eine Variable steigt, dann steigt die andere Variable um einen einheitlichen Betrag. Diese Beziehung bildet eine perfekte Linie. Der Korrelationseffizient nach Spearman ist in diesem Fall ebenfalls +1. 
# 
# - Wenn die Beziehung so geartet ist, dass eine Variable ansteigt, während die andere Variable ansteigt, der Betrag jedoch nicht einheitlich ist, ist der `Pearson-Korrelationskoeffizient` positiv, jedoch kleiner als +1. Der `Spearman-Koeffizient` ist in diesem Fall immer noch gleich +1.
# 
# 
# Entscheidung: 
# - Der Unterschied der beiden Berechnungsmethoden wird für uns dann interessant, wenn wir uns nicht nur dafür interessieren, ob die Variablen monotone Zusammenhänge aufweisen.
# - Wenn uns die linearen Abhängigkeiten interessieren, dann entscheiden wir uns für Pearson. 
# - Wenn wir lediglich wissen möchten, ob es monotone Beziehungen zwischen den Variablen gibt, dann entscheiden wir uns für Spearman.
# 
# => Da wir uns für die linearen Abhängigkeiten interessieren entscheiden wir uns für Pearson!

# # Missing Values
# 

# In[24]:


# Count Missing Values per Variable und in DataFrame überführen
y_count_mv = pd.DataFrame(df.isnull().sum())
y_count_mv.columns = ['count']
y_count_mv.index.names = ['Name']
y_count_mv['Name'] = y_count_mv.index
y_count_mv = y_count_mv[y_count_mv['count'] != 0]
y_count_mv.sort_values(by=['count'], inplace=True, ascending=True)


missing_values = pd.DataFrame(y_count_mv['count'] / len(df) * 100)
missing_values.columns = ['count']
missing_values.index.names = ['Name']
missing_values.sort_values(by=['count'], inplace=True, ascending=True)

x = y_count_mv['Name']

# Creating two subplots
fig = make_subplots(rows=1, cols=2, specs=[[{}, {}]], shared_xaxes=True,
                    shared_yaxes=False, vertical_spacing=0.001)

fig.append_trace(go.Bar(
    x=missing_values['count'],
    y=x,
    marker=dict(
        color='rgba(18, 63, 90, 0.95)',
        line=dict(
            color='rgba(18, 63, 90, 1.0)',
            width=1),
    ),
    name='Relative Anzahl der fehlenden Werte (%)',
    orientation='h',
), 1, 1)

fig.append_trace(go.Scatter(
    x=y_count_mv['count'], y=x,
    mode='lines+markers',
    line_color='rgb(0, 68, 27)',
    name='Absolute Anzahl der fehlenden Werte',
), 1, 2)

fig.update_layout(
    title='Absolute und relative Anzahl der Missing Values je Variable',
    yaxis=dict(
        showgrid=False,
        showline=False,
        showticklabels=True,
        domain=[0, 0.85],
    ),
    yaxis2=dict(
        showgrid=False,
        showline=False,
        showticklabels=False,
        linecolor='rgba(102, 102, 102, 0.8)',
        linewidth=2,
        domain=[0, 0.85],
    ),
    xaxis=dict(
        zeroline=False,
        showline=False,
        showticklabels=True,
        showgrid=True,
        domain=[0, 0.42],
    ),
    xaxis2=dict(
        zeroline=False,
        showline=False,
        showticklabels=True,
        showgrid=True,
        domain=[0.47, 1],
        dtick=2000,
    ),
    legend=dict(x=0.029, y=1.038, font_size=10),
    margin=dict(l=100, r=20, t=70, b=70),
    paper_bgcolor='rgb(248, 248, 255)',
    plot_bgcolor='rgb(248, 248, 255)',
)

annotations = []

y_s = np.round(missing_values['count'], decimals=2)
y_nw = np.rint(y_count_mv['count'])

# Label hinzufügen
for ydn, yd, xd in zip(y_nw, y_s, x):
    annotations.append(dict(xref='x2', yref='y2',
                            y=xd, x=ydn + 500 if ydn == max(y_nw) else ydn -500, # Verschieben der Zahlen im Liniendiagramm zu besseren Darstellung
                            text='{:,}'.format(ydn),
                            font=dict(family='Arial', size=12,
                                      color='rgb(0, 68, 27)'),
                            showarrow=False))
    annotations.append(dict(xref='x1', yref='y1',
                            y=xd, x=yd + 0.25,
                            text=str(yd) + '%',
                            font=dict(family='Arial', size=12,
                                      color='rgb(18, 63, 90)'),
                            showarrow=False))

fig.update_layout(annotations=annotations)

fig.show()


# - Die höchste Anzahl an Missing Values befinden sich mit 2,86% (10.892 Werte) in der Variable Age 
# - In der Variable Gender sind insgesamt 0,28% (1051 Stk) der Werte Missing Values 
# - In den Varialben vintage, vehicle_damage, vehicle_age, previously_insured und driving_license fehlen jeweils 0,01% (51 Stk) der Werte

# Wie viele Werte sind insgesamt Missing Values in dem Datensatz
# 

# In[25]:


def missing_values_table(df):
    # Total missing values
    mis_val = df.isnull().sum()

    # Percentage of missing values
    mis_val_percent = 100 * df.isnull().sum() / len(df)

    # Make a table with the results
    mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)

    # Rename the columns
    mis_val_table_ren_columns = mis_val_table.rename(
        columns={0: 'Missing Values', 1: '% of Total Values'})

    # Sort the table by percentage of missing descending
    mis_val_table_ren_columns = mis_val_table_ren_columns[
        mis_val_table_ren_columns.iloc[:, 1] != 0].sort_values(
        '% of Total Values', ascending=False).round(1)

    # Print some summary information
    print("Your selected dataframe has " + str(df.shape[1]) + " columns.\n"
          "There are " + str(mis_val_table_ren_columns.shape[0]) +
          " columns that have missing values.")

    # Return the dataframe with missing information
    return mis_val_table_ren_columns


train_missing = missing_values_table(df)
train_missing


# In[26]:


# Prozentuale Anzahl an Missing Values im gesamten Datensatz
np.round(train_missing['% of Total Values'].sum(),2)

