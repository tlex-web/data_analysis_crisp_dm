#!/usr/bin/env python
# coding: utf-8

# # Data Understanding
# {ref}`beabe`

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
import missingno as msno


from sklearn.model_selection import KFold
from sklearn.impute import SimpleImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.impute import KNNImputer
from sklearn.svm import SVR
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.utils import resample
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as mse

from imblearn.over_sampling import SMOTE

import featuretools as ft

import tensorflow as tf


# In[2]:


def read_and_set_df(filepath: str, train: bool) -> pd.DataFrame:

    # with open(filepath) as file:
    # file.readlines()

    # Datensatz einlesen
    df = pd.read_csv(filepath, sep='$',  # r'([$-,])+/g'
                     decimal=".", engine='python')  # , na_values=[np.nan, pd.NA], keep_default_na=True)

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

    #df = df.replace(to_replace=['NaN', '<NA>', 'NAN', 'nan',
    #                pd.NA, np.nan, np.NaN, np.NAN], value=np.NaN, inplace=True)

    return df


# In[3]:


df = read_and_set_df('../data/train.csv', train=False)

set_datatypes(df)


# In[4]:


ind = df['gender'].isna()
df['gender'].loc[ind].unique()


# In[5]:


df[df['age'].isnull()]


# In[6]:


# transpose = Tabelle transponieren für eine bessere Ansicht
df.describe(include='all').transpose()


# In[7]:


pd.isna(df).sum()


# # Grafische Datenanalyse
# 
# ## Geschlechtsverteilung
# 

# In[8]:


dff = df['gender'].value_counts()[:10]
label = dff.index
size = dff.values

colors = ['rgb(20,29,67)', 'rgb(18,116,117)']
trace = go.Pie(labels=label, values=size, marker=dict(colors=colors), hole=.2)

data = [trace]
layout = go.Layout(
    title='Geschlechtsverteilung'
)

fig = go.Figure(data=data, layout=layout)
py.iplot(fig)


# ## Vehicle Age
# 

# In[9]:


df_vehicle_age = round(df['vehicle_age'].value_counts(normalize=True).
                       to_frame().sort_index(), 4)
male_to_vehicle_age = round(df[df['gender'] == 'Male']
                            ['vehicle_age'].value_counts(normalize=True).
                            to_frame().sort_index(), 4)
female_to_vehicle_age = round(df[df['gender'] == 'Female']
                              ['vehicle_age'].value_counts(normalize=True).
                              to_frame().sort_index(), 4)

trace = [
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

layout = go.Layout(title='',
                   paper_bgcolor='rgb(240, 240, 240)',
                   plot_bgcolor='rgb(240, 240, 240)',
                   autosize=True,
                   xaxis=dict(title="",
                              titlefont=dict(size=20),
                              tickmode="linear"),
                   yaxis=dict(title="%",
                              titlefont=dict(size=17)),
                   )

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

fig = dict(data=trace, layout=layout)
py.iplot(fig)


# In[10]:


df_region_code = round(df['region_code'].value_counts(normalize=True).
                       to_frame().sort_index(), 4)
male_to_region_code = round(df[df['gender'] == 'Male']
                            ['region_code'].value_counts(normalize=True).
                            to_frame().sort_index(), 4)
female_to_region_code = round(df[df['gender'] == 'Female']
                              ['region_code'].value_counts(normalize=True).
                              to_frame().sort_index(), 4)

trace = [
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

layout = go.Layout(title='',
                   paper_bgcolor='rgb(240, 240, 240)',
                   plot_bgcolor='rgb(240, 240, 240)',
                   autosize=True,
                   xaxis=dict(title="",
                              titlefont=dict(size=20),
                              tickmode="linear"),
                   yaxis=dict(title="%",
                              titlefont=dict(size=17)),
                   )

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

fig = dict(data=trace, layout=layout)
py.iplot(fig)


# # Kategroische Variablen in Relation zur Zielvariable
# 
# ### Gender zur Zielvariable
# 

# In[11]:


# Gender zur Zielvariable
df_g = df.groupby(['gender', 'response']).size().reset_index()
df_g['percentage'] = df.groupby(['gender', 'response']).size().groupby(
    level=0).apply(lambda x: 100 * x / float(x.sum())).values
df_g.columns = ['gender', 'response', 'Counts', 'Percentage']

fig = px.bar(df_g, x='gender', y=['Counts'], color='response',
             text=df_g['Percentage'].apply(lambda x: '{0:1.2f}%'.format(x)),
             color_discrete_map={
    'yes': 'rgb(18,116,117)',
    'no': 'rgb(20,29,67)'
})

fig.update_layout(title='Gender in Bezug zur Zielvariable',
                  xaxis_title='Gender',
                  yaxis_title='Count')
fig.show()


# ### Driving_License zur Zielvariable
# 

# In[12]:


# driving_license zur Zielvariable
df_dl = df.groupby(['driving_license', 'response']).size().reset_index()
df_dl['percentage'] = df.groupby(['driving_license', 'response']).size(
).groupby(level=0).apply(lambda x: 100 * x / float(x.sum())).values
df_dl.columns = ['driving_license', 'response', 'Counts', 'Percentage']

fig = px.bar(df_dl, x='driving_license', y=['Counts'], color='response',
             text=df_dl['Percentage'].apply(lambda x: '{0:1.2f}%'.format(x)),
             color_discrete_map={
    'yes': 'rgb(18,116,117)',
    'no': 'rgb(20,29,67)'
})

fig.update_layout(title='Driving_License in Bezug zur Zielvariable',
                  xaxis_title='Driving_License',
                  yaxis_title='Count')
fig.show()


# ### Previously_Insured zur Zielvariable
# 

# In[13]:


# previously_insured zur Zielvariable
df_pi = df.groupby(['previously_insured', 'response']).size().reset_index()
df_pi['percentage'] = df.groupby(['previously_insured', 'response']).size(
).groupby(level=0).apply(lambda x: 100 * x / float(x.sum())).values
df_pi.columns = ['previously_insured', 'response', 'Counts', 'Percentage']

fig2 = px.bar(df_pi, x='previously_insured', y=['Counts'], color='response',
              text=df_pi['Percentage'].apply(lambda x: '{0:1.2f}%'.format(x)),
              color_discrete_map={
    'yes': 'rgb(18,116,117)',
    'no': 'rgb(20,29,67)'
})

fig2.update_layout(title='previously_insured in Bezug zur Zielvariable',
                   xaxis_title='previously_insured',
                   yaxis_title='Count')
fig2.show()


# ### Vehicle_Age zur Zielvariable
# 

# In[14]:


# vehicle_age zur Zielvariable
df_va = df.groupby(['vehicle_age', 'response']).size().reset_index()
df_va['percentage'] = df.groupby(['vehicle_age', 'response']).size().groupby(
    level=0).apply(lambda x: 100 * x / float(x.sum())).values
df_va.columns = ['vehicle_age', 'response', 'Counts', 'Percentage']

fig2 = px.bar(df_va, x='vehicle_age', y=['Counts'], color='response',
              text=df_va['Percentage'].apply(lambda x: '{0:1.2f}%'.format(x)),
              color_discrete_map={
    'yes': 'rgb(18,116,117)',
    'no': 'rgb(20,29,67)'
})

fig2.update_layout(title='vehicle_age in Bezug zur Zielvariable',
                   xaxis_title='vehicle_age',
                   yaxis_title='Count')
fig2.show()


# ### Vehicle_Damage zur Zielvariable
# 

# In[15]:


# vehicle_damage zur Zielvariable
df_vg = df.groupby(['vehicle_damage', 'response']).size().reset_index()
df_vg['percentage'] = df.groupby(['vehicle_damage', 'response']).size(
).groupby(level=0).apply(lambda x: 100 * x / float(x.sum())).values
df_vg.columns = ['vehicle_damage', 'response', 'Counts', 'Percentage']

fig = px.bar(df_vg, x='vehicle_damage', y=['Counts'], color='response',
             text=df_vg['Percentage'].apply(lambda x: '{0:1.2f}%'.format(x)),
             color_discrete_map={
    'yes': 'rgb(18,116,117)',
    'no': 'rgb(20,29,67)'
})

fig.update_layout(title='Vehicle_Damage in Bezug zur Zielvariable',
                  xaxis_title='Vehicle_Damage',
                  yaxis_title='Count')
fig.show()


# ### Response
# 

# In[16]:


# Response
df_response = df.groupby(['response']).size().reset_index()
fig = px.bar(df_response, x='response', y=df['response'].value_counts(normalize=True), color='response',
             text=df['response'].value_counts(normalize=True).apply(
                 lambda x: '{0:1.2f}%'.format(x)),
             color_discrete_map={
    'yes': 'rgb(18,116,117)',
    'no': 'rgb(20,29,67)'
})

fig.update_layout(title='Relative Verteilung der Auprägungen von Response',
                  xaxis_title='Response',
                  yaxis_title='Count')
fig.show()


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

# In[17]:


# Data Frame mit den Variablen age und response erstellen
df_age_response = df[['age', 'response']]
df_age_response = df_age_response.dropna()

# Age = response ist yes
age_response_yes = df_age_response[df_age_response['response'] == 'yes']


# Age = response ist no
age_response_no = df_age_response[df_age_response['response'] == 'no']

dmax = 8
dmin = 0

# Histogram
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


# ### Annual Premium
# 

# In[18]:


# Data Frame mit den Beiden variablen erzeugen
df_annual_premium = df[['annual_premium', 'response']]

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
                         y=[0, 10000],
                         mode='lines', opacity=.4,
                         line=dict(color='#1f77b4', width=2, dash='dash'),
                         name=f'Median: {np.round(np.mean(annual_premium_response_yes["annual_premium"]), 2)}')
              )

# Mittlewert annual_premium von response = "no"
fig.add_trace(go.Scatter(x=[np.mean(annual_premium_response_no['annual_premium']), np.mean(annual_premium_response_no['annual_premium'])],
                         y=[0, 10000],
                         mode='lines',
                         line=dict(color='rgba(248, 118, 109, 0.5)',
                                   width=2, dash='dash'),
                         name=f'Median: {np.round(np.mean(annual_premium_response_no["annual_premium"]), 2)}')
              )

# Sklaierung der Achsen anpassen
fig.update_layout(xaxis_type="linear", yaxis_type="log")

# Überschrift hinzufügen
fig.update_layout(
    title='Normalisiertes Historgram der Variable Annual_Premium nach der Zielvariable')

# Plot erzeugen
fig.show()


# ## Vintage
# 

# In[19]:


# Data Frame mit den Variablen vintage und response erstellen und NaN mit 0 ersetzen
df_vintage_response = df[['vintage', 'response']]
df_vintage_response['vintage'] = df_vintage_response['vintage']
df_vintage_response = df_vintage_response.dropna()


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
fig.update_layout(xaxis_type="linear", yaxis_type="log")

# Überschrift hinzufügen
fig.update_layout(
    title='Normalisiertes Historgram der Variable Vintage nach der Zielvariable')

# Plot erzeugen
fig.show()


# ## Violin Plots
# 
# ### Customer Age - Sales Lead
# 

# In[20]:


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

# In[21]:


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

# In[22]:


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
# 
# -   Es wird ersichtlich, dass die Variable `annual_premium` und `age` die größte positive Korrelation mit 6,7% aufweisen.
# -   `policy_channel` und `age` weisen eine negative Korrelation von 57% auf.
# -   Des Weiteren weisen die restlichen Variablen keine nennenswerten Korrelationen auf.
# 

# In[23]:


# Create the correlation matrix
corr = df.corr()

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
        showline=True,
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
        side='top',
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

# Adding labels
for ydn, yd, xd in zip(y_nw, y_s, x):
    # labeling the scatter savings
    annotations.append(dict(xref='x2', yref='y2',
                            y=xd, x=ydn+500,
                            text='{:,}'.format(ydn),
                            font=dict(family='Arial', size=12,
                                      color='rgb(0, 68, 27)'),
                            showarrow=False))
    # labeling the bar net worth
    annotations.append(dict(xref='x1', yref='y1',
                            y=xd, x=yd + 0.15,
                            text=str(yd) + '%',
                            font=dict(family='Arial', size=12,
                                      color='rgb(18, 63, 90)'),
                            showarrow=False))

fig.update_layout(annotations=annotations)

fig.show()


# Wie viele Werte sind insgesamt Missing Values in dem Datensatz
# 

# In[25]:


# get the number of missing data points per column
missing_values_count = df.isnull().sum()

# how many total missing values do we have?
total_cells = np.product(df.shape)
total_missing = missing_values_count.sum()

# percent of data that is missing
total_percentage_missing = (total_missing/total_cells) * 100 * 100
print(f"{round(total_percentage_missing,2)} %")


# 

# In[26]:


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

