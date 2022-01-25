#!/usr/bin/env python
# coding: utf-8

# # Deployment

# In[1]:


import pandas as pd
import json
from sklearn.ensemble import  GradientBoostingClassifier
import plotly.graph_objects as go
import plotly.express as px
import plotly.offline as py
import numpy as np


# Hyperparamter einlesen

# In[2]:


with open('../data/gradient_boosting_classifier_hyper_params.json','r') as file:
    hyper_params = json.load(file)


# Trainingsdaten für unser Modell einlesen

# In[3]:


X_train = pd.read_csv('../data/x_train_clean.csv', sep='$', decimal=".", engine='python') 
y_train = pd.read_csv('../data/y_train_clean.csv', sep='$', decimal=".", engine='python')   

del X_train['Unnamed: 0']
del y_train['Unnamed: 0']


# Testdatensatz einlesen und für das Modelling vorbereiten

# In[4]:


df = pd.read_csv('../data/test.csv', sep='[,$]' , decimal=".", engine='python')

df.columns = df.columns.str.lower()

df.rename(columns={
    'id': 'id',
    'gender': 'gender',
    'age': 'age',
    'driving_license': 'driving_license',
    'region_code': 'region_code',
    'previously_insured': 'previously_insured',
    'vehicle_age': 'vehicle_age',
    'vehicle__damage': 'vehicle_damage',
    'annual__premium': 'annual_premium',
    'policy_sales_channel': 'policy_sales_channel',
    'vintage': 'vintage'
},
    inplace=True)



df_one_hot_encoded_data = pd.get_dummies(df, columns = ['driving_license', 'gender', 'vehicle_age', 'vehicle_damage','previously_insured'])

df_one_hot_encoded_data.rename(columns={
    'id': 'id',
    'gender': 'gender',
    'age': 'age',
    'driving_license_0': 'driving_license_No',
    'driving_license_1': 'driving_license_Yes',
    'region_code': 'region_code',
    'previously_insured_0': 'previously_insured_No',
    'previously_insured_1': 'previously_insured_Yes',
    'vehicle_age': 'vehicle_age',
    'vehicle_damage': 'vehicle_damage',
    'annual_premium': 'annual_premium',
    'policy_sales_channel': 'policy_sales_channel',
    'vintage': 'vintage'
},
    inplace=True)

df_one_hot_encoded_data = df_one_hot_encoded_data.reindex(columns=['id', 'age', 'region_code', 'annual_premium', 'policy_sales_channel', 'vintage', 'driving_license_No', 'driving_license_Yes', 'vehicle_age_1-2 Year', 'vehicle_age_< 1 Year', 'vehicle_age_> 2 Years', 'vehicle_damage_No', 'vehicle_damage_Yes', 'previously_insured_No', 'previously_insured_Yes', 'gender_Female', 'gender_Male'])


# ## Modelling 

# In[5]:


# Modell für den Gradient Boosting Classifier instanzieren
gbc = GradientBoostingClassifier(**hyper_params ,random_state=42)

# Datensatz vorbereiten
X = df_one_hot_encoded_data.drop('id', axis=1)

# Modell fitten
fit_gbc = gbc.fit(X_train, y_train['response'])

# Vorhersagen 
y_pred = fit_gbc.predict(X)
y_pred


# In[26]:


# Data Frame mit den Vorhersagen erstellen
df_y_pred = pd.DataFrame(y_pred, columns=['y_pred'])

res = df_y_pred.set_index(df['id']).reset_index()
res['y_pred'].value_counts(normalize=True)


# In[27]:


df_pred = pd.merge(df, res, on='id')

# Data Frame mit den Daten der positiven Klasse erstellen
df_positive = df_pred[df_pred['y_pred'] == 1]
df_positive.describe().transpose()


# In[28]:


# DataFrame von der Variable response erzeugen
df_pred_response = df_pred.groupby(['y_pred']).size().reset_index()
df_pred_response['y_pred'] = df_pred_response['y_pred'].astype('str')

# Barplot der prozentualen Anteile in der Ausprägung von response erstellen
fig = px.bar(df_pred_response, x='y_pred', y=df_pred['y_pred'].value_counts(normalize=True), color='y_pred',
             text=df_pred['y_pred'].value_counts(normalize=True).apply(
                 lambda x: f'{np.round(x * 100, 2)}%'),
             color_discrete_map={
    '0': 'rgb(18,116,117)',
    '1': 'rgb(20,29,67)'
})

# Achsenbeschriftung hinzufügen
fig.update_layout(title='Relative Verteilung der Predictions',
                  xaxis_title='y_pred',
                  yaxis_title='Count')

# Plot erstellen
fig.show()


# In[29]:


# Prozentuale Verteilung je Kategorie in region_code absteigend nach 
# den höchsten Werten in einem df ablegen
df_region_code = round(df_positive['region_code'].value_counts(normalize=True).to_frame().sort_index(), 4)

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

# Plot erstellen
fig = dict(data=trace, layout=layout)
py.iplot(fig)


# In[30]:


# Prozentuale Verteilung je Kategorie in vehicle age absteigend nach 
# den höchsten Werten in einem df ablegen
df_vehicle_age = round(df_positive['vehicle_age'].value_counts(normalize=True).to_frame().sort_index(), 4)

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

# Plot erzeugen
fig = dict(data=trace, layout=layout)
py.iplot(fig)


# In[31]:


# Absolute Anzahl von Male und Female in dff abspeichern
dff = df_positive['gender'].value_counts() 

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


# In[32]:


# Prozentuale Verteilung je Kategorie in policy_sales_channel absteigend nach 
# den höchsten Werten in einem df ablegen

df_policy_sales_channel = pd.DataFrame()
df_policy_sales_channel = round(df_positive['policy_sales_channel'].value_counts(normalize=True).to_frame().sort_index().reset_index(), 4)
df_policy_sales_channel.columns = ['Channel', 'percentage']
df_policy_sales_channel['Channel'] = df_policy_sales_channel['Channel'].astype('str')
df_percentage_index = df_policy_sales_channel[df_policy_sales_channel['percentage'] < 0.02].index

df_policy_sales_channel = df_policy_sales_channel.drop(df_percentage_index)

trace = [
    # Barchart für die Verteilung insgesamt
    go.Bar(x=df_policy_sales_channel['Channel'],
           y=df_policy_sales_channel['percentage'],
           text=df_policy_sales_channel['percentage'],
           textposition="outside",
           name="total",
           hoverinfo="y",
           marker=dict(
               color=df_policy_sales_channel['percentage'],
               colorscale='ice',
               reversescale=True,
               showscale=True)
           )
]


# Layout des Plots definieren
layout = go.Layout(title=dict(text='Policy Sales Channel', y=.95),
                   yaxis_range=[0,.4],
                   plot_bgcolor='rgb(240, 240, 240)',
                   xaxis=dict(title="Policy Sales Channel",
                              titlefont=dict(size=16),
                              tickmode="linear"),
                   yaxis=dict(title="%",
                              titlefont=dict(size=17)),
                   )

fig = dict(data=trace, layout=layout)
py.iplot(fig)


# In[33]:


# DataFrame mit den Variablen age und response erstellen
df_age_response = df_pred[['age', 'y_pred']].interpolate(method='ffill')

# 
age_response_yes = df_age_response[df_age_response['y_pred'] == 1]

# Age = response ist no
age_response_no = df_age_response[df_age_response['y_pred'] == 0]

dmax = 10
dmin = 0

# Histogram erstellen
fig = px.histogram(df_age_response, x="age", color="y_pred",
                   marginal='box', histnorm='percent')
fig.update_layout(barmode='overlay')  # man kann auch stacked verwenden
fig.update_traces(opacity=0.65)

# Mean age von response = "yes" hinzufügen
fig.add_trace(go.Scatter(x=[np.mean(age_response_yes['age']), np.mean(age_response_yes['age'])],
                         y=[dmin, dmax],
                         mode='lines', opacity=.4,
                         line=dict(color='#1f77b4', width=2, dash='dash'),
                         name=f'Mean: {np.round(np.mean(age_response_yes["age"]), 2)}')
              )

# Mean age von response = "no" hinzufügen
fig.add_trace(go.Scatter(x=[np.mean(age_response_no['age']), np.mean(age_response_no['age'])],
                         y=[dmin, dmax],
                         mode='lines',
                         line=dict(color='rgba(248, 118, 109, 0.5)',
                                   width=2, dash='dash'),
                         name=f'Mean: {np.round(np.mean(age_response_no["age"]), 2)}')
              )

# Skalierung der Achsen anpassen
fig.update_layout(xaxis_type="linear", yaxis_type="linear")

fig.update_layout(
    title='Normalisiertes Historgram der Variable Age nach der Zielvariable')

# Generate Plot
fig.show()


# In[34]:


# Data Frame mit den Beiden variablen erzeugen
df_annual_premium = df_pred[['annual_premium', 'y_pred']]

df_annual_premium['annual_premium'] = df_annual_premium[['annual_premium']]

# annual_premium = response ist yes
annual_premium_response_yes = df_annual_premium[df_annual_premium['y_pred'] == 1]

# annual_premium = response ist no
annual_premium_response_no = df_annual_premium[df_annual_premium['y_pred'] == 0]

# Histogram definieren
fig = px.histogram(df_annual_premium, x="annual_premium",
                   color="y_pred", marginal='box', histnorm='percent')
fig.update_layout(barmode='overlay')  # man kann auch stacked verwenden
fig.update_traces(opacity=0.65)

# Mittlewert age von response = "yes"
fig.add_trace(go.Scatter(x=[np.mean(annual_premium_response_yes['annual_premium']), np.mean(annual_premium_response_yes['annual_premium'])],
                         y=[0, 10],
                         mode='lines', opacity=.4,
                         line=dict(color='#1f77b4', width=2, dash='dash'),
                         name=f'Median: {np.round(np.mean(annual_premium_response_yes["annual_premium"]), 2)}')
              )

# Mittlewert annual_premium von response = "no"
fig.add_trace(go.Scatter(x=[np.mean(annual_premium_response_no['annual_premium']), np.mean(annual_premium_response_no['annual_premium'])],
                         y=[0, 10],
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

