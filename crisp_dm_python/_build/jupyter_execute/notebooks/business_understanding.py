#!/usr/bin/env python
# coding: utf-8

# # Data Analysis mit Python Projekt: CRISP-DM
# 
# ### Marc Bösen, Tim Lui und Feras Ghazal
# ### 16.11.2021

# # Business Understanding
# ## Ausgangssituation
# 
# Wir arbeiten für ein Beratungsunternehmen welches datengetriebene Lösungen für betriebswirtschaftliche Problemstellungen entwickelt. Unser Unternehmen wurde von einer Versicherungsgesellschaft beauftragt, ein Modell zu entwickeln um vorherzusagen ob ein Kunde ein Versicherungsprodukt abschließt oder nicht (`response` → Zielvariable).
# 
# Für das Cross Selling setzt die Versicherungsgesellschaft verschiedene Kanäle ein. Unter anderem Telefon, E-Mail, Recommendations im Online-Banking oder per App.
# 
# Zu diesem Zweck haben wir von unserem Auftraggeber einen Datenbestand mit 380.999 Zeilen und 12 verschiedenen Variablen erhalten. Neben dem Datenbestand haben wir von einem Verantwortlichen der NextGenInsurance eine Kurzbeschreibung des Unternehmens und des Produktes erhalten. Darüber hinaus wurde uns eine eine kurze Beschreibung der Daten in Form eines Data Dictionaries erhalten.  

# # Vorgehensweise
# Wir führen eine praktische Analyse des Datensatzes nach CRISP-DM durch. Auf den Schritt "Deployment" wird verzichtet, da die Ergebnisse vorher dem Auftraggeber präsentiert werden 

# Speicherort muss festelegt werden für Bilder und muss ein Markdown sein und kein Code 
# `<img src="Data_Analysis_WS2122/CRISP-DM_Process_Diagram.png" width="800" height="400">`

# In[1]:


import pandas as pd
import numpy as np


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


# In[4]:


df.describe(include = 'all').transpose() #transpose = Tabelle transponieren für eine bessere Ansicht
df.head()


# ### Attribute
# - `id`: Einmalige ID für einen Kunden 
# - `gender`: Geschlecht des Kunden 
# - `age`: Alter des Kunden
# - `driving_license`: 0: Der Kunde hat keinen Führerschein, 1: Der Kunde hat eine Führerschein
# - `region_code`: Eindeutiger Code für die Region des Kunden 
# - `previously_insured`: 0: Kunde hat keine Autoversicherung, 1: Kunde hat eine Autoversicherung 
# - `vehicle_age`: Alter des Fahrzeugs
# - `vehicle_damage`: 1 : Der Kunde hatte in der Vergangenheit einen Schaden an seinem Fahrzeug. 0 : Der Kunde hatte in der Vergangenheit einen Schaden an seinem Fahrzeug
# - `annual_premium`: Der Betrag, den der Kunde im Jahr als Prämie für die Krankenversicherung zu zahlen hat.
# - `policy_sales_channel`: Anonymisierter Code für den Kanal, über den der Kunde erreicht wird, d.h. verschiedene Agenten, per Post, per Telefon, persönlich, usw.
# - `vintage`: Anzahl der Tage, die der Kunde mit dem Unternehmen verbunden ist. 
# - `response`: 1: Der Kunde ist interessiert, 0: Der Kunde ist nicht interessiert

# ### Einheitliche Auspägungen bei den einzelnen Variablen erzeugen
# 
# - Mit der Fuktion `df['variable_name'].unique()` haben wir die verschiedenen Ausprägungen der Variablen untersucht
# - Im Folgenden erzeugen wir dann einheitliche Ausprägungen bei allen Variablen erzeugt 

# ### Casting der einzelnen Variablen 
# Die Datentypen werden für die weitere Verwendung angepasst:

# In[5]:


#set_datatypes(df)

df.info()

