#!/usr/bin/env python
# coding: utf-8

# # Data Preparation

# In[1]:


from collections import Counter
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score, auc
from sklearn.model_selection import train_test_split
import warnings

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV, GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as imbpipeline


# In[2]:


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


# In[3]:


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


# In[4]:


df = read_and_set_df('../data/train.csv', train=False)

set_datatypes(df)

df.head()


# In[5]:


df.describe(include='all').transpose()


# ## Missing Values

# Die Funktion `heatmap()` von `missingno` misst die Nullkorrelation: wie stark das Vorhandensein oder Fehlen einer Variable das Vorhandensein einer anderen Variable beeinflusst.
# 
# Die Nullkorrelation reicht von -1 (wenn eine Variable auftritt, tritt die andere definitiv nicht auf) über 0 (auftauchende oder nicht auftauchende Variablen haben keinen Einfluss aufeinander) bis 1 (wenn eine Variable auftritt, tritt die andere definitiv auf).
# 

# ### Verteilung der Missing Values 

# In[6]:


# Verteilung der Missing Values innerhalb der Variablen 
msno.matrix(df.sample(1000), sparkline=False, figsize=(10,5), fontsize=12, color=(0.27, 0.52, 1.0))


# In[7]:



plt.figure(figsize=(10,6))
sns.heatmap(df.isna().transpose(),
            cmap="YlGnBu",
            cbar_kws={'label': 'Missing Data'})


# In[8]:


# Visualize the correlation between the number of
# missing values in different columns as a heatmap
msno.heatmap(df)


# 
# 
# Um dieses Diagramm zu interpretieren, lesen Sie es aus einer Top-Down-Perspektive. Clusterblätter, die in einem Abstand von Null miteinander verbunden sind, sagen das Vorhandensein des jeweils anderen vollständig voraus - eine Variable könnte immer leer sein, wenn eine andere gefüllt ist, oder sie könnten immer beide gefüllt oder beide leer sein, und so weiter. In diesem speziellen Beispiel klebt das Dendrogramm die Variablen zusammen, die erforderlich und daher in jedem Datensatz vorhanden sind.
# 
# Clusterblätter, die sich in der Nähe von Null aufspalten, aber nicht bei Null, sagen sich gegenseitig sehr gut, aber immer noch unvollkommen voraus. Wenn Ihre eigene Interpretation des Datensatzes darin besteht, dass diese Spalten tatsächlich in Null übereinstimmen oder übereinstimmen sollten (z. B. als BETEILIGUNGSFAKTOR FAHRZEUG 2 und FAHRZEUG-TYPCODE 2), dann sagt Ihnen die Höhe des Clusterblatts in absoluten Zahlen, wie oft die Datensätze "nicht übereinstimmen" oder falsch abgelegt sind - d. h. wie viele Werte Sie ausfüllen oder streichen müssten, wenn Sie dazu geneigt sind.
# 
# Beschreibung: 
# Das Dendrogramm verwendet einen hierarchischen Clustering-Algorithmus, um die Variablen anhand ihrer Nullkorrelation gegeneinander abzugrenzen. 
# 
# 
# Erklärung: 
# Clutster, die sich in bei Null aufspalten, sagen sich untereinander vollkommen voraus (Korrelation von 1). Auf Grundlage 
# 
# Auf jeder Stufe des Baums werden die Variablen auf der Grundlage der Kombination aufgeteilt, die den Abstand der verbleibenden Cluster minimiert.  
# 
# Je monotoner die Variablen sind, desto näher liegt ihr Gesamtabstand bei Null und desto näher liegt ihr durchschnittlicher Abstand (die y-Achse) bei Null.   

# In[9]:


msno.dendrogram(df, orientation='top')


# In[10]:


# Verteilung der Missing Values innerhalb der Variablen 
#msno.matrix(df, freq='Tim', sparkline=False)
#df.iloc[:, 0]
#msno.matrix(df.set_index(pd.period_range('1/1/2011', '2/1/2015', freq='M')) , freq='BQ')


# In[11]:


# Drop all rows with NaNs in A OR B

#x = df.dropna(subset=['previously_insured', 'driving_license', 'vehicle_age', 'vehicle_damage', 'vintage'])


df.info()


# In[12]:


df_na_bool = pd.DataFrame(pd.isna(df))

df.drop(df_na_bool[(df_na_bool['previously_insured'] == True) & 
           (df_na_bool['driving_license'] == True) &
           (df_na_bool['vehicle_age'] == True) &
           (df_na_bool['vehicle_damage'] == True) &
           (df_na_bool['vintage'] == True)].index, inplace=True)

df.info()


# In[13]:


pd.isna(df).sum()


# Listenweiser Fallausschluss - Wir haben mittels dem listenweisen Fallausschluss 51 Zeilen aus dem Datensatz entfernt. Dabei haben wir ebenfalls die Anzahl der missing values bei den Variablen von age und gender um 51 Werte reduziert.

# ## Behandlung von Anomalien

# ### Age
# 
# - Untergrenze: 18
# - Obergrenze: 100

# In[14]:


index_max_age = df[df["age"] >= 100].index
df.drop(index_max_age, inplace=True)

index_min_age = df[df["age"] < 18].index
df.drop(index_min_age, inplace=True)

df["age"].describe()


# ### Annual Premium
# 
# - Untergrenze: 0
# - Obergrenze: 150.000

# In[15]:


index_min_premium = df[df["annual_premium"] <= 0].index
df.drop(index_min_premium, inplace=True)

index_max_premium = df[df["annual_premium"] >= 150000].index
df.drop(index_max_premium, inplace=True)

df["annual_premium"].describe()

df['vehicle_age'].unique()


# ## Train Test Splitting

# In[16]:


X = df.drop(['response', 'id'], axis=1)
y = df[['response']]

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.005, test_size = 0.009, random_state=42)

X_train_df = pd.DataFrame(X_train)
y_train_df = pd.DataFrame(y_train)
X_test_df = pd.DataFrame(X_test)
y_test_df = pd.DataFrame(y_test)

len(X_train_df), len(y_train_df), len(X_test_df), len(y_test_df)


# #### Categorial Mapping

# In[17]:


def map_categorials_x(df):

    # driving_license_map = {
    #     'No': 0,
    #     'Yes': 1
    # }

    # previously_insured_map = {
    #     'No': 0,
    #     'Yes': 1
    # }

    # vehicle_age_map = {
    #     '< 1 Year': 0,
    #     '1-2 Year': 1,
    #     '> 2 Years': 2
    # }

    # vehicle_damage_map = {
    #     'No': 0,
    #     'Yes': 1
    # }

    # df.loc[:,'driving_license'] = df['driving_license'].map(driving_license_map)
    # df.loc[:,'previously_insured'] = df['previously_insured'].map(previously_insured_map)
    # df.loc[:,'vehicle_age'] = df['vehicle_age'].map(vehicle_age_map).astype('Int64')
    # df.loc[:,'vehicle_damage'] = df['vehicle_damage'].map(vehicle_damage_map)

    LE = LabelEncoder()
    df['driving_license'] = LE.fit_transform(df.loc[:, 'driving_license'])
    df['previously_insured'] = LE.fit_transform(df.loc[:, 'previously_insured'])
    df['vehicle_age'] = LE.fit_transform(df.loc[:, 'vehicle_age'])
    df['vehicle_damage'] = LE.fit_transform(df.loc[:, 'vehicle_damage'])
    df['region_code'] = LE.fit_transform(df.loc[:,'region_code'])
    df['gender'] = LE.fit_transform(df.loc[:, 'gender'])

    return df

X_train_label_encoded = map_categorials_x(X_train_df.copy())
X_test_label_encoded = map_categorials_x(X_test_df.copy())

def map_categorials_y(df):
    LE = LabelEncoder()

    df['response'] = LE.fit_transform(df.loc[:,'response'])

    return df

y_train_label_encoded = map_categorials_y(y_train_df)
y_test_label_encoded = map_categorials_y(y_test_df)


# ## Imputationverfahren für die Variable Age

# ### Imputation auf der Train Batch

# #### Mean Imputation 

# In[18]:


# Create dataset
imputed_train = pd.DataFrame()
imputed_test = pd.DataFrame()
X_test_mean = X_test['age']
X_train_mean = X_train['age']

# Fill missing values of Age with the average of Age (mean)
imputed_train['age'] = X_train_mean.fillna(round(X_train_mean.mean(),0)).astype("Int64")
imputed_test['age'] = X_test_mean.fillna(round(X_test_mean.mean(),0)).astype("Int64")

train_imputed_mean_age = np.array(imputed_train['age'], dtype=int)
test_imputed_mean_age = np.array(imputed_test['age'], dtype=int)
actual_df_age = np.array(df['age'].dropna(), dtype=int)


# #### Median Imputation 

# In[19]:


# Create dataset
imputed_train_median = pd.DataFrame()
imputed_test_median = pd.DataFrame()
X_test_median = X_test['age']
X_train_median = X_train['age']

# Fill missing values of Age with the average of Age (median)
imputed_train_median['age'] = X_train_median.fillna(round(X_train_median.median(),0)).astype("Int64")
imputed_test_median['age'] = X_test_median.fillna(round(X_test_median.mean(),0)).astype("Int64")

train_imputed_median_age = np.array(imputed_train_median['age'], dtype=int)
test_imputed_median_age = np.array(imputed_test_median['age'], dtype=int)
actual_df_age = np.array(df['age'].dropna(), dtype=int)


# #### K-Nearest Neighbour

# In[20]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline

results = list()
strategies = [str(i) for i in [1,3,5,7,9,15,18,21]]
for s in strategies:
	# create the modeling pipeline
	pipeline = Pipeline(steps=[('i', KNNImputer(n_neighbors=int(s))), ('m', RandomForestClassifier())])
	# evaluate the model
	cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
	scores = cross_val_score(pipeline, X_train_label_encoded, y_train_label_encoded, scoring='accuracy', cv=cv, n_jobs=-1)
	# store results
	results.append(scores)
	print('>%s %.3f (%.3f)' % (s, np.mean(scores), np.std(scores)))
# plot model performance for comparison
plt.boxplot(results, labels=strategies, showmeans=True)
plt.show()


# In[21]:


# Modellierung auf Trainingsdaten
X_train_df_knn_X = X_train_label_encoded.copy()
knn = KNNImputer(n_neighbors=7, weights='uniform')

X_knn = knn.fit_transform(X_train_df_knn_X)
X_train_df_knn = pd.DataFrame(X_knn, columns=X_train_df_knn_X.columns)

# Modellierung auf Testdaten
X_test_df_knn_X = X_test_label_encoded.copy()
knn = KNNImputer(n_neighbors=7, weights='uniform')

X_knn = np.round(knn.fit_transform(X_test_df_knn_X))
X_test_df_knn = pd.DataFrame(X_knn, columns=X_test_df_knn_X.columns)


# In[22]:


# Create the correlation matrix
corr = X_train_df_knn.corr()

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
    linewidths=.5, # Width of lines that divide cells
    cbar_kws={"shrink": .5}  # Extra kwargs for the legend; in this case, shrink by 50%
)


# In[23]:


X_train_label_encoded['age'].unique()


# #### Miss Forest

# In[24]:


from sklearn.experimental import enable_iterative_imputer

# Modellierung auf den Trainingsdaten
X_train_df_mice_X = X_train_label_encoded.copy()

mice_imputer = IterativeImputer()
X_mice = mice_imputer.fit_transform(X_train_df_mice_X)
X_train_df_mice = pd.DataFrame(X_mice, columns=X_train_df_mice_X.columns)
X_train_df_mice['gender'] = X_train_df_mice['gender'].round()

# Modellierung auf den Testdaten
X_test_df_mice_X = X_test_label_encoded.copy()

X_mice = mice_imputer.fit_transform(X_test_df_mice_X)
X_test_df_mice = pd.DataFrame(X_mice, columns=X_test_df_mice_X.columns)
X_test_df_mice['gender'] = X_test_df_mice['gender'].round()


# #### Lineares Regressionsmodell

# ONE HOT encoding:
# 
# ONE-HOT-ENCODiNG transformiert kategoriale Variablen zu binären Variablen mittels des 'one-hot' Verfahrens.
# Dieser Schritt der Kodierung kategorialer Variablen ist nötig, um diese später in linearen Modellen und Vektor Maschinen zu verwenden.

# In[25]:


X_train_df_one_hot_encoded_data = pd.get_dummies(X_train_df, columns = ['driving_license', 'gender', 'vehicle_age', 'vehicle_damage', 'previously_insured'])#.dropna()
X_train_df_one_hot_encoded_data.head()


# In[26]:


corr = X_train_df_one_hot_encoded_data.corr()
corr.style.background_gradient(cmap='coolwarm').set_precision(2)


# In[27]:


# Modellierung auf den Trainingsdaten
df_linear_model_w_na = X_train_df_one_hot_encoded_data.copy().dropna(subset=['age', 'region_code'])
df_linear_model = X_train_df_one_hot_encoded_data.copy()[['age', 'region_code']]

X_reg = df_linear_model_w_na[['region_code']]
y_reg = df_linear_model_w_na[['age']]

age_missing = df_linear_model['age'].isnull()
df_age_missing = pd.DataFrame(df_linear_model['region_code'][age_missing])

X_train_regression, X_test_regression, y_train_regression, y_test_regression = train_test_split(X_reg, y_reg, train_size=0.8, test_size = 0.2, random_state=42)

lm = LinearRegression()
lm.fit(X_train_regression, y_train_regression)

#yp = pd.DataFrame(lm.predict(df_age_missing).round(), columns=['pred'])

#df_linear_model = df_linear_model['age'].apply(lambda x: x.fillna())
#df_linear_model.isna().sum(), yp['pred']


# #### Visualisierung der Imputationsverfahren

# In[28]:


hist_data_train = [X_train_df_mice['age'], X_train_df_knn['age'], train_imputed_mean_age, train_imputed_median_age, actual_df_age]

group_labels = ['train_imputed_mice_age','train_imputed_knn_age','train_imputed_mean_age', 'train_imputed_median_age', 'actual_df_age']
colors = ['#333F44', '#37AA9C', '#f3722c', '#6a994e', '#0077b6']

# Create distplot with curve_type set to 'normal'
fig = ff.create_distplot(hist_data_train, group_labels, show_hist=False, colors=colors, rug_text=None, show_rug=False)

# Add title
fig.update_layout(title_text='Verteilung der Variable nach Imputationsverfahren in der Train Batch')
fig.show()


# ## Imputationverfahren für die Variable Gender

# In[29]:



fig = make_subplots(rows=1)

fig.add_trace(go.Bar(name="Ursprüngliche Verteilung",x = X_train_df['gender'].replace({0: 'Female', 1: 'Male'}), y = X_train_df['gender'].value_counts(normalize=True),
                                    text = X_train_df['gender'].value_counts(normalize=True).apply(lambda x: '{0:1.3f}%'.format(x))
                        ))

fig.add_trace(go.Bar(name="Verteilung nach MICE Imputation", x = X_train_df_mice['gender'].replace({0: 'Female', 1: 'Male'}), y = X_train_df_mice['gender'].value_counts(normalize=True),
                                    text = X_train_df_mice['gender'].value_counts(normalize=True).apply(lambda x: '{0:1.3f}%'.format(x))
                        ))

fig.add_trace(go.Bar(name="Verteilung nach MICE Imputation", x = X_train_df_knn['gender'].replace({0: 'Female', 1: 'Male'}), y = X_train_df_knn['gender'].value_counts(normalize=True),
                                    text = X_train_df_knn['gender'].value_counts(normalize=True).apply(lambda x: '{0:1.3f}%'.format(x))
                        ))


fig.update_layout(title_text="Relative Verteilung der Variable Gender")

fig.show()


# ### Imputation auf der Test Batch 

# In[30]:


hist_data_test = [X_test_df_mice['age'], X_test_df_knn['age'], test_imputed_mean_age, test_imputed_median_age, actual_df_age]

group_labels = ['test_imputed_mice_age','test_imputed_knn_age','test_imputed_mean_age', 'test_imputed_median_age', 'actual_df_age']
colors = ['#333F44', '#37AA9C', '#f3722c', '#6a994e', '#0077b6']

# Create distplot with curve_type set to 'normal'
fig = ff.create_distplot(hist_data_test, group_labels, show_hist=False, colors=colors, rug_text=None, show_rug=False)

# Add title
fig.update_layout(title_text='Verteilung der Variable nach Imputationsverfahren in der Test Batch')
fig.show()


# In[31]:


corr = X_train_df_knn.corr()

mask = np.zeros_like(corr, dtype=bool)
mask[np.triu_indices_from(mask)] = True

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
    linewidths=.5, # Width of lines that divide cells
    cbar_kws={"shrink": .5}  # Extra kwargs for the legend; in this case, shrink by 50%
)


# #### Oversampling

# In[32]:


X_train_df_os = pd.concat([X_train,y_train],axis=1)

response_no = X_train_df_os[X_train_df_os.response == 'no']
response_yes = X_train_df_os[X_train_df_os.response == 'yes']

# upsample minority
response_upsampled = resample(response_yes,
                          replace=True, # sample with replacement
                          n_samples=len(response_no), # match number in majority class
                          random_state=42) # reproducible results

# combine majority and upsampled minority
upsampled = pd.concat([response_no, response_upsampled])

# check new class counts
upsampled.response.value_counts()


# In[33]:


df_upsampled_response = upsampled.groupby(['response']).size().reset_index()
fig = px.bar(df_upsampled_response, x='response', y=upsampled['response'].value_counts(normalize=False), color='response',
                                    text=upsampled['response'].value_counts(normalize=False),
                                    color_discrete_map={
                                        'yes': 'rgb(18,116,117)',
                                        'no': 'rgb(20,29,67)'
                                    })

fig.update_layout(title='Relative Verteilung der Ausprägungen von Response nach dem Upsampling',
                 xaxis_title='Response',
                 yaxis_title='Count')
fig.show()


# In[34]:


X_train_df_us = pd.concat([X_train_df_knn.reset_index(drop=True), y_train_df.reset_index(drop=True)], axis=1)
print(X_train_df_us.head())


# #### Undersampling mit einem einfachen Modell evaluieren 

# In[35]:


from imblearn.under_sampling import RandomUnderSampler
from collections import Counter

# define undersampling strategy
undersample = RandomUnderSampler(sampling_strategy='auto')

# summarize class distribution
print("Before undersampling: ", Counter(y_train['response']))

# fit and apply the transform
X_train_under, y_train_under = undersample.fit_resample(X_train_df_knn, y_train)
X_test_under, y_test_under = undersample.fit_resample(X_test_df_knn, y_test)

print(len(X_train_under), len(y_train_under))
print(len(X_test_under), len(y_test_under))
# summarize class distribution
print("After undersampling: ", Counter(y_train_under['response']))


from sklearn.svm import SVC
from sklearn.metrics import classification_report, roc_auc_score

model=SVC()
clf_under = model.fit(X_train_under, y_train_under)
pred_under = clf_under.predict(X_test_df_knn)

print("ROC AUC score for undersampled data: ", roc_auc_score(y_test_label_encoded, pred_under))


# In[36]:


# df_downsampled_response = downsampled.groupby(['response']).size().reset_index()
# fig = px.bar(df_downsampled_response, x='response', y=downsampled['response'].value_counts(normalize=False), color='response',
#                                     text=downsampled['response'].value_counts(normalize=False),
#                                     color_discrete_map={
#                                         'yes': 'rgb(18,116,117)',
#                                         'no': 'rgb(20,29,67)'
#                                     })

# fig.update_layout(title='Relative Verteilung der Ausprägungen von Response nach dem Undersampling',
#                  xaxis_title='Response',
#                  yaxis_title='Count')
# fig.show()


# #### Oversampling \w SMOTE

# SMOTE (Synthetic Minority Oversampling Technique) consists of synthesizing elements for the minority class, based on those that already exist. It works randomly picking a point from the minority class and computing the k-nearest neighbors for this point. The synthetic points are added between the chosen point and its neighbors.

# In[37]:


# import SMOTE oversampling and other necessary libraries 

pipeline = imbpipeline(steps = [['smote', SMOTE(sampling_strategy='auto' ,random_state=11, n_jobs=-1)],
                                ['scaler', MinMaxScaler()],
                                ['classifier', LogisticRegression(random_state=11,
                                                                  max_iter=1000)]])

stratified_kfold = StratifiedKFold(n_splits=5,
                                       shuffle=True,
                                       random_state=11)

param_grid = {'classifier__C':[0.001, 0.01, 0.1, 1, 10, 100, 1000]}
grid_search = GridSearchCV(estimator=pipeline,
                           param_grid=param_grid,
                           scoring='roc_auc',
                           cv=stratified_kfold,
                           n_jobs=-1)


grid_search.fit(X_train_df_knn, y_train_df['response'])
print(grid_search.best_params_)
cv_score = grid_search.best_score_
test_score = grid_search.score(X_test_df_knn, y_test_df['response'])
print(f'Cross-validation score: {cv_score}\nTest score: {test_score}')


# In[38]:


smote = SMOTE()

X_train_SMOTE, y_train_SMOTE = smote.fit_resample(X_train_df_knn, y_train_df['response'])
X_test_SMOTE, y_test_SMOTE = smote.fit_resample(X_test_df_knn, y_test_df['response'])


# In[39]:


# Create the correlation matrix
corr = X_train_SMOTE.corr()

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
    linewidths=.5, # Width of lines that divide cells
    cbar_kws={"shrink": .5}  # Extra kwargs for the legend; in this case, shrink by 50%
)


# ##### Feature Engineering 

# In[40]:


import featuretools as ft
from feature_engine.creation import CombineWithReferenceFeature
from feature_engine.creation import MathematicalCombination
from feature_engine.selection import SelectByTargetMeanPerformance

from sklearn.preprocessing import MinMaxScaler, QuantileTransformer, StandardScaler


# In[41]:


combinator = MathematicalCombination(
    variables_to_combine=['annual_premium', 'vintage'],
    math_operations=['mean', 'prod', 'sum', 'std'],
    new_variables_names=['mean_annual_vintage', 'prod_annual_vintage', 'sum_annual_vintage', 'std_annual_vintage']
)

X_train_df_fe = combinator.fit_transform(X_train_df_knn, y_train)

print(combinator.combination_dict_)

print(X_train_df_fe.loc[:, ['annual_premium', 'vintage', 'mean_annual_vintage', 'prod_annual_vintage', 'sum_annual_vintage', 'std_annual_vintage']].head())


# In[42]:


age_bin = X_train_df_fe[['age']]

X_train_df_fe['age_bin'] = pd.cut(X_train_df_fe['age'], bins=[18, 40, 60, 80, 100], labels=['18-40', '40-60', '60-80', '80-100'])
X_train_df_fe.isna().sum()


# Standard scaling removes mean and scale data to unit variance.

# In[43]:


standart_scaler = StandardScaler()

scaled_data = X_train_df_fe[['annual_premium']]

X_train_df_fe['annual_premium_scaled'] = standart_scaler.fit_transform(X_train_df_fe[['annual_premium']])

print('Mean:', X_train_df_fe['annual_premium_scaled'].mean())
print('Standard Deviation:', X_train_df_fe['annual_premium_scaled'].std())


# The most popular scaling technique is normalization (also called min-max normalization and min-max scaling). It scales all data in the 0 to 1 range.

# In[44]:


minmax_scaler = MinMaxScaler()

X_train_df_fe['annual_premium_min_max_scaled'] = minmax_scaler.fit_transform(X_train_df_fe[['annual_premium']])

print('Mean:', X_train_df_fe['annual_premium_min_max_scaled'].mean())
print('Standard Deviation:', X_train_df_fe['annual_premium_min_max_scaled'].std())


# As we mentioned, sometimes machine learning algorithms require that the distribution of our data is uniform or normal.

# In[45]:


qtrans = QuantileTransformer()

X_train_df_fe['annual_premium_q_trans_uniform'] = qtrans.fit_transform(X_train_df_fe[['annual_premium']])

print('Mean:', X_train_df_fe['annual_premium_q_trans_uniform'].mean())
print('Standard Deviation:', X_train_df_fe['annual_premium_q_trans_uniform'].std())


# In[46]:


qtrans_normal = QuantileTransformer(output_distribution='normal', random_state=42)

X_train_df_fe['annual_premium_q_trans_normal'] = qtrans_normal.fit_transform(X_train_df_fe[['annual_premium']])

print('Mean:', X_train_df_fe['annual_premium_q_trans_normal'].mean())
print('Standard Deviation:', X_train_df_fe['annual_premium_q_trans_normal'].std())


# In[47]:


corr = X_train_df_fe.corr()
corr.style.background_gradient(cmap='coolwarm').set_precision(2)


# #### Evalutation

# In[48]:


X_train_SMOTE.to_csv('./data/x_train_clean.csv', sep="$", decimal=".")
y_train_SMOTE.to_csv('./data/y_train_clean.csv', sep="$", decimal=".")

X_test_SMOTE.to_csv('./data/x_test_clean.csv', sep="$", decimal=".")
y_test_SMOTE.to_csv('./data/y_test_clean.csv', sep="$", decimal=".")

