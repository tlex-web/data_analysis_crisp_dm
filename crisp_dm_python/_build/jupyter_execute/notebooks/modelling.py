#!/usr/bin/env python
# coding: utf-8

# # Modelling

# In[1]:


import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_recall_curve, roc_curve, roc_auc_score, confusion_matrix, make_scorer, accuracy_score
from sklearn.metrics import auc

import json

import kds

from sklearn.model_selection import cross_val_score
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials , space_eval

import tensorflow 
from keras.models import Sequential
from keras.layers import Dense, BatchNormalization, Dropout
from tensorflow.keras.optimizers import Adam, SGD, RMSprop, Adadelta, Adagrad, Adamax, Nadam, Ftrl

from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.wrappers.scikit_learn import KerasClassifier
from math import floor
from sklearn.metrics import make_scorer, accuracy_score
from bayes_opt import BayesianOptimization
from sklearn.model_selection import StratifiedKFold
from keras.layers import LeakyReLU
LeakyReLU = LeakyReLU(alpha=0.1)
import warnings
warnings.filterwarnings('ignore')


# In[2]:


# Exportierten Datensätze der Data Preparation einlesen
X_train = pd.read_csv('../data/x_train_clean.csv', sep='$', decimal=".", engine='python') 
y_train = pd.read_csv('../data/y_train_clean.csv', sep='$', decimal=".", engine='python')   
X_test = pd.read_csv('../data/x_test_clean.csv', sep='$', decimal=".", engine='python') 
y_test = pd.read_csv('../data/y_test_clean.csv', sep='$', decimal=".", engine='python') 

del X_train['Unnamed: 0']
del y_train['Unnamed: 0']
del X_test['Unnamed: 0']
del y_test['Unnamed: 0']


# In[3]:


def get_auc_pr(y_test, y_proba):
    precision, recall, _ = precision_recall_curve(y_test, y_proba)
    return auc(recall, precision)


# ## Random Forest Classifier Train Set

# Default Values for Random Forest:
# 
# * `n_estimators`: 100
# * `criterion`: 'gini'
# * `max_depth`: None
# * `main_samples_split`: 2
# * `min_samples_leaf`: 1
# * `min_weight_frachtion`: 0
# * `max_features`: 'auto'

# In[4]:


def hyperopt_train_test(params):
    """
    Instanz für den Random Forest Classifier erstellen 
    Berechnung des Mittelwertes bei 5-facher Crossvalidierung auf Basis des pr_auc scorings
    """
    clf = RandomForestClassifier(**params, n_jobs=-1, random_state=42)

    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    cv = cross_val_score(clf, X_train, y_train['response'], scoring='roc_auc', cv=kfold, n_jobs=-1).mean()
    return cv

# Hyperparamter definieren
space = {'n_estimators': hp.choice('n_estimators', range(250, 3001, 250)),
         'max_depth': hp.choice('max_depth', [None, 5, 10, 25, 50, 100, 200]),
         'max_features': hp.choice('max_features', (0.01, 0.05, 0.5, 'log2', 'sqrt', 'auto')),
         'min_samples_leaf': hp.choice('min_samples_leaf', (1, 3, 9, 15)),
         'min_samples_split': hp.choice('min_samples_split', (0.5, 2, 5, 10, 15, 100)),   
         'max_leaf_nodes': hp.choice('max_leaf_nodes', [None, 2, 5, 10]),
         'criterion': hp.choice('criterion', ["gini", "entropy"])
        }

best = 0
def fn(params):
    """
    Berechnung des roc_auc durch aufrufen der Funktion `hyperopt_train_test` und durch übergeben des Parameters `params`
    Extrahieren des höchsten roc_auc 
    """
    global best
    roc_auc = hyperopt_train_test(params)
    if roc_auc > best:
        best = roc_auc
    print(f'best: {best}\nparams: {params}')
    # Da wir den roc_auc maxmieren und in der Funktion `fmin` minimieren, müssen wir diesen als negativen Wert zurückgeben
    return {'loss': -roc_auc, 'status': STATUS_OK}

# Zum speichern der einzelen Ergebnisse der Funktion `fn` instanzieren wir das Klassenobjekt `trials`
trials = Trials()

# Die Funktion `fmin` erstellt mittels des TPE-EI Algorithmus die besten Hyperparamterkombinationen und übergibt diese in der 
# Funktion `fn`. Insgesamt testen wir 50 mögliche Kombinationen an Hyperparamtern. Die Ergebnisse werden im Objekt `trials` gespeichert. 
# Das Ergebnis der Minimierung ist ein dictionary mit den besten Hyperparamtern, also die Hyperparameter, bei denen der
# roc_auc score am Höchsten war
best = fmin(fn,
            space,
            algo=tpe.suggest,
            max_evals=20, 
            trials=trials
            )

print('best:', best)


# In[43]:


# Anhand der Indices der Hyperparamter in der Variable `best` filtert die Funktion `space_eval` die entsprechenden Hyperparamter aus `space`
params_random_forest_opt = space_eval(space, best)

# Speichern der Hyperparamter als JSON Datei
with open('../data/random_forest_hyper_params.json', mode='x') as f:
    json.dump(params_random_forest_opt, fp=f)

params_random_forest_opt


# Trainieren des finalen Modelles

# In[44]:


clf = RandomForestClassifier(**params_random_forest_opt, n_jobs=-1)

fit_rf = clf.fit(X_train, y_train['response'])

# Wahrscheinlichkeit für die Vorhersage der positiven Klasse der Zielvariable
y_pred_proba = fit_rf.predict_proba(X_test)[:,1]
# Vorhersagen der Zielvariable 
y_pred = fit_rf.predict(X_test)

print(y_pred)
print(y_pred_proba)


# Funktion zur Berechnung des pr_auc

# In[45]:


pr_auc_rf = get_auc_pr(y_test[['response']], y_pred)
pr_auc_rf


# ### Evaluation 

# #### Entscheidung für ein Gütemaß
# 
# Accuracy:  Die Accuracy wird anhand der vorhergesagten Klassen berechnet. Das bedeutet, dass sie auch den verwendeten Threshold beinhaltet, der zunächst noch optimiert werden muss. 
# 
# ROC-AUC und PR-AUC:
# Der roc_auc und der pr_auc betrachten beide die Vorhersagewerte von Klassifizierungsmodellen und nicht die Klassenzuordnungen mit Schwellwerten. Der Unterschied besteht jedoch darin, dass roc_auc den Anteil der wahren Positiven (TPR) und den Anteil der falschen Positiven (FPR) berücksichtigt, während der pr_auc den positiven Vorhersagewert (PPV) und den Anteil der wahren Positiven (TPR) berücksichtigt. Da wir in unserem Fall mehr Fokus auf die positive Klasse legen, ist die Verwendung des pr_auc, der empfindlicher auf Verbesserungen für die positive Klasse reagiert, die bessere Wahl.
# 
# Darum werden wir als maßgebliches Gütemaß den pr_auc in diesem Anwendungsfall verwenden.

# #### Visuelle Darstellung der Kunfusionsmatrix

# In[46]:


# Konfusionsmatrix erstellen
cf_matrix = confusion_matrix(y_test[['response']], y_pred)

# Klassennamen definieren
group_names = ['True Neg','False Pos','False Neg','True Pos']

# Ausprägungen je Klasse zählen
group_counts = ["{0:0.0f}".format(value) for value in
                cf_matrix.flatten()]

# relativen Anteile der Klasse bestimmen
group_percentages = ["{0:.2%}".format(value) for value in cf_matrix.flatten()/np.sum(cf_matrix)]

# Die obigen Informationen zusammenführen 
labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in
          zip(group_names,group_counts,group_percentages)]

labels = np.asarray(labels).reshape(2,2)

# Plot erstellen
sns.heatmap(cf_matrix, annot=labels, fmt='', cmap='Blues')


# #### Feature Importance 

# In[47]:


# Feature Importance für den Trainingsdatensatz plotten

#Feature Importance in einem Data Frame ablegen und absteigend nach den höchsten Werten sortieren 
feature_importances=pd.DataFrame({'features':X_train.columns,'feature_importance':fit_rf.feature_importances_})
feature_importances.sort_values('feature_importance',ascending=False)

# Bar Chart erstellen 
fig = go.Figure(go.Bar(
            x=feature_importances['feature_importance'].sort_values(ascending=True),
            y=X_train.columns,
            text=round(feature_importances['feature_importance'].sort_values(ascending=True), 4),
            marker_color= 'rgb(18,116,117)',
            orientation='h'))

fig.update_traces(textposition='outside')
fig.update_layout(title_text = 'Feature Importance')
fig.show()


# #### ROC Kurve

# In[48]:


# False Positive, True Postive Rate bestimmen
fpr, tpr, _ = roc_curve(y_test['response'], y_pred_proba)

# Fläche unter der False Positive, True Postive Rate berechnen
auc_score = roc_auc_score(y_test['response'], y_pred_proba)

# Area Plot erstellen
fig = px.area(
    x=fpr, y=tpr,
    title=f'ROC-Curve (AUC={auc(fpr, tpr):.4f})',
    labels=dict(x='False Positive Rate', y='True Positive Rate'),
    width=700, height=500
)

# Linie des Zufallsmodelles einfügen
fig.add_shape(
    type='line', line=dict(dash='dash'),
    x0=0, x1=1, y0=0, y1=1
)

# Skalierung anpassen
fig.update_yaxes(scaleanchor="x", scaleratio=1)
fig.update_xaxes(constrain='domain')

# Plot anzeigen
fig.show()


# #### Precision-Recall-Curve

# Der `Recall` sagt also etwas darüber aus, wieviele der in der Datenbank vorhandenen relevanten Dokumente gefunden wurden – ins Verhältnis gesetzt zur Anzahl aller relevanten Dokumente in der Datenbank. 
# 
# Die `Precision` setzt jene Zahl ins Verhältnis zur Zahl der insgesamt gefundenen Dokumente, sie gibt an, wieviele der gefundenen relevant sind. Grob gesprochen: Recall – wieviel habe ich gefunden, wieviel Substanz hat die Datenbank ; Precision – wieviel Unbrauchbares habe ich gefunden, wie genau kann man in der Datenbank suchen?

# In[49]:


# Precision, Recall und Thresholds berechnen
precision_recall_threshold = pd.DataFrame(precision_recall_curve(y_test[['response']], y_pred_proba)).transpose()

# Spaltennamen anpassen
precision_recall_threshold.columns = ['Precision', 'Recall', 'Threshold']

# Area Plot erstellen
fig = px.area(
    x=precision_recall_threshold['Recall'], y=precision_recall_threshold['Precision'],
    title=f'Precision-Recall Curve (AUC={pr_auc_rf:.4f})',
    labels=dict(x='Recall', y='Precision'),
    width=700, height=500
)

# Linie des Zufallsmodells einfügen
fig.add_shape(
    type='line', line=dict(dash='dash'),
    x0=0, x1=1, y0=1, y1=0
)

# Skalierung anpassen
fig.update_yaxes(scaleanchor="x", scaleratio=1)
fig.update_xaxes(constrain='domain')

# Plot anzeigen
fig.show()


# #### Precision-Recall-Threshold Curve

# In[50]:


# Data Frame erzeugen, der die precision, recall, threshold werte enthält 
precision_recall_threshold = np.transpose(pd.DataFrame(precision_recall_curve(y_test[['response']], y_pred_proba)))
precision_recall_threshold.columns = ['Precision', 'Recall', 'Threshold']

# Schnittpunkt von Precision und Threshold berechnen und den dazugehörigen Threshold in einem Data Frame ablegen 
precision_recall_threshold['distance']  =  abs(precision_recall_threshold['Precision'] - precision_recall_threshold['Recall'])
intersection_precision_recall_threshold = precision_recall_threshold.loc[precision_recall_threshold['distance'] == precision_recall_threshold['distance'].min()]
intersection = intersection_precision_recall_threshold['Threshold']


# Plot erzeugen 
fig = go.Figure()

# Trace für die Precison und die dazugehörigen Thresholds
fig.add_trace(go.Scatter(x=precision_recall_threshold['Threshold'], y=precision_recall_threshold['Precision'],
                    fill='tozeroy', fillcolor='rgba(248, 118, 109, 0.5)', opacity=0.5,
                    hoveron = 'points+fills', # select where hover is active
                    line_color='rgba(248, 118, 109, 0.5)',
                    name= 'Precision',
                    hoverinfo = 'text+x+y'))


# Trace für den Recall und die dazugehörigen Thresholds
fig.add_trace(go.Scatter(x=precision_recall_threshold['Threshold'], y=precision_recall_threshold['Recall'],
                    fill='tozeroy', fillcolor = 'rgba(0, 191, 196, 0.5)',
                    hoveron='points',
                    line_color='rgba(0, 191, 196, 0.5)',
                    name= 'Recall',
                    hoverinfo='text+x+y')) 


# Schnittpunkt von Precision und Recall einfügen
fig.add_traces(go.Scatter(
    x= [intersection.values[0],intersection.values[0]],
    y=[0,1],
    mode='lines',
    name=f'threshold {intersection.values[0]:.4f}',
    line={'dash': 'dash', 
          'color': 'silver',
          'width': 2}
    ))


# Layout anpoassen
fig.update_layout(title = f'Precision-Recall-Threshold Curve (AUC={pr_auc_rf:.4f})')
fig.update_layout(hovermode='x unified')
fig.update_xaxes(title_text='Threshold')
fig.update_yaxes(title_text='Precision / Recall')

fig.show()


# #### Sensitivity-Specificity-Threshold Curve

# In[51]:


fpr, tpr, thresholds =roc_curve(y_test[['response']], y_pred_proba)
sensitivity_specificity_threshold = pd.DataFrame({'Sensitivity' : tpr,'Specificitiy' : 1 - fpr, 'Threshold': thresholds})

# Es war ein Threshold mit dem Wert von 1,7 drin - Den habe ich gelöscht, da der plot sonst extrem verzerrt wird
sensitivity_specificity_threshold = sensitivity_specificity_threshold.iloc[1: , :]

# Schnittpunkt von Precision und Threshold berechnen und den dazugehörigen Threshold in einem Data Frame ablegen 
sensitivity_specificity_threshold['distance']  =  abs(sensitivity_specificity_threshold['Sensitivity'] - sensitivity_specificity_threshold['Specificitiy'])
intersection_specificity_threshold = sensitivity_specificity_threshold.loc[sensitivity_specificity_threshold['distance'] == sensitivity_specificity_threshold['distance'].min()]
intersection_spt = intersection_specificity_threshold['Threshold']


# Plot erzeugen 
fig = go.Figure()

# Trace für die Precison und die dazugehörigen Thresholds
fig.add_trace(go.Scatter(x=sensitivity_specificity_threshold['Threshold'], y=sensitivity_specificity_threshold['Sensitivity'],
                    fill='tozeroy', fillcolor='rgba(248, 118, 109, 0.5)', opacity=0.5,
                    hoveron = 'points+fills', # select where hover is active
                    line_color='rgba(248, 118, 109, 0.5)',
                    name= 'Sensitivity',
                    hoverinfo = 'text+x+y'))


# Trace für den Recall und die dazugehörigen Thresholds
fig.add_trace(go.Scatter(x=sensitivity_specificity_threshold['Threshold'], y=sensitivity_specificity_threshold['Specificitiy'],
                    fill='tozeroy', fillcolor = 'rgba(0, 191, 196, 0.5)',
                    hoveron='points',
                    line_color='rgba(0, 191, 196, 0.5)',
                    name= 'Specificitiy',
                    hoverinfo='text+x+y')) 


# Schnittpunkt von Precision und Recall einfügen
fig.add_traces(go.Scatter(
    x= [intersection_spt.values[0],intersection_spt.values[0]],
    y=[0,1],
    mode='lines',
    name=f'threshold {intersection_spt.values[0]:.4f}',
    line={'dash': 'dash', 
          'color': 'silver',
          'width': 2}
    ))


# Layout anpoassen
fig.update_layout(title = 'Sensitivity-Specificity-Threshold Curve')
fig.update_layout(hovermode='x unified')
fig.update_xaxes(title_text='Threshold')
fig.update_yaxes(title_text='Sensitivity / Specificity')

fig.show()


# #### Cumulative Gain Chart

# In[52]:


kds.metrics.report(np.array(y_test['response']), y_pred_proba,plot_style='ggplot')


# - Lift Plot: Unser Modell sagt bis zur einer Grunggesamtheit von 100% (10tes Dezil) die positive Klasse besser vorher, als ein Zufallsmodell.
# - Cumulative Gain Plot: Wenn wir ~ 70% der von unserem Modell positiv vorhergesagten Klasse erreicht habe, habe ich bereits fast 100% der positiven Klasse erreicht. 

# ## Gradiant Boosting Classifier

# Default Values for Gradient Boosting Classifier:
# 
# * `n_estimators`: 100
# * `loss`: 'deviance'
# * `learning_rate`: 0.1
# * `subsample`: 1
# * `criterion`= "friedman_mse"
# * `min_samples_split`: 2
# * `min_samples_leaf`: 1
# * `min_weight_frachtion`: 0
# * `max_features`: None
# * `max_depth`: 3

# In[15]:


def hyperopt_boosted_tree_train_test(params):
    """
    Instanz für den Gradient Boosting Classifier erstellen 
    Berechnung des Mittelwertes bei 5-facher Crossvalidierung auf Basis des roc_auc scorings
    """
    gbc = GradientBoostingClassifier(**params, random_state=42)

    kfold = StratifiedKFold(n_splits=5)

    cv = cross_val_score(gbc, X_train, y_train['response'], scoring='roc_auc', cv=kfold, n_jobs=-1).mean()
    return cv
    
# Hyperparamter definieren
space_bt = {'loss': hp.choice('loss', ['deviance', 'exponential']),
            'learning_rate': hp.choice('learing_rate', (0.01, 0.05, 0.1, 0.2, 0.25, 0.5, 1)),
            'min_samples_split': hp.choice('min_samples_split', (0.1, 0.5, 2, 5, 10)),
            'min_samples_leaf': hp.choice('min_samples_leaf', (0.1, 0.5, 1, 2, 5, 12)),
            'criterion': hp.choice('criterion', ('friedman_mse', 'squared_error')),
            'max_features': hp.choice('max_features', ('log2', 'sqrt', 'auto')),
            'subsample': hp.choice('subsample', (0.5, 0.65, 0.85, 0.9, 1)),
            'n_estimators': hp.choice('n_estimators', range(250, 3001, 250))
            }

best_bt = 0

def fn(params):
    """
    Berechnung des roc_auc durch aufrufen der Funktion `hyperopt_train_test` und durch übergeben des Parameters `params`
    Extrahieren des höchsten roc_auc 

    """
    global best_bt
    roc_auc = hyperopt_boosted_tree_train_test(params)
    if roc_auc > best_bt:
        best_bt = roc_auc
    print('new best:', best_bt, params)
    # Da wir den roc_auc maxmieren und in der Funktion `fmin` minimieren, müssen wir diesen als negativen Wert zurückgeben
    return {'loss': -roc_auc, 'status': STATUS_OK}

# Zum speichern der einzelen Ergebnisse der Funktion `fn` instanzieren wir das Klassenobjekt `trials`
trials = Trials()

# Die Funktion `fmin` erstellt mittels des TPE-EI Algorithmus die besten Hyperparamterkombinationen und übergibt diese in der 
# Funktion `fn`. Insgesamt testen wir 50 mögliche Kombinationen an Hyperparamtern. Die Ergebnisse werden im Objekt `trials` gespeichert. 
# Das Ergebnis der Minimierung ist ein dictionary mit den besten Hyperparamtern, also die Hyperparameter, bei denen der
# roc_auc score am Höchsten war
best_bt = fmin(fn,
            space_bt,
            algo=tpe.suggest,
            max_evals=20, 
            trials=trials
            )
            
print('best_bt:', best_bt)


# In[16]:


# Anhand der Indices der Hyperparamter in der Variable `best_bt` filtert die Funktion `space_eval` die entsprechenden Hyperparamter aus `space_bt`
params_boosted_tree_opt = space_eval(space_bt, best_bt)

# Speichern der Hyperparamter als JSON Datei
with open('../data/gradient_boosting_classifier_hyper_params.json', mode='x') as f:
    json.dump(params_boosted_tree_opt, fp=f)

params_boosted_tree_opt


# Trainieren des finalen Modelles

# In[17]:


gbc = GradientBoostingClassifier(**params_boosted_tree_opt, random_state=42)

fit_bt = gbc.fit(X_train, y_train['response'])
y_pred_proba_bt = fit_bt.predict_proba(X_test)[:,1]
y_pred_bt = fit_bt.predict(X_test)

print(y_pred_bt)
print(y_pred_proba_bt)


# In[26]:


pr_auc_bt = get_auc_pr(y_test[['response']], y_pred_bt)
pr_auc_bt


# #### Evaluation Gradient Boosting Classifier

# In[19]:


# Konfusionsmatrix erstellen
cf_matrix = confusion_matrix(y_test[['response']], y_pred_bt)

# Klassennamen definieren
group_names = ['True Neg','False Pos','False Neg','True Pos']

# Ausprägungen je Klasse zählen
group_counts = ["{0:0.0f}".format(value) for value in
                cf_matrix.flatten()]

# relativen Anteile der Klasse bestimmen
group_percentages = ["{0:.2%}".format(value) for value in cf_matrix.flatten()/np.sum(cf_matrix)]

# Die obigen Informationen zusammenführen 
labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in
          zip(group_names,group_counts,group_percentages)]

labels = np.asarray(labels).reshape(2,2)

# Plot erstellen
sns.heatmap(cf_matrix, annot=labels, fmt='', cmap='Blues')


# In[20]:


# Feature Importance für den Trainingsdatensatz plotten
#Feature Importance in einem Data Frame ablegen und absteigend nach den höchsten Werten sortieren 
feature_importances=pd.DataFrame({'features':X_train.columns,'feature_importance':fit_bt.feature_importances_})
feature_importances.sort_values('feature_importance',ascending=False)

# Bar Chart erstellen 
fig = go.Figure(go.Bar(
            x=feature_importances['feature_importance'].sort_values(ascending=True),
            y=X_train.columns,
            text=round(feature_importances['feature_importance'].sort_values(ascending=True), 4),
            marker_color= 'rgb(18,116,117)',
            orientation='h'))

fig.update_traces(textposition='outside')
fig.update_layout(title_text = 'Feature Importance')
fig.show()


# #### ROC-Curve

# In[21]:


# False Positive, True Postive Rate bestimmen
fpr, tpr, _ = roc_curve(y_test['response'], y_pred_proba_bt)

# Fläche unter der False Positive, True Postive Rate berechnen
auc_score = roc_auc_score(y_test['response'], y_pred_proba_bt)

# Plot Design erstllen
fig = px.area(
    x=fpr, y=tpr,
    title=f'ROC-Curve (AUC={auc(fpr, tpr):.4f})',
    labels=dict(x='False Positive Rate', y='True Positive Rate'),
    width=700, height=500
)

# Linie des Zufallsmodelles einfügen
fig.add_shape(
    type='line', line=dict(dash='dash'),
    x0=0, x1=1, y0=0, y1=1
)

# Skalierung anpassen
fig.update_yaxes(scaleanchor="x", scaleratio=1)
fig.update_xaxes(constrain='domain')

# Plot erstellen
fig.show()


# #### Precision-Recall Curve

# In[22]:


# Precision, Recall und Thresholds berechnen
precision_recall_threshold = pd.DataFrame(precision_recall_curve(y_test[['response']], y_pred_proba_bt)).transpose()

# Spaltennamen anpassen
precision_recall_threshold.columns = ['Precision', 'Recall', 'Threshold']

# Area Plot erstellen
fig = px.area(
    x=precision_recall_threshold['Recall'], y=precision_recall_threshold['Precision'],
    title=f'Precision-Recall Curve (AUC={pr_auc_bt:.4f})',
    labels=dict(x='Recall', y='Precision'),
    width=700, height=500
)

# Linie des Zufallsmodells einfügen
fig.add_shape(
    type='line', line=dict(dash='dash'),
    x0=0, x1=1, y0=1, y1=0
)

# Skalierung anpassen
fig.update_yaxes(scaleanchor="x", scaleratio=1)
fig.update_xaxes(constrain='domain')

# Plot anzeigen
fig.show()


# In[23]:


kds.metrics.report(np.array(y_test['response']), y_pred_proba_bt, plot_style='ggplot')


# - Lift Plot: Unser Modell sagt bis zur einer Grunggesamtheit von 100% (10tes Dezil) die positive Klasse besser vorher, als ein Zufallsmodell.
# - Cumulative Gain Plot: Wenn wir ~ 70% der von unserem Modell positiv vorhergesagten Klasse erreicht habe, habe ich bereits fast 100% der positiven Klasse erreicht. 

# #### Precision-Recall-Threshold Curve

# In[24]:


# Data Frame erzeugen, der die precision, recall, threshold werte enthält 
precision_recall_threshold_bt = np.transpose(pd.DataFrame(precision_recall_curve(y_test[['response']], y_pred_proba_bt)))
precision_recall_threshold_bt.columns = ['Precision', 'Recall', 'Threshold']

# Schnittpunkt von Precision und Threshold berechnen und den dazugehörigen Threshold in einem Data Frame ablegen 
precision_recall_threshold_bt['distance']  =  abs(precision_recall_threshold_bt['Precision'] - precision_recall_threshold_bt['Recall'])
intersection_precision_recall_threshold_bt = precision_recall_threshold_bt.loc[precision_recall_threshold_bt['distance'] == precision_recall_threshold_bt['distance'].min()]
intersection_bt = intersection_precision_recall_threshold_bt['Threshold']


# Plot erzeugen 
fig = go.Figure()

# Trace für die Precison und die dazugehörigen Thresholds
fig.add_trace(go.Scatter(x=precision_recall_threshold_bt['Threshold'], y=precision_recall_threshold_bt['Precision'],
                    fill='tozeroy', fillcolor='rgba(248, 118, 109, 0.5)', opacity=0.5,
                    hoveron = 'points+fills', # select where hover is active
                    line_color='rgba(248, 118, 109, 0.5)',
                    name= 'Precision',
                    hoverinfo = 'text+x+y'))


# Trace für den Recall und die dazugehörigen Thresholds
fig.add_trace(go.Scatter(x=precision_recall_threshold_bt['Threshold'], y=precision_recall_threshold_bt['Recall'],
                    fill='tozeroy', fillcolor = 'rgba(0, 191, 196, 0.5)',
                    hoveron='points',
                    line_color='rgba(0, 191, 196, 0.5)',
                    name= 'Recall',
                    hoverinfo='text+x+y')) 


# Schnittpunkt von Precision und Recall einfügen
fig.add_traces(go.Scatter(
    x= [intersection_bt.values[0],intersection_bt.values[0]],
    y=[0,1],
    mode='lines',
    name=f'threshold {intersection_bt.values[0]:.4f}',
    line={'dash': 'dash', 
          'color': 'silver',
          'width': 2}
    ))


# Layout anpoassen
fig.update_layout(title = f'Precision-Recall-Threshold Curve (pr_AUC={pr_auc_bt:.4f})')
fig.update_layout(hovermode='x unified')
fig.update_xaxes(title_text='Threshold')
fig.update_yaxes(title_text='Precision / Recall')

fig.show()


# #### Sensitivity-Specificity-Threshold Curve

# In[25]:


fpr, tpr, thresholds =roc_curve(y_test[['response']], y_pred_proba_bt)
sensitivity_specificity_threshold_bt = pd.DataFrame({'Sensitivity' : tpr,'Specificity' : 1 - fpr, 'Threshold': thresholds})

# Es war ein Threshold mit dem Wert von 1,7 drin - Den habe ich gelöscht, da der plot sonst extrem verzerrt wird
sensitivity_specificity_threshold_bt = sensitivity_specificity_threshold_bt.iloc[1: , :]

# Schnittpunkt von Precision und Threshold berechnen und den dazugehörigen Threshold in einem Data Frame ablegen 
sensitivity_specificity_threshold_bt['distance']  =  abs(sensitivity_specificity_threshold_bt['Sensitivity'] - sensitivity_specificity_threshold_bt['Specificity'])
intersection_specificity_threshold = sensitivity_specificity_threshold_bt.loc[sensitivity_specificity_threshold_bt['distance'] == sensitivity_specificity_threshold_bt['distance'].min()]
intersection_spt_bt = intersection_specificity_threshold['Threshold']


# Plot erzeugen 
fig = go.Figure()

# Trace für die Precison und die dazugehörigen Thresholds
fig.add_trace(go.Scatter(x=sensitivity_specificity_threshold_bt['Threshold'], y=sensitivity_specificity_threshold_bt['Sensitivity'],
                    fill='tozeroy', fillcolor='rgba(248, 118, 109, 0.5)', opacity=0.5,
                    hoveron = 'points+fills', # select where hover is active
                    line_color='rgba(248, 118, 109, 0.5)',
                    name= 'Sensitivity',
                    hoverinfo = 'text+x+y'))


# Trace für den Recall und die dazugehörigen Thresholds
fig.add_trace(go.Scatter(x=sensitivity_specificity_threshold_bt['Threshold'], y=sensitivity_specificity_threshold_bt['Specificity'],
                    fill='tozeroy', fillcolor = 'rgba(0, 191, 196, 0.5)',
                    hoveron='points',
                    line_color='rgba(0, 191, 196, 0.5)',
                    name= 'Specificity',
                    hoverinfo='text+x+y')) 


# Schnittpunkt von Precision und Recall einfügen
fig.add_traces(go.Scatter(
    x= [intersection_spt_bt.values[0],intersection_spt_bt.values[0]],
    y=[0,1],
    mode='lines',
    name=f'threshold {intersection_spt_bt.values[0]:.4f}',
    line={'dash': 'dash', 
          'color': 'silver',
          'width': 2}
    ))


# Layout anpoassen
fig.update_layout(title = 'Sensitivity-Specificity-Threshold Curve')
fig.update_layout(hovermode='x unified')
fig.update_xaxes(title_text='Threshold')
fig.update_yaxes(title_text='Sensitivity / Specificity')

fig.show()


# ## Neuronal Network 

# In[29]:


def nn_cl_bo(neurons, activation, optimizer, learning_rate,  batch_size, epochs ):
    """
    Funktion zur Bestimmung der Hyperparamter des Neuronalen Netzes.
    Die Hyperparamter werden auf Basis des zufällig bestimmten Index ausgewählt und auf einem Standardmodell trainiert.
    Mittels der Crossvalidierung geben wir das beste Ergebnis des ersten Tunings zurück
    """
    optimizerL = ['SGD', 'Adam', 'RMSprop', 'Adadelta', 'Adagrad', 'Adamax', 'Nadam', 'Ftrl','SGD']
    optimizerD = {'Adam':Adam(lr=learning_rate), 'SGD':SGD(lr=learning_rate),
                 'RMSprop':RMSprop(lr=learning_rate), 'Adadelta':Adadelta(lr=learning_rate),
                 'Adagrad':Adagrad(lr=learning_rate), 'Adamax':Adamax(lr=learning_rate),
                 'Nadam':Nadam(lr=learning_rate), 'Ftrl':Ftrl(lr=learning_rate)}
    activationL = ['relu', 'sigmoid', 'softplus', 'softsign', 'tanh', 'selu',
                   'elu', 'exponential', LeakyReLU,'relu']
    neurons = round(neurons)
    optimizer = optimizerL[round(optimizer)]
    activation = activationL[round(activation)]
    batch_size = round(batch_size)
    epochs = round(epochs)

    def nn_cl_fun():
        opt = Adam(lr = learning_rate)
        nn = Sequential()
        nn.add(Dense(neurons, input_dim=16, activation=activation))
        nn.add(Dense(neurons, activation=activation))
        nn.add(Dense(1, activation='sigmoid'))
        nn.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
        return nn
        
    es = EarlyStopping(monitor='accuracy', mode='max', verbose=0, patience=20)
    nn = KerasClassifier(build_fn=nn_cl_fun, epochs=epochs, batch_size=batch_size,
                         verbose=0)
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=123)
    score = cross_val_score(nn, X_train, y_train['response'], scoring=make_scorer(accuracy_score), cv=kfold, fit_params={'callbacks':[es]}).mean()
    return score


# In[30]:


# Hyperparameter definieren
params_nn ={
    'neurons': (5, 10),
    'activation':(0, 9),
    'optimizer':(0,7),
    'learning_rate':(0.01, 1),
    'batch_size':(20, 50),
    'epochs':(10, 50)
}

# Bayesian Optimierung durchführen
nn_bo = BayesianOptimization(nn_cl_bo, params_nn, random_state=111)

# Filtere nach dem höchsten target score
nn_bo.maximize(init_points=25, n_iter=4)


# Die besten hyperparameter ausgeben lassen 

# In[31]:


params_nn_ = nn_bo.max['params']
activationL = ['relu', 'sigmoid', 'softplus', 'softsign', 'tanh', 'selu',
               'elu', 'exponential', LeakyReLU,'relu']
params_nn_['activation'] = activationL[round(params_nn_['activation'])]
params_nn_


# Erstellen einer Funktion zum Tunen des Modells

# In[32]:


def nn_cl_bo2(neurons, activation, optimizer, learning_rate, batch_size, epochs,
              layers1, layers2, normalization, dropout, dropout_rate):
    optimizerL = ['SGD', 'Adam', 'RMSprop', 'Adadelta', 'Adagrad', 'Adamax', 'Nadam', 'Ftrl','SGD']
    optimizerD= {'Adam':Adam(lr=learning_rate), 'SGD':SGD(lr=learning_rate),
                 'RMSprop':RMSprop(lr=learning_rate), 'Adadelta':Adadelta(lr=learning_rate),
                 'Adagrad':Adagrad(lr=learning_rate), 'Adamax':Adamax(lr=learning_rate),
                 'Nadam':Nadam(lr=learning_rate), 'Ftrl':Ftrl(lr=learning_rate)}
    activationL = ['relu', 'sigmoid', 'softplus', 'softsign', 'tanh', 'selu',
                   'elu', 'exponential', LeakyReLU,'relu']
    neurons = round(neurons)
    activation = activationL[round(activation)]
    optimizer = optimizerD[optimizerL[round(optimizer)]]
    batch_size = round(batch_size)
    epochs = round(epochs)
    layers1 = round(layers1)
    layers2 = round(layers2)
    def nn_cl_fun():
        nn = Sequential()
        nn.add(Dense(neurons, activation=activation))
        if normalization > 0.5:
            nn.add(BatchNormalization())
        for i in range(layers1):
            nn.add(Dense(neurons, activation=activation))
        if dropout > 0.5:
            nn.add(Dropout(dropout_rate, seed=123))
        for i in range(layers2):
            nn.add(Dense(neurons, activation=activation))
        nn.add(Dense(1, activation='sigmoid'))
        nn.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        return nn
    es = EarlyStopping(monitor='accuracy', mode='max', verbose=0, patience=20)
    nn = KerasClassifier(build_fn=nn_cl_fun, epochs=epochs, batch_size=batch_size, verbose=0)
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=123)
    score = cross_val_score(nn, X_train, y_train['response'], scoring=make_scorer(accuracy_score), cv=kfold, fit_params={'callbacks':[es]}).mean()
    return score


# Im Folgenden wird nun nach den optimalen Hyperparametern und Layern gesucht 

# In[33]:


params_nn2 ={
    'neurons': (16, 100),
    'activation':(0, 9),
    'optimizer':(0,7),
    'learning_rate':(0.01, 1),
    'batch_size':(200, 1000),
    'epochs':(20, 100),
    'layers1':(1,3),
    'layers2':(1,3),
    'normalization':(0,1),
    'dropout':(0,1),
    'dropout_rate':(0,0.3)
}

# Bayesian Optimierung durchführen
nn_bo = BayesianOptimization(nn_cl_bo2, params_nn2, random_state=111)

# Filtere nach dem höchsten target score
nn_bo.maximize(init_points=25, n_iter=4)


# Die besten Hyperparamter und Layer ausgeben lassen

# In[34]:


params_nn_ = nn_bo.max['params']
learning_rate = params_nn_['learning_rate']
activationL = ['relu', 'sigmoid', 'softplus', 'softsign', 'tanh', 'selu',
               'elu', 'exponential', LeakyReLU,'relu']
params_nn_['activation'] = activationL[round(params_nn_['activation'])]
params_nn_['batch_size'] = round(params_nn_['batch_size'])
params_nn_['epochs'] = round(params_nn_['epochs'])
params_nn_['layers1'] = round(params_nn_['layers1'])
params_nn_['layers2'] = round(params_nn_['layers2'])
params_nn_['neurons'] = round(params_nn_['neurons'])
optimizerL = ['Adam', 'SGD', 'RMSprop', 'Adadelta', 'Adagrad', 'Adamax', 'Nadam', 'Ftrl','Adam']
optimizerD= {'Adam':Adam(lr=learning_rate), 'SGD':SGD(lr=learning_rate),
             'RMSprop':RMSprop(lr=learning_rate), 'Adadelta':Adadelta(lr=learning_rate),
             'Adagrad':Adagrad(lr=learning_rate), 'Adamax':Adamax(lr=learning_rate),
             'Nadam':Nadam(lr=learning_rate), 'Ftrl':Ftrl(lr=learning_rate)}
params_nn_['optimizer'] = optimizerD[optimizerL[round(params_nn_['optimizer'])]]
params_nn_


# Finales Model trainieren

# In[35]:


def nn_cl_fun():
    nn = Sequential()
    nn.add(Dense(params_nn_['neurons'], input_dim=16, activation=params_nn_['activation']))
    if params_nn_['normalization'] > 0.5:
        nn.add(BatchNormalization())
    for i in range(params_nn_['layers1']):
        nn.add(Dense(params_nn_['neurons'], activation=params_nn_['activation']))
    if params_nn_['dropout'] > 0.5:
        nn.add(Dropout(params_nn_['dropout_rate'], seed=42))
    for i in range(params_nn_['layers2']):
        nn.add(Dense(params_nn_['neurons'], activation=params_nn_['activation']))
    nn.add(Dense(1, activation='sigmoid'))
    nn.compile(loss='binary_crossentropy', optimizer=params_nn_['optimizer'], metrics=['accuracy'])
    return nn
        
es = EarlyStopping(monitor='accuracy', mode='max', verbose=0, patience=20)

nn = KerasClassifier(build_fn=nn_cl_fun, epochs=params_nn_['epochs'], batch_size=params_nn_['batch_size'],verbose=0)

standart_scaler = StandardScaler()
X_train[['annual_premium', 'vintage']] = standart_scaler.fit_transform(X_train[['annual_premium', 'vintage']])
X_test[['annual_premium', 'vintage']] = standart_scaler.fit_transform(X_test[['annual_premium', 'vintage']])
 
nn_final_model = nn.fit(X_train, y_train, validation_data=(X_test, y_test), verbose=1)


# Evaluation des Modells auf dem Testdatensatz

# In[36]:


y_pred_nn = pd.DataFrame(nn.predict(X_test), columns=['pred_response_nn'])
y_pred_nn


# Ausgeben der Vorhersagen

# In[37]:


y_pred_nn_proba = nn.predict_proba(X_test)[:,1]
y_pred_nn_proba


# Berechnung des pr auc

# In[38]:


pr_auc_nn = get_auc_pr(y_test['response'], y_pred_nn_proba)


# #### Evaluation des Neuronal Network 

# In[39]:


# Konfusionsmatrix
cf_matrix_bt = confusion_matrix(y_test[['response']], y_pred_nn)

group_names_bt = ['True Neg','False Pos','False Neg','True Pos']

group_counts_bt = ["{0:0.0f}".format(value) for value in
                cf_matrix_bt.flatten()]

group_percentages_bt = ["{0:.2%}".format(value) for value in cf_matrix_bt.flatten()/np.sum(cf_matrix_bt)]


labels_bt = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in
          zip(group_names_bt,group_counts_bt,group_percentages_bt)]

labels_bt = np.asarray(labels_bt).reshape(2,2)

sns.heatmap(cf_matrix_bt, annot=labels_bt, fmt='', cmap='Blues')


# #### ROC-Curve

# In[40]:


# False Positive, True Postive Rate bestimmen
fpr, tpr, _ = roc_curve(y_test['response'], y_pred_nn_proba)

# Fläche unter der False Positive, True Postive Rate berechnen
auc_score = roc_auc_score(y_test['response'], y_pred_nn_proba)

# Plot Design erstllen
fig = px.area(
    x=fpr, y=tpr,
    title=f'ROC-Curve (AUC={auc(fpr, tpr):.4f})',
    labels=dict(x='False Positive Rate', y='True Positive Rate'),
    width=700, height=500
)

# Linie des Zufallsmodelles einfügen
fig.add_shape(
    type='line', line=dict(dash='dash'),
    x0=0, x1=1, y0=0, y1=1
)

# Skalierung anpassen
fig.update_yaxes(scaleanchor="x", scaleratio=1)
fig.update_xaxes(constrain='domain')

# Plot erstellen
fig.show()


# ## Vergleich aller Modelle

# #### Modell vergleiche anhand eines Gütemaßes in einer Tabelle 

# In[53]:


# Modell Evaluation
# Hier können wir dann alle Modell anhand von verschiedenen Gütemaßen in einer Tabelle vergleichen 
models = pd.DataFrame({
    'Model': ['Random Forest Classifier', 'Gradient Boosting Classifier', 'Neural Network'],
    'Score': [pr_auc_rf, pr_auc_bt, pr_auc_nn]
})

# Die cmpa wird erst bei mehreren Zeilen richtig angezeigt
models.sort_values(by='Score', ascending=False).style.background_gradient(cmap='Greens',subset = ['Score'])   


# Da der AUC vom Precision Recall beim `Gradient Boosting Classifier` am Höchsten ist, entscheiden wir uns für dieses Modell für die Vorhersage der Zielvariable im Zuge des Deployments. 
