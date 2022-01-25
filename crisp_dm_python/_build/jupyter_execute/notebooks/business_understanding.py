#!/usr/bin/env python
# coding: utf-8

# # Business Understanding

# ## Ausgangssituation
# 
# Wir arbeiten für ein Beratungsunternehmen welches datengetriebene Lösungen für betriebswirtschaftliche Problemstellungen entwickelt. Unser Unternehmen wurde von einer Versicherungsgesellschaft beauftragt, ein Modell zu entwickeln um vorherzusagen ob ein Kunde ein Versicherungsprodukt abschließt oder nicht (`response` → Zielvariable).
# 
# Für das Cross Selling setzt die Versicherungsgesellschaft verschiedene Kanäle ein. Unter anderem Telefon, E-Mail, Recommendations im Online-Banking oder per App.
# 
# Zu diesem Zweck haben wir von unserem Auftraggeber einen Datenbestand mit 380.999 Zeilen und 12 verschiedenen Variablen erhalten. Neben dem Datenbestand haben wir von einem Verantwortlichen der NextGenInsurance eine Kurzbeschreibung des Unternehmens und des Produktes erhalten. Darüber hinaus wurde uns eine eine kurze Beschreibung der Daten in Form eines Data Dictionaries erhalten.  

# ## Vorgehensweise
# Wir führen eine praktische Analyse des Datensatzes nach CRISP-DM durch. Auf den Schritt "Deployment" wird verzichtet, da die Ergebnisse vorher dem Auftraggeber als PowerPoint präsentiert werden 

# ```{figure} crisp_dm_python/crisp_dm_image.png
# :height: 500px
# :name: CRISP-DM
# :align: center
# 
# https://miro.medium.com/max/1055/1*d-WD7tNAn9s5i2Z0tDCsag.png (10.01.22)
# ```

# ## Data Dictionary
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
