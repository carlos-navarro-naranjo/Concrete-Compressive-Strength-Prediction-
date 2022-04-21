# Concrete-Compressive-Strength-Prediction-
My task is to create a regression model that accurately predicts the compressive strength of concrete using a dataset

In engineering we encounter many problems for which theoretical models are either unavailable or would be impractical to use. One area this occurs frequently is materials processing.
Complex factors: chemistry, thermodynamics, ambient conditions, random variability combine to make the optimization of material properties very difficult.

My task is to create a regression model that accurately predicts the compressive strength of concrete.
I was given a dataset (CSV file) of samples from a series of experiments. 
The dataset has the following features:
- Cement [kg/m3] 
- Blast furnace slag [kg/m3]
- Fly ash [kg/m3]
- Water [kg/m3] 
- Superplasticizer [kg/m3]
- Coarse aggregate [kg/m3] 
- Fine aggregate [kg/m3] 
- Age [days] 

The unusual units are interpreted as how many kg of the indicated material go into a cubic meter of total mixture. It is a unit of convenience for the plant. The age indicates how long the resulting mixture cured prior to the strength test. All components are mixed identically. 

The dataset has one response variable: 
- Concrete compressive strength MPa
