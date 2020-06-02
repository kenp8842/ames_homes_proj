# Ames, Iowa Home Price Prediction: Project Overview
* Created a model to estimate home prices(MAE ~ $16.4K) in Ames, Iowa to show skills in machine learning.
* Used existing data for homes that were sold between 2006 and 2010.
* Engineered features from the data to better quantify the liveable square footage, total bathrooms and the effective age of the homes.
* Optimized Linear, Lasso, and Random Forest Regressors using GridSearchCV to find the best model.

## Code and References Used
**Python Version:** 3.7                                                                                                    
**Packages:** pandas, numpy, sklearn, matplotlib, seaborn

## Resources

### Data
https://www.tandfonline.com/doi/abs/10.1080/10691898.2011.11889627

### Data Dictionary
http://jse.amstat.org/v19n3/decock/DataDocumentation.txt

### Feature Engineering

**Liveable Square Footage**

https://www.opendoor.com/w/blog/factors-that-influence-home-value

**Bathrooms**

https://nationalpost.com/life/homes/adding-value-convenience-with-basement-bathroom
https://www.thetruthaboutrealty.com/half-bath-vs-full-bath-wheres-the-value/

**Effective Age**

https://www.corelogic.com/blog/2016/11/effective-age-versus-actual-age.aspx                   
https://www.homeadvisor.com/cost/additions-and-remodels/remodel-multiple-rooms/                             
https://themreport.com/daily-dose/12-10-2019/smaller-home-sizes-to-become-the-norm-in-2020

## Data Cleaning

After getting the data, it needed to be cleaned up so it was useful for our model. I made the following changes.
* Dropped columns that had over 40% of values missing
* Filled in the Electrical and Masonry veneer type with their most common values
* Replaced NaN values in the 9 basement and garage columns that didn't include build date with None
* Changed all None values in Garage year built to 0 so we could perform calculations if needed
* Took the numeric columns and filled in all their NaN values with the most common value

## EDA
I looked at the distributions of various columns in the data both in the numeric and categorical columns. Below are a few higlights from this. 

