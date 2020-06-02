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
* Dropped columns containing over 40% of values missing
* Filled in the Electrical and Masonry veneer type with their most common values
* Replaced NaN values in the nine basement and garage columns that didn't include their build date with None
* Changed all None values in Garage year built to zero so we could perform calculations if needed
* Took the numeric columns and filled in all their NaN values with the most common value

## EDA
I looked at the distributions of various columns in the data including both the numeric and categorical columns. Below are a few higlights from this.

![](https://github.com/kenp8842/ames_homes_proj/blob/master/Correlation%20Heatmap.png "Correlation Heatmap")
![](https://github.com/kenp8842/ames_homes_proj/blob/master/Ordinal%20Columns%20Distribution.png "Ordinal Columns Distribution")

## Feature Engineering

* Created a liveable square footage feature by adding total square footage columns(including only 70% for basement square footage) and subtracting the low quality square footage
* Built a total bathroom column by combining above ground and basement full and half baths. Also, set maximum bathrooms at 3.5 for model.
* Created an effective house age feature for the model. This involved getting the houses age from the year built, as well as the age of
the remodel and applying formula for effective age.

## Model Building

First I transformed the categorical variables into dummy variables. Then I split the data into train and test sets with a test size of 20%.

I tried three different model and evaluated them using mean absolute error. I chose mean absolute error for its ease of understanding.

Three different Models:
* **Linear Regression** - Served as a baseline for the model and helped in understanding feature importance
* **Lasso Regression** - Due to the sparseness of data in the categorical variables, I thought a normalized regression like Lasso could be effective.
* **Random Forest** - Thought it would be good fit given sparsity of the data.

## Model Performance

The Random Forest model outperformed the other models for the train and test sets.

* **Random Forest**: MAE = 16379
* **Lasso Regression**: MAE = 19405
* **Linear Regression**: MAE = 19436
