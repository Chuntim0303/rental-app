## Problem Statement
This project aims to estimate property rental prices using a predictive model. By analyzing various property attributes like location, number of rooms, size, and furnishing status, this tool provides an estimate of potential rent. The tool helps both renters and landlords make informed decisions by providing rental price predictions.<br>
<br>

## Data Overview
The dataset is scraped from Mudah.my and can be found on Kaggle. It includes several features such as:

Location
- Size (square feet)
- Number of rooms
- Number of bathrooms
- Parking availability
- Furnishing status
<br>


## Exploratory Data Analysis (EDA)
The EDA helps uncover insights and patterns within the data. It also helps in data cleaning and making model more accurate. Key visualizations and analyses performed include:
Boxplots of Continuous Variables:
The 'size' feature shows a right-skewed distribution with outliers. These outliers are removed using a Z-score threshold of 3 to improve model performance.
Distribution of Property Sizes:
A scatter plot is used to examine the relationship between property size and number of rooms, showing a positive correlation. Larger properties tend to have higher rental prices.
<br>


## Correlation Heatmap:
A heatmap displays correlations between features such as:

Size vs Bathroom (0.44), Size vs Rooms (0.38): Moderate positive correlations.
Rooms vs Bathroom (0.68): Strong positive relationship.
Weak correlations between parking and other features.
<br>


## Modeling
Several machine learning algorithms were tested to predict rental prices:

Random Forest Regressor:
- Best model with lowest RMSE (306.82) and highest R² (0.78).
Gradient Boosting Regressor:
- Moderate performance with RMSE of 391.78 and R² of 0.64.
Linear Regression:
- Worst performance with RMSE (466.96) and R² (0.49).
<br>
<br>

## Feature Importance:
- Feature importance from the Random Forest model identified key factors affecting rental prices:

- Size (0.3548): The most significant predictor.
- Furnished (0.2497): Second most important.
- Location (0.1770): Also significant.
Other factors like property type, region, parking, rooms, and bathrooms had a lesser impact.
<br>

## Conclusion
Key Findings:
- The Random Forest model provides accurate rental price predictions.
- Size, furnishing, and location are the most influential factors in determining rental prices.
<br>


## Recommendations:
For Renters: Use this tool to estimate how much you might pay based on property features.
For Landlords: Use this tool to set competitive rental prices based on comparable properties.


