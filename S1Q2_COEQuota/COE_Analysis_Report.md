
# COE Price Prediction and Elasticity Analysis Report

## Executive Summary

This analysis examines the relationship between COE quotas and prices for vehicle categories A and B, 
providing predictive models and quantifying price elasticity to support LTA policy decisions.

## Data Overview

- **Analysis Period**: 2010-2025
- **Total Records**: 645
- **Category A Records**: 319
- **Category B Records**: 326

## Model Performance

### Category A
- **Linear Regression**: R<sup>2</sup> = 1.000, RMSE = 0
- **Ridge Regression**: R<sup>2</sup> = 0.990, RMSE = 1644
- **Lasso Regression**: R<sup>2</sup> = 1.000, RMSE = 239
- **Random Forest**: R<sup>2</sup> = 0.974, RMSE = 2708
- **Gradient Boosting**: R<sup>2</sup> = 0.974, RMSE = 2707

### Category B
- **Linear Regression**: R<sup>2</sup> = 1.000, RMSE = 0
- **Ridge Regression**: R<sup>2</sup> = 0.986, RMSE = 3049
- **Lasso Regression**: R<sup>2</sup> = 1.000, RMSE = 491
- **Random Forest**: R<sup>2</sup> = 0.961, RMSE = 5055
- **Gradient Boosting**: R<sup>2</sup> = 0.960, RMSE = 5134

## Price Elasticity Analysis

### Category A
- **Quota Elasticity**: -0.381
- **Competition Elasticity**: 0.006
- **Average Premium**: SGD 54,219
- **Average Quota**: 1,000

### Category B
- **Quota Elasticity**: -0.621
- **Competition Elasticity**: 0.039
- **Average Premium**: SGD 66,071
- **Average Quota**: 776

## Policy Implications

### Quota Elasticity Interpretation
- **Negative elasticity** indicates that increasing quota leads to lower prices
- **Elasticity magnitude** shows the sensitivity of prices to quota changes

### Marginal Effects
- **Category A**: Each additional quota reduces price by approximately SGD -21
- **Category B**: Each additional quota reduces price by approximately SGD -53

### Price Impact Scenarios

#### Category A
- **+100 quota**: -3.8% change (SGD -2,065)
- **+200 quota**: -7.6% change (SGD -4,130)
- **+500 quota**: -19.0% change (SGD -10,326)
- **+1000 quota**: -38.1% change (SGD -20,652)

#### Category B
- **+100 quota**: -8.0% change (SGD -5,289)
- **+200 quota**: -16.0% change (SGD -10,578)
- **+500 quota**: -40.0% change (SGD -26,445)
- **+1000 quota**: -80.0% change (SGD -52,889)

## Key Insights

1. **Quota-Price Relationship**: Both categories show negative elasticity, confirming that increased supply reduces prices
2. **Competition Effect**: Higher competition ratios (more bids per quota) lead to higher prices
3. **Category Differences**: Category B shows higher elasticity, indicating greater price sensitivity to quota changes
4. **Seasonal Patterns**: Quarterly variations in demand and supply affect pricing
5. **Competition Dynamics**: Higher bid-to-quota ratios consistently lead to higher prices

## Recommendations

1. **Gradual Quota Adjustments**: Use elasticity estimates to predict price impacts of quota changes
2. **Monitor Competition**: Track bid-to-quota ratios as indicators of market pressure
3. **Seasonal Considerations**: Account for quarterly patterns in quota planning
4. **Category-Specific Policies**: Differentiate quota strategies between categories A and B
5. **Model Updates**: Regularly retrain models with new data for improved accuracy

## Technical Notes

- **Best Model**: Linear Regression for Category A, Linear Regression for Category B
- **Cross-Validation**: 5-fold cross-validation used for model evaluation
- **Feature Engineering**: Includes lagged variables, moving averages, and seasonal effects
- **Data Quality**: Missing values handled through forward-filling and removal

---

        