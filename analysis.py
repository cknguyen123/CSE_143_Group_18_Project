import pandas as pd
import matplotlib.pyplot as plt
import calendar
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor

def plotEnergyConsumptionEachZoneDaily(data):
    '''a function to plot the average energy consumption for each zone daily.
    Input: data(DataFrame of whole data we are analyzing)'''
    
    df = data.copy()
    df = df[['Zone 1 Power Consumption', 'Zone 2  Power Consumption', 'Zone 3  Power Consumption']]
    
    # daily data and sum the consumption for each zone
    df_daily = df.resample('D').sum()
    
    # plot energy consumption for each zone over time
    df_daily.plot(figsize=(12, 6))
    plt.title('Daily Energy Consumption for Each Zone')
    plt.xlabel('Month')
    plt.ylabel('Energy Consumption')
    plt.show()


def plotEnergyConsumptionMonthly(data):
    '''a function to plot the average energy consumption for each zone monthly, 
    compare the energy consumption between different zones
    Input: data(DataFrame of whole data we are analyzing)'''
    
    df = data.copy()
    df = df[['Zone 1 Power Consumption', 'Zone 2  Power Consumption', 'Zone 3  Power Consumption']]
    
    # analyzing monthly average power consumption
    monthly_average = df.resample('M').mean()
    
    monthly_average.plot(kind='bar', figsize=(12, 7))
    plt.title('Monthly Average Energy Consumption for Each Zone')
    plt.xlabel('Month')
    plt.ylabel('Average Energy Consumption')
    
    # set x-ticks to month names
    month_names = [calendar.month_abbr[i] for i in range(1, 13)]
    plt.xticks(range(len(month_names)), month_names, rotation=45)
    
    plt.show()


def plotFeatureZoneVariation(data, feature, zone):
    '''a function to plot the feature variation and the specific zone's power consumption 
    variation for each hour of the day.
    Input: data(DataFrame of whole data we are analyzing), 
           feature(string of the feature we want to see the variation)
           zone(string of the zone's name)'''
    
    df = data.copy()
    
    # convert DateTime column to datetime type
    df['DateTime'] = pd.to_datetime(df['DateTime'])

    # extract hour from DateTime
    df['Hour'] = df['DateTime'].dt.hour

    plt.figure(figsize=(12, 6))

    # plot feature variation
    plt.subplot(2, 1, 1)
    sns.boxplot(x='Hour', y=feature, data=df, color='skyblue')
    plt.title(f'Daily {feature} Variation')
    plt.xlabel('Hour')
    plt.ylabel(f'{feature}')

    # plot zone power consumption variation
    plt.subplot(2, 1, 2)
    sns.boxplot(x='Hour', y=f'{zone}', data=df, color='skyblue')
    plt.title(f'Daily {zone} Variation')
    plt.xlabel('Hour')
    plt.ylabel(f'{zone}')

    plt.tight_layout()
    plt.show()


def findFeatureImportances(data):
    '''a function to find the importance of all the features in dataset that 
    influence on power consumption of each zone.
    Input: data(DataFrame of whole data we are analyzing)
    Output: feature_importance_dfs(DataFrame of feature importances)'''

    df = data.copy()

    # define features and target variable
    columns_to_drop = ['DateTime']
    df = df.drop(columns=columns_to_drop)
    features = df.drop(columns=['Zone 1 Power Consumption', 'Zone 2  Power Consumption', 'Zone 3  Power Consumption']).columns
    target = ['Zone 1 Power Consumption', 'Zone 2  Power Consumption', 'Zone 3  Power Consumption']
    
    
    rfs = {}
    for target in targets:
        # split the data into features and target variable
        X = df[features].values
        y = df[target].values

        # initialize and train the Random Forest Regressor model
        rf = RandomForestRegressor()
        rf.fit(X, y)
        rfs[target] = rf
        
        
    # extract feature importances from the trained models for each zone
    feature_importances = {}
    for target, rf in rfs.items():
        feature_importances[target] = rf.feature_importances_    


    # create DataFrames to display feature importances for each zone
    feature_importance_dfs = {}
    for target, importance in feature_importances.items():
        feature_importance_dfs[target] = pd.DataFrame({'Feature': features, 'Importance': importance})
        feature_importance_dfs[target] = feature_importance_dfs[target].sort_values(by='Importance', ascending=False)
    
    return feature_importance_dfs


def plotFeatureImportances(feature_importance_dfs):
    '''a function to plot the importance of all the features in dataset that 
    influence on power consumption of each zone.
    Input: feature_importance_dfs(DataFrame of feature importances)'''
    
    # find the maximum importance score across all zones
    max_importance = max([df['Importance'].max() for df in feature_importance_dfs.values()])
    
    # plot feature importances for each zone (Horizontal Bar Plot)
    fig, axs = plt.subplots(len(feature_importance_dfs), 1, figsize=(8, 3*len(feature_importance_dfs)), sharex=True)
    for ax, (target, df) in zip(axs, feature_importance_dfs.items()):
        ax.barh(df['Feature'], df['Importance'], color='skyblue')
        ax.set_title(f'Feature Importances for {target}')
        ax.set_ylabel('Feature')

    plt.xlabel('Importance')
    # set the same scale for all plots
    plt.xlim(0, max_importance * 1.1)
    plt.tight_layout()
    plt.show()

