## Dataset
Dataset Obtained at: https://archive.ics.uci.edu/dataset/849/power+consumption+of+tetouan+city


## Data Extraction
Code File: `143_proj_data_extraction.py`

In the file `143_proj_data_extraction.py`, we removed any outlier data and cleaned the output into a new CSV file. This demonstrates a practical example of data extraction and preprocessing.


## Data Visualization and Analysis
Code File: `Final_Project_data_vis_and_analysis.py`

In the file,  `Final_Project_data_vis_and_analysis.py`, we used the cleaned data from the extraction section as our main dataset. We further processed the data frame to create hourly, daily, and monthly averages for each feature. We used this new dataset to create various plots to check features against time or power consumption. The purpose of this was to analyze the visual plots to find any correlation of other features to power consumption in order to build a predictor.


## Future: Predictor model
We were unable to complete this is time. However, we laid out the steps we would have gone through to implement the predictor

### General Plan:
- Randomize and split the dataset to create training data.
- Model will be based on features: temperature, hour of the day.
- Use a regression model to generate power consumption prediction.
- Final predictor model would be able to predict power consumption given: month and hour of day.

### Improvements:
- After creating a predictor, we would check accuracy with the actual average from the dataset.
- Determine the margin of error.
- Determine if addition of other features would improve accuracy of prediction.
