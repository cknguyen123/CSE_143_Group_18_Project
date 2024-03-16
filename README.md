
## Dataset
Dataset Obtained at: https://archive.ics.uci.edu/dataset/849/power+consumption+of+tetouan+city

## File Structure:
1. 143_proj_data_extraction.py
3. Final_Project_data_vis_and_analysis.py
4. analysis.py (merged into Final_Project_data_vis_and_analysis.py)
5. Final_Project_data_vis_and_analysis.ipynb

Modules used: pandas, numpy, matplotlib.pyplot, defaultdict, seaborn, sklearn, sklearn.ensemble, calendar

To run our code, first download the dataset from the website above, then run 143_proj_data_extraction.py to generate a cleaned dataset. You can then run Final_Project_data_vis_and_analysis.ipynb in jupyter notebook to obtain the data plots.

If you download a zip of our repo, you should be able to open a jupyter notebook and run 'Final_Project_data_vis_and_analysis.ipynb' as all necessary files are contained. 

Final_Project_data_vis_and_analysis.ipynb is our main file that produces the visual data. 143_proj_data_extraction.py will generate a cleaned csv file that will use as our dataset. The file is included in this repo as 'Tetuan_City_power_consumption_cleaned.csv'. Our analysis and conclusion can be read from our presentation slide pdf

## Presentation Slides File:
ECE_143_Final_Presentation_Group.pdf

## Data Extraction
Code File: `143_proj_data_extraction.py`

In the file `143_proj_data_extraction.py`, we removed any outlier data and cleaned the output into a new CSV file. This demonstrates a practical example of data extraction and preprocessing.

To run the code: 
- Needs the dataset file Tetuan_City_power_consumption.csv
- "python 143_project_extraction.py"


## Data Visualization and Analysis
Code File: `Final_Project_data_vis_and_analysis.ipynb`

In the file,  `Final_Project_data_vis_and_analysis.ipynb`, we used the cleaned data from the extraction section as our main dataset. We further processed the data frame to create hourly, daily, and monthly averages for each feature. We used these new datasets to create various visual plots to check features against time or power consumption. We merged the functions from 'analysis.py' in the code file `Final_Project_data_vis_and_analysis.ipynb`. The purpose of this was to analyze the visual plots to find any correlation of other features to power consumption in order to build a predictor.

To run the code:
Open juptyer notebook, open `Final_Project_data_vis_and_analysis.ipynb` and run all cells

Can also run "python Final_Project_data_vis_and_analysis.py"

## Future: Predictor model
We were unable to complete this in time. However, we laid out the steps we would have gone through to implement the predictor

### General Plan:
- Randomize and split the dataset to create training data.
- Model will be based on features: temperature, hour of the day.
- Use a regression model to generate power consumption prediction.
- Final predictor model would be able to predict power consumption given: month and hour of day.

### Improvements:
- After creating a predictor, we would check accuracy with the actual average from the dataset.
- Determine the margin of error.
- Determine if addition of other features would improve accuracy of prediction.
