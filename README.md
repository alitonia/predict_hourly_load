### Introduction

This is a project to predict hourly load

#### Project Goals

- Develop a machine learning model to forecast hourly electricity load
- Utilize historical data to train and validate the model
- Implement a user-friendly interface for data visualization and model predictions

### Project Structure

- `data/`: Contains historical electricity load data
- `models/`: Stores trained models
- `code/`: Jupyter notebooks for data exploration and model development
- `derivatives/`: Application for data visualization and predictions
- `requirements.txt`: List of Python packages required for the project

### Project Dependencies

- Python 3.8+
- Flask
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn

### How to run

To run the project, follow these steps:

1. Clone the repository: `git clone https://github.com/yourusername/yourproject.git`
2. Navigate to the project directory: `cd yourproject`
3. Install dependencies: `pip install -r requirements.txt`
4. Explore data in the __code__ repository.
5. Start forecasting application: `cd derivatives/forecast_service && bash run.sh`
6. Start Interactive dashboard: TODO

### Remaining tasks

1. Download hourly weather data and finish integrating it into baseline data
2. Create new prediction model with weather data
3. Integrating new model into derivatives
4. Finish the interactive dashboard
5. Make slides

### Note

- File __data/individual_household_electric_power_consumption/household_power_consumption.csv__ is missing, but you can
  get it from the zip file.