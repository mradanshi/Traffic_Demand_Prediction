# Traffic Demand Prediction

Machine learning project for predicting traffic demand using road, weather, location, and time-based features.

## Overview

This project was developed as part of a traffic demand prediction hackathon. The goal was to predict traffic demand from structured tabular data using regression techniques.

## Features Used

* Geohash (location)
* Day
* Road Type
* Number of Lanes
* Large Vehicles Allowed
* Nearby Landmarks
* Temperature
* Weather
* Hour
* Minute

## Data Preprocessing

* Missing value handling
* Timestamp feature extraction
* Binary feature encoding
* Categorical feature processing using CatBoost

## Model

**CatBoost Regressor**

Best configuration:

* Iterations: 2000
* Depth: 8
* Learning Rate: 0.05

## Libraries Used

* Pandas
* Scikit-learn
* CatBoost

## Results

Best public leaderboard score:

**90.87459**

## Project Structure

```text
.
├── main.py
├── report.txt
├── README.md
└── .gitignore
```

