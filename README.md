# Indoor Respiratory Disease Transmission Modeling and Analysis

This repository provides a Python package designed for the analysis and simulation of respiratory disease transmission in indoor environments. Using real-time movement data collected via UWB RTLS devices, the package calculates individual transmission rates and basic reproduction numbers (R₀). By retrofitting disease-specific probability functions, it offers a comprehensive framework for studying disease spread in various high-risk indoor settings. 

This package is an essential tool for researchers, healthcare professionals, and policymakers aiming to understand and mitigate the spread of respiratory diseases in indoor settings.

## Key Features

- **Transmission Rate Calculation**: Models the probability of disease spread based on contact intensity and proximity.
- **$R_0$ Simulation**: Estimates the basic reproduction number for different indoor environments.
- **Customizable Parameters**: Includes options to simulate interventions like mask use and ventilation adjustments.
- **Data Integration**: Works with real-time movement data to create accurate and dynamic simulations.
- **Visualizations**: Generates detailed insights, including contact networks and transmission heatmaps.

## Input Data

The package requires movement data in a CSV file format. This data is critical for modeling transmission rates and simulating the basic reproduction number (R₀). The input data should have the following columns:

### Required Fields
| Column Name     | Description                                                                 |
|------------------|-----------------------------------------------------------------------------|
| `tagId`          | Unique identifier for each tag (e.g., healthcare worker, object).          |
| `x`              | X-coordinate of the tag's position in the environment.                    |
| `y`              | Y-coordinate of the tag's position in the environment.                    |
| `z`              | Z-coordinate (optional, default is 0 for 2D environments).                |
| `timestamp`      | Unix timestamp of the recorded position.                                  |


### Example Input Data
```csv
tagId,x,y,z,timestamp
1,12.5,34.7,0,1684251456
2,14.2,36.3,0,1684251460
3,10.9,33.8,0,1684251464


