# Indoor Respiratory Disease Transmission Modeling and Analysis

This repository provides a Python package designed for the analysis and simulation of respiratory disease transmission in indoor environments. Using real-time movement data collected via UWB RTLS devices, the package perfoms the set of contact analytics (contact tracing, contact intensity, contact density) and then calculates the individual transmission rates ($\beta$) and basic reproduction numbers ($R_0$ ). By retrofitting disease-specific probability functions, it offers a comprehensive framework for studying disease spread in various high-risk indoor settings. 

This package is an essential tool for researchers, healthcare professionals, and policymakers aiming to understand and mitigate the spread of respiratory diseases in indoor settings.

**Implementation of Analyses Described in the Journal**

This repository provides the code implementation for the analyses and simulations described in our journal article. Please refer to the article for detailed explanations of the methodology, data requirements, and interpretations.

**Reference**

For complete details on the research and implementation, please cite to our journal article:
[Author Names, “Article Title,” Journal Name, Year, DOI/Link]

## Key Features

- **Transmission Rate Calculation**: Models the probability of disease spread based on contact intensity and proximity.
- **$R_0$  Simulation**: Estimates the basic reproduction number for different indoor environments.
- **Customizable Parameters**: Includes options to simulate interventions like mask use and ventilation adjustments.
- **Data Integration**: Works with real-time movement data to create accurate and dynamic simulations.
- **Visualizations**: Generates detailed insights, including contact networks and transmission heatmaps.

## Input Data

The package requires movement data in a CSV file format. This data is critical for modeling transmission rates and simulating the basic reproduction number ($R_0$ ). The data format, regardless of the RTLS device used, should be structured as tagId, x, y, z, timestamp, date, time as follows:

### Required Fields
| Column Name     | Description                                                                 |
|------------------|-----------------------------------------------------------------------------|
| `tagId`          | Unique identifier for each tag (e.g., healthcare worker, object).         |
| `x`              | X-coordinate of the tag's position in the environment.                    |
| `y`              | Y-coordinate of the tag's position in the environment.                    |
| `z`              | Z-coordinate (optional, default is 0 for 2D environments).                |
| `timestamp`      | Unix timestamp of the recorded position.                                  |
| `date`           | Date of the record (in `YYYY-MM-DD` format).                              |
| `time`           | Time of the record (in `HH:MM:SS` format).                                |


### Example Input Data
```csv
tagId,x,y,z,timestamp,date,time
1,12.5,34.7,0,1684251456,2024-06-01,12:30:56
2,14.2,36.3,0,1684251460,2024-06-01,12:31:00
3,10.9,33.8,0,1684251464,2024-06-01,12:31:04
```

# Data Analysis and Visualization with Python

This project uses Python for data manipulation, visualization, and network analysis. The required libraries and tools are listed below.

### Pre-requisites

Ensure you have the following installed before running the project:

- **Python**: Version 3.7 or higher
- **Required Python Libraries**:
  Install the necessary libraries by running:
  ```bash
  pip install pandas numpy matplotlib seaborn networkx

**Imported Libraries and Their Purpose**
- `pandas`: For data manipulation and analysis.
- `numpy`: For numerical computations.
- `os and sys`: To handle file paths and system-level operations.
- `seaborn`: For statistical data visualization.
- `matplotlib.pyplot`: For creating plots and visualizations.
- `networkx`: For graph and network analysis.
- `datetime`: To manage date and time-related data.
- `matplotlib.cm`: To handle color maps for visualizations.
- `matplotlib.backends.backend_pdf`: To export visualizations to PDF files.

