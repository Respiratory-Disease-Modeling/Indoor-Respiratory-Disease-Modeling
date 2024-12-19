# Indoor Respiratory Disease Transmission Modeling and Analysis

This repository provides a Python package designed for the analysis and simulation of respiratory disease transmission in indoor environments. Using real-time movement data collected via UWB RTLS devices, the package perfoms the set of contact analytics (contact tracing, contact intensity, contact density) and then calculates the individual transmission rates ($\beta$) and basic reproduction numbers ($R_0$). By retrofitting disease-specific probability functions, it offers a comprehensive framework for studying disease spread in various high-risk indoor settings. 

This package is an essential tool for researchers, healthcare professionals, and policymakers aiming to understand and mitigate the spread of respiratory diseases in indoor settings.

**Interpretation of Analyses Described in the Journal**

This repository provides the code implementation for the analyses and simulations described in our journal article. Please refer to the article for detailed explanations of the methodology, data requirements, and interpretations.

**Reference**

For complete details on the research and implementation, please cite to our journal article: **Currently, the journal is under peer review. Once it is published and publicly available, we will update the DOI/link**

[Author Names, “Article Title,” Journal Name, Year, DOI/Link]

## Key Features

- **Transmission Rate Calculation**: Models the probability of disease spread based on contact intensity and proximity.
- **$R_0$  Simulation**: Estimates the basic reproduction number for different respiratory diseases in indoor environments.
- **Customizable Parameters**: Includes options to simulate interventions like mask use and social distancing threshold adjustments.
- **Data Integration**: Works with real-time movement data to create accurate and dynamic simulations.
- **Visualizations**: Generates detailed insights, including contact tracing and transmission dynamics.

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

## How to Use the Package

1. **Provide Your Data**  
   - Open the `driver.py` file in your project directory.
   - On **line 19**, specify the file path to your CSV data file:
     ```python
     input_file_path = "path/to/your/file.csv"
     ```

2. **Choose Your Desired Plots/Results**  
   - The package offers **19 kinds of plots/results**, referred to as "parts." Please refer next section for detailed information about avaiable parts.
   - Uncomment the line corresponding to the part you want to generate in the `driver.py` file. 
   - Example:
     ```python
     # Uncomment the desired part to run
     # Part 1
     # Part 2
     ```

   > **Note:** Some parts may depend on the results of previous parts. Ensure to run those prerequisite parts before running the desired one.

3. **Run the Script**  
   - Save the changes in the `driver.py` file.
   - Execute the script in your terminal:
     ```bash
     python3 driver.py
     ```

4. **Access the Results**  
   - Once the script completes execution, you can find the generated **plots** and **results** in the `outputs` directory:
     - Plots: Saved as `.pdf` files.
     - Results: Saved as `.csv` files.

# Available Parts and Their Functions

| **Part** | **Description**                                                             |
|----------|-----------------------------------------------------------------------------|
| Part 1   | Generate pre-processed data                                                |
| Part 2   | Data analysis                                                              |
| Part 3   | Plotting anchors                                                           |
| Part 4   | Plotting individual tags                                                   |
| Part 5   | Plot individual tag trajectories with respect to anchors                   |
| Part 6   | Plot tag pair trajectories                                                 |
| Part 7   | Plot all tags trajectories                                                 |
| Part 8   | Plot all tags trajectory points                                            |
| Part 9   | Contact duration                                                           |
| Part 10  | Contact intensity                                                          |
| Part 11  | Plot contact density                                                       |
| Part 12  | Distance matrices                                                          |
| Part 13  | Tag pair distances                                                         |
| Part 14  | Contact tracing network for longer duration                                |
| Part 15  | Contact tracing network (5-second duration)                                |
| Part 16  | Contact tracing network for each individual tag                            |
| Part 17  | Individual transmission rates for various diseases                         |
| Part 18  | Individual transmission rates for COVID-19                                 |
| Part 19  | Simulation of basic reproduction numbers ($R_0$)
