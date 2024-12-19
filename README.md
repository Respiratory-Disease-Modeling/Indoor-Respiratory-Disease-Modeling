# Indoor Respiratory Disease Transmission Modeling and Analysis

This repository provides a Python package designed for the analysis and simulation of respiratory disease transmission in indoor environments. Using real-time movement data collected via UWB RTLS devices, the package calculates individual transmission rates and basic reproduction numbers (R₀). By retrofitting disease-specific probability functions, it offers a comprehensive framework for studying disease spread in various high-risk indoor settings.

## Key Features

- **Transmission Rate Calculation**: Models the probability of disease spread based on contact intensity and proximity.
- **R₀ Simulation**: Estimates the basic reproduction number for different indoor environments.
- **Customizable Parameters**: Includes options to simulate interventions like mask use and ventilation adjustments.
- **Data Integration**: Works with real-time movement data to create accurate and dynamic simulations.
- **Visualizations**: Generates detailed insights, including contact networks and transmission heatmaps.

This package is an essential tool for researchers, healthcare professionals, and policymakers aiming to understand and mitigate the spread of respiratory diseases in indoor settings.

## Installation

To install the package, run:

```bash
pip install <package-name>
