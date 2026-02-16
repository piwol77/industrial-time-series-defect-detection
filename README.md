# Industrial Defect Detection – ML Feasibility Study

## Overview
This repository presents a feasibility study of machine learning methods for early defect detection in an industrial manufacturing process based on time-series data.

The primary objective of the project was to evaluate and compare different ML approaches for classifying process-related time-series signals, with a focus on early detection of rare defects under real-world data constraints.  
The project was designed as an end-to-end data and ML pipeline, covering data ingestion, storage, processing, modeling and visualization.

---

## Business Context
Early identification of defects in industrial processes is critical to:
- reduce operational costs,
- minimize scrap and rework,
- prevent costly downstream failures,
- improve overall process reliability.

This study investigates whether machine learning models can support these goals when applied to realistic industrial data and integrated with modern data platforms.

---

## Process Context
The analyzed system represents a generic industrial production process, including:
- identification of key process parameters,
- characterization of potential defect types,
- analysis of available sensor signals recorded during operation.

The project intentionally abstracts from a specific process to focus on general challenges of industrial defect detection and data-driven decision support.

---

## Data Sources & Ingestion
Raw process data was collected from operational systems and ingested into the analytical environment using a dedicated data ingestion pipeline:
- original process data retrieved from FTP-based data sources,
- batch-oriented ingestion of new data into a centralized data warehouse,
- separation of raw, processed and analytical data layers.

---

## Data Engineering Architecture

### Data Storage & Processing
- Snowflake used as the central data warehouse for storing raw and processed time-series data
- Data transformations and feature engineering implemented using Snowflake DataFrame API (compute Snowflake's layer)
- Processed datasets written back to Snowflake for reuse by downstream analytics and ML workflows

### Data Pipelines
- Automated ingestion of newly available data batches
- Reproducible data preprocessing and feature engineering pipelines
- Clear separation between:
  - data ingestion,
  - data preparation,
  - model training,
  - model scoring

---

## Data Challenges
The project addressed several real-world industrial data issues:
- highly imbalanced dataset with a very small number of defective samples,
- noisy time-series sensor data,
- limited number of recorded process parameters,
- constraints on data volume, data freshness and labeling.

---

## Methodology

### Data Preparation & Orchestration
- Data preparation and experimentation workflows implemented using Dataiku
- Rapid prototyping of feature engineering and ML pipelines
- Integration of Dataiku workflows with Snowflake as the underlying data platform

### Machine Learning Models
The following model architectures were implemented and evaluated:
- XGBoost (reference model)
- 1D Convolutional Neural Network (CNN)
- Recurrent Neural Networks:
  - LSTM
  - GRU

### Model Optimization
- Hyperparameter tuning using:
  - Optuna
  - GridSearchCV
- Evaluation using metrics suitable for imbalanced classification:
  - Recall
  - F2-score

---

## Results
- The GRU-based model achieved the highest performance among the evaluated approaches.
- Overall model effectiveness was limited primarily by data quality and class imbalance rather than model architecture.
- Increasing model complexity did not compensate for insufficient or noisy data.

---

## Visualization
To support practical interpretation of results, a dedicated visualization interface was implemented in an industrial visualization environment:
- real-time visualization of incoming process signals,
- display of classification and scoring results,
- support for operational monitoring and analysis.

---

## Key Findings
- Data quality and class imbalance are dominant factors in industrial ML success.
- Reliable data engineering and data availability are prerequisites for effective ML models.
- Model complexity alone cannot compensate for insufficient or poorly structured data.
- Early feasibility studies are essential before deploying ML-based solutions in production environments.

---

## Conclusions and Future Work
The study demonstrates that ML-based defect detection in industrial settings requires:
- robust and scalable data ingestion pipelines,
- improved data acquisition strategies,
- increased availability of representative defect samples,
- richer and more informative process parameterization.

Future improvements may include enhanced sensor coverage, improved labeling processes, stronger data validation and hybrid approaches combining domain knowledge with data-driven models.

---

## Tech Stack
- Python
  - NumPy & Pandas
  - scikit-learn
  - XGBoost
  - PyTorch
  - Optuna
- Snowflake (data warehouse & compute layer)
- Dataiku (data preparation and ML workflows)
- FTP-based data ingestion
- Industrial visualization platform

---

## Project Status
🚧 Work in progress  
This repository is a recreated demo based on a university thesis.  

---

## Disclaimer
This project is a recreated demonstration based on an academic master's thesis.  
No proprietary company data, code or confidential information is included.
