# 🎓 Student Exam Performance Indicator

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/FastAPI-0.95%2B-green)](https://fastapi.tiangolo.com/)
[![ML](https://img.shields.io/badge/Scikit--Learn-1.2%2B-orange)](https://scikit-learn.org/)

##  Project Overview
The **Student Exam Performance Indicator** is an End-to-End Machine Learning project designed to understand how various demographic and academic factors influence student performance. 

The application predicts a student's **Math Score** based on input parameters such as gender, race/ethnicity, parental level of education, lunch type, and test preparation course.

This project is built using a modular architecture with **Python**, **Scikit-Learn**, and **FastAPI** for the web interface.

##  Demo


## 🛠️ Tech Stack
* **Backend:** Python, FastAPI, Uvicorn
* **Machine Learning:** Scikit-Learn, Pandas, NumPy, Dill, XGBoost/CatBoost (if used)
* **Frontend:** HTML, CSS, Jinja2 Templates
* **Development Tools:** VS Code, Git, GitHub

##  Project Structure
The project follows a standard production-grade Machine Learning directory structure:

```bash
├── artifacts/              # Stores the trained model.pkl and preprocessor.pkl
├── notebook/               # Jupyter notebooks for EDA and experimentation
├── src/
│   ├── components/         # Core ML modules
│   │   ├── data_ingestion.py      # Reads data, splits into train/test
│   │   ├── data_transformation.py # OneHotEncoding, Scaling, Feature Engineering
│   │   └── model_trainer.py       # Trains models and evaluates performance
│   ├── pipeline/           # Inference pipelines
│   │   └── predict_pipeline.py    # Handles user input and prediction logic
│   ├── logger.py           # Custom logging configuration
│   ├── exception.py        # Custom exception handling
│   └── utils.py            # Utility functions (save/load objects)
├── templates/              # HTML templates for the web app
├── app.py                  # FastAPI application entry point
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation
