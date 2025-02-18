# Synergy-AI

A machine learning model to predict the fare prices for airline tickets based on various features such as class, source, destination, and more.

## Table of Contents
- [Introduction](#introduction)
- [Installation](#installation)
- [Usage](#usage)
- [License](#license)

## Introduction

This project is a machine learning model that predicts the fare prices for airline tickets based on various features such as class, source, destination, and more. The model is built using the Random Forest Regressor algorithm and is trained on a dataset containing information about flights from the UAE.

Built for the Artificial Intelligence and Data Science competition at Emirates Aviation University.

## Installation

To install the required dependencies, run the following command:

```bash
pip install -r requirements.txt
```

## Usage

To build the model and save it as a .pkl file, run the following command:

```bash
python build.py
```

This command will preprocess the data, train the model, evaluate it, and save the trained model to a file named flight_fare_prediction_model_uae.pkl.  To run the Flask application, use:

```bash
python app.py
```

This command will start the Flask web server, allowing you to interact with the model through a web interface.

## License

This project is licensed under the GNU GPL v3 License - see the [LICENSE](LICENSE) file for details.