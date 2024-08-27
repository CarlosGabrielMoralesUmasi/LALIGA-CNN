# La Liga Data Analysis and Prediction using RNN

This project aims to analyze and predict the final points of La Liga football teams using a Recurrent Neural Network (RNN) with Long Short-Term Memory (LSTM) layers. The project is implemented using Python and TensorFlow, and includes steps for data preprocessing, model building, training, and evaluation.

## Project Structure

- **LaLiga_dataset.csv**: The dataset containing historical match data for La Liga teams.
- **laligaESP.ipynb**: Jupyter Notebook containing the data analysis and model training code.
- **main.py**: The main Python script that preprocesses the data, builds the model, trains it, and generates predictions.
- **README.md**: This file, providing an overview of the project and instructions on how to run the code.

## Requirements

Before running the code, ensure you have the following Python packages installed:

```bash
pip install numpy pandas matplotlib scikit-learn tensorflow
```

## Data Preprocessing

The dataset is loaded and the following features are selected for the model:

- **home_goals**: Number of goals scored by the team at home.
- **away_goals**: Number of goals scored by the team away.
- **goal_difference**: Difference between goals scored and conceded.

The target variable is:

- **points**: The points earned by the team.

The data is then normalized using `MinMaxScaler` and sequences are created for training the RNN.

## Model Architecture

The model is built using a Bidirectional LSTM architecture:

- **Layer 1**: Bidirectional LSTM with 50 units, followed by Dropout (0.4).
- **Layer 2**: Bidirectional LSTM with 50 units, followed by Dropout (0.4).
- **Output Layer**: Dense layer with 1 unit to predict the final points.

The model uses the `Adam` optimizer with a learning rate of 0.001 and `mean_squared_error` as the loss function.

## Training and Early Stopping

The model is trained for 100 epochs with a batch size of 32. Early stopping is applied to avoid overfitting, with a patience of 10 epochs.

## Evaluation and Visualization

After training, the model's predictions are compared with the actual points using a line plot, allowing for visual comparison between the predicted and actual values.

## Usage

To run the project, execute the following command:

```bash
python main.py
```

Make sure that `LaLiga_dataset.csv` is in the same directory as `main.py`.

## Results

The model predicts the final points of La Liga teams with reasonable accuracy. Further improvements can be made by tuning the model architecture, using additional features, or employing other machine learning models.

## Contributing

Contributions to the project are welcome. Please fork the repository and submit a pull request with your improvements.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.
