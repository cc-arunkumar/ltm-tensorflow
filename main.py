import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import tensorflow as tf

def preprocess_data(dataframe):
    """
    Selects the three feature columns and extracts the target column,
    casting them to float32 numpy arrays for TensorFlow compatibility.
    """
    feature_columns = ['Transaction_Amount_USD', 'Transaction_Hour', 'Is_International']
    features = dataframe[feature_columns].to_numpy(dtype=np.float32)
    target = dataframe['Is_Fraud'].to_numpy(dtype=np.float32)
    return features, target

def train_model(features, target):
    """
    Splits the data into training and testing sets, creates and adapts a Normalization layer,
    builds a Keras Sequential model, compiles, trains, evaluates, and exports it as a SavedModel.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        features, target, test_size=0.2, random_state=42
    )
    
    # Create and adapt the normalization layer
    normalizer = tf.keras.layers.Normalization()
    normalizer.adapt(X_train)
    
    # Build Keras Sequential model architecture
    model = tf.keras.Sequential([
        normalizer,
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(8, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    
    # Compile model
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    # Train for 10 epochs
    model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=1)
    
    # Evaluate accuracy on test set
    y_pred_prob = model.predict(X_test)
    y_pred = (y_pred_prob >= 0.5).astype(np.float32).flatten()
    accuracy = accuracy_score(y_test, y_pred)
    
    # Export model as a SavedModel
    model.export("fraud_model/1")
    
    return model, accuracy

if __name__ == "__main__":
    # Load dataset from data directory
    data_path = "data/transactions.csv"
    df = pd.read_csv(data_path)
    
    # Preprocess data
    features, target = preprocess_data(df)
    
    # Train and export model
    model, accuracy = train_model(features, target)
    
    print(f"Model Accuracy: {accuracy:.4f}")
    print("Model successfully trained and exported to fraud_model/1/")