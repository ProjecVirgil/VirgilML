"""
    Returns:
        _type_: _description_
"""
import joblib
import numpy as np
import seaborn as sns
from pandas import read_csv
from sklearn.svm import SVC
from sklearn.utils import shuffle 
from sklearn.model_selection import cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.calibration import CalibratedClassifierCV
from matplotlib import pyplot as plt

class ML:
    """
    Class for creating, analyzing and developing models for Machine learning by @retr0
    """
    def __init__(self) -> None:
        self.input_dirty = "data/data_test/Dataset_LearningW.csv"
        self.testing_dirty = "data/data_test/Dataset_Testing.csv"
        self.possible_value = ["OR","MT","GDS","MC","NW","EV","TM","VL","MU","AL"]

        self.load()
        self.split()

        # Create vector TFI
        self.tfidf_vectorizer = TfidfVectorizer()
        self.train_features = self.tfidf_vectorizer.fit_transform(self.train_data)
        self.test_features =  self.tfidf_vectorizer.transform(self.test_data)

        # Training the model
        self.svm_model = SVC(kernel='linear', probability=True)  # Aggiungi 'probability=True' per calcolare le probabilità
        self.calibrated_model = CalibratedClassifierCV(self.svm_model)  # Modello calibrato per le probabilità
        self.calibrated_model.fit(self.train_features, self.train_labels)

        self.performance()
        self.save_model()
        
    def visual_matrix(self):
        """
        Calculate the confusion matrix
        """
        conf_matrix = confusion_matrix(ml.test_labels, ml.predictions, labels=ml.possible_value)        
        plt.figure(figsize=(10, 8))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=ml.possible_value, yticklabels=ml.possible_value)
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')
        plt.show()       
    def visual_accuracy(self):
        """
        Create a graph for show the accuracy
        """
        plt.figure(figsize=(8, 6))
        plt.bar(['Test Accuracy', 'Train Accuracy'], [ml.accuracy, ml.train_acc], color=['blue', 'green'])
        plt.ylim(0, 1)
        plt.ylabel('Accuracy')
        plt.title('Test vs Train Accuracy')
        plt.show()

    def performance(self):
        """
        Evaluates the performance of the model on the test set
        """
        self.predictions =  self.calibrated_model.predict(self.test_features)
        self.accuracy = accuracy_score(self.test_labels, self.predictions)
        self.train_predictions =  self.calibrated_model.predict(self.train_features)
        self.train_acc = accuracy_score(self.train_labels, self.train_predictions)
        print("Test Accuracy:", self.accuracy)
        print("Train Accuracy:", self.train_acc)
        
    def save_model(self):
        """
        Export the model
        """
        model_filename = "model/model.pkl"
        joblib.dump(self.calibrated_model, model_filename)
        vectorizer_filename = "model/tfidf_vectorizer.pkl"
        joblib.dump(self.tfidf_vectorizer, vectorizer_filename)

    def cross_validation(self):
        """
        Calculates accuracy using cross-validation
        """
        num_folds = 5  # Number of fold for cross-validation
        scores = cross_val_score(self.calibrated_model, self.train_features, self.train_labels, cv=num_folds)
        for fold_num, score in enumerate(scores, start=1):
            print(f"Fold {fold_num} Accuracy: {score:.4f}")

         # Stampa la media delle precisioni su tutti i fold
        average_accuracy = scores.mean()
        print(f"Average Accuracy: {average_accuracy:.4f}")
        
        # Crea un grafico per mostrare i punteggi ottenuti nei vari fold
        plt.figure(figsize=(10, 6))
        plt.bar(np.arange(1, num_folds + 1), scores, color='orange')
        plt.xlabel('Fold')
        plt.ylabel('Score')
        plt.title('Cross-Validation Scores')
        plt.xticks(np.arange(1, num_folds + 1))
        plt.ylim(0, 1)  # Assicurati che l'asse delle y vada da 0 a 1
        plt.show()

    def load(self):
        """
        Load and shuffle the dataset
        """ 
        self.dataset_training = shuffle(read_csv(self.input_dirty, sep=';', encoding='utf-8'), random_state=42)
        self.dataset_testing = shuffle(read_csv(self.testing_dirty, sep=';', encoding='utf-8'), random_state=42)

    def split(self):
        """
        Split the data in train and test
        """ 
        self.train_data = self.dataset_training["Frasi"]
        self.test_data = self.dataset_testing["Frasi"]
        
        self.train_labels = self.dataset_training["Contesto"]
        self.test_labels = self.dataset_testing["Contesto"]

    def prevision(self, sentence: str):
        """
        Make a prediction based on the model

        Args:
            sentence (str): _description_

        Returns:
            _type_: _description_
        """
        example_features = self.tfidf_vectorizer.transform([sentence])
        predicted_context2 = self.calibrated_model.predict(example_features)                        
        return predicted_context2[0]

ml = ML()
ml.visual_accuracy()
ml.visual_matrix()
ml.cross_validation()
prevision = ml.prevision("Ciao come stai?")
print(prevision)