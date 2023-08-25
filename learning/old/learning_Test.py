import numpy as np
import matplotlib.pyplot as plt
from pandas import read_csv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV  # Per calcolare le probabilità calibrate

class ML_TEST:
    def __init__(self) -> None:
        
        input_worked = "data/Dataset_LearningW.csv"
        input_dirty = "data/Dataset_Learning.csv"
        
        testing_worked = "data/Dataset_TestingW.csv"
        testing_dirty = "data/Dataset_Testing.csv"
        
        # Carica i dataset
        self.datasetTraingW = read_csv(input_worked, sep=';', encoding='utf-8')
        self.datasetTraing = read_csv(input_dirty, sep=';', encoding='utf-8')

        self.datasetTestingW = read_csv(testing_worked, sep=';', encoding='utf-8')
        self.datasetTesting = read_csv(testing_dirty, sep=';', encoding='utf-8')

        # Mescola i dati
        self.datasetTraingW = shuffle(self.datasetTraingW, random_state=42)
        self.datasetTraing = shuffle(self.datasetTraing, random_state=42)
        self.datasetTestingW = shuffle(self.datasetTestingW, random_state=42)
        self.datasetTesting = shuffle(self.datasetTesting, random_state=42)

        # Dividi i dati in train e test
        self.train_dataW = self.datasetTraingW["Frasi"]
        self.train_data = self.datasetTraing["Frasi"]
        self.test_data = self.datasetTesting["Frasi"]

        self.train_labels = self.datasetTraing["Contesto"]
        self.test_labels = self.datasetTesting["Contesto"]
        self.train_labelsW = self.datasetTraingW["Contesto"]
        
        
        # Crea il vettore delle features TF-IDF
        self.tfidf_vectorizer = TfidfVectorizer()
        self.train_features = self.tfidf_vectorizer.fit_transform(self.train_data)
        self.test_features =  self.tfidf_vectorizer.transform(self.test_data)

        self.tfidf_vectorizerW = TfidfVectorizer()
        self.train_featuresW = self.tfidf_vectorizerW.fit_transform(self.train_dataW)
        self.test_featuresW =  self.tfidf_vectorizerW.transform(self.test_data)
        
        # Addestra il modello SVM
        self.svm_model = SVC(kernel='linear', probability=True)  # Aggiungi 'probability=True' per calcolare le probabilità
        self.calibrated_model = CalibratedClassifierCV(self.svm_model)  # Modello calibrato per le probabilità
        self.calibrated_model.fit(self.train_features, self.train_labels)

        # Addestra il modello SVM W
        self.svm_modelW = SVC(kernel='linear', probability=True)  # Aggiungi 'probability=True' per calcolare le probabilità
        self.calibrated_modelW = CalibratedClassifierCV(self.svm_modelW)  # Modello calibrato per le probabilità
        self.calibrated_modelW.fit(self.train_featuresW, self.train_labelsW)
        
        # Valuta le prestazioni del modello sul set di test
        self.predictions =  self.calibrated_model.predict(self.test_features)
        self.accuracy = accuracy_score(self.test_labels, self.predictions)
        print("Test Accuracy:", self.accuracy)
        self.train_predictions =  self.calibrated_model.predict(self.train_features)
        self.train_acc = accuracy_score(self.train_labels, self.train_predictions)
        print("Train Accuracy:", self.train_acc)
        
        # Valuta le prestazioni del modello sul set di test
        self.predictionsW =  self.calibrated_modelW.predict(self.test_featuresW)
        self.accuracyW = accuracy_score(self.test_labels, self.predictionsW)
        print("Test Accuracy data worked:", self.accuracyW)
        self.train_predictionsW =  self.calibrated_modelW.predict(self.train_featuresW)
        self.train_accW = accuracy_score(self.train_labelsW, self.train_predictionsW)
        print("Train Accuracy data worked :", self.train_accW)
    
    
    
    def prevision(self, command: str):
        
        # Esempio di frase da testare
        example_sentence = command
        # Trasforma l'esempio di frase in una rappresentazione TF-IDF
        example_features = self.tfidf_vectorizer.transform([example_sentence])
        
        
        # Fai una previsione utilizzando il modello
        predicted_context2 = self.calibrated_model.predict(example_features)
        # Fai una previsione utilizzando il modello calibrato
        predicted_probs = self.calibrated_model.predict_proba(example_features)[0]
        
        # Stampa le probabilità previste per ciascuna classe
        possible_value = ["OR","MT","GDS","MC","NW","EV","TM","VL","MU"]
        for value, prob in zip(possible_value, predicted_probs):
            print(f"Value {value}: Probability {prob:.4f}")
         # Trova l'indice della classe con la massima probabilità
        predicted_class_idx = np.argmax(predicted_probs)
        predicted_context = possible_value[predicted_class_idx]
        
        predicted_context2W = self.calibrated_modelW.predict(example_features)
        # Fai una previsione utilizzando il modello calibrato
        predicted_probsW = self.calibrated_modelW.predict_proba(example_features)[0]
        
        # Stampa le probabilità previste per ciascuna classe
        possible_value = ["OR","MT","GDS","MC","NW","EV","TM","VL","MU"]
        for value, prob in zip(possible_value, predicted_probsW):
            print(f"Value {value}: Probability {prob:.4f}")
        # Trova l'indice della classe con la massima probabilità
        predicted_class_idx = np.argmax(predicted_probsW)
        predicted_contextW = possible_value[predicted_class_idx]
            
            
        result = f"Non worked:{predicted_context} {predicted_context2[0]}; Worked:{predicted_contextW} {predicted_context2W[0]}"
        # Stampa il contesto previsto per l'esempio di frase
        return result


    
    