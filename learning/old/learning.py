import joblib
import numpy as np
from pandas import read_csv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.utils import shuffle  # Per calcolare le probabilità calibrate
#CONCLUSIONE C'E DA RENDERE SEMPLICEMENTE PIU EFFICENTI LE VARIE CATEGORIE DEL DATASET COME GDS E ALTRE
class ML:
    def __init__(self) -> None:
        input_dirty = "data/data_test/Dataset_LearningW.csv"
        testing_dirty = "data/data_test/Dataset_Testing.csv"
        
        # Carica i dataset
        self.datasetTraining = read_csv(input_dirty, sep=';', encoding='utf-8')
        self.datasetTesting = read_csv(testing_dirty, sep=';', encoding='utf-8')

        # Mescola i dati
        self.datasetTraining = shuffle(self.datasetTraining, random_state=42)
        self.datasetTesting = shuffle(self.datasetTesting, random_state=42)

        # Dividi i dati in train e test
        self.train_data = self.datasetTraining["Frasi"]
        self.test_data = self.datasetTesting["Frasi"]

        self.train_labels = self.datasetTraining["Contesto"]
        self.test_labels = self.datasetTesting["Contesto"]
            
        # Crea il vettore delle features TF-IDF
        self.tfidf_vectorizer = TfidfVectorizer()
        self.train_features = self.tfidf_vectorizer.fit_transform(self.train_data)
        self.test_features =  self.tfidf_vectorizer.transform(self.test_data)

        # Addestra il modello SVM
        self.svm_model = SVC(kernel='linear', probability=True)  # Aggiungi 'probability=True' per calcolare le probabilità
        self.calibrated_model = CalibratedClassifierCV(self.svm_model)  # Modello calibrato per le probabilità
        self.calibrated_model.fit(self.train_features, self.train_labels)

        # Valuta le prestazioni del modello sul set di test
        self.predictions =  self.calibrated_model.predict(self.test_features)
        self.accuracy = accuracy_score(self.test_labels, self.predictions)
        print("Test Accuracy:", self.accuracy)
        self.train_predictions =  self.calibrated_model.predict(self.train_features)
        self.train_acc = accuracy_score(self.train_labels, self.train_predictions)
        print("Train Accuracy:", self.train_acc)
        
        # Calcola la precisione utilizzando la validazione incrociata
        num_folds = 5  # Numero di fold per la validazione incrociata
        scores = cross_val_score(self.calibrated_model, self.train_features, self.train_labels, cv=num_folds)

        # Stampa le precisioni ottenute in ciascun fold
        for fold_num, score in enumerate(scores, start=1):
            print(f"Fold {fold_num} Accuracy: {score:.4f}")

        # Stampa la media delle precisioni su tutti i fold
        average_accuracy = scores.mean()
        print(f"Average Accuracy: {average_accuracy:.4f}")

        model_filename = "model.pkl"
        joblib.dump(self.calibrated_model, model_filename)
        # Salva il vettorizzatore TF-IDF
        vectorizer_filename = "tfidf_vectorizer.pkl"
        joblib.dump(self.tfidf_vectorizer, vectorizer_filename)
        
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
        possible_value = ["OR","MT","GDS","MC","NW","EV","TM","VL","MU","AL"]
        for value, prob in zip(possible_value, predicted_probs):
            print(f"Value {value}: Probability {prob:.4f}")
        
        # Trova l'indice della classe con la massima probabilità
        predicted_class_idx = np.argmax(predicted_probs)
        predicted_context = possible_value[predicted_class_idx]
            
        # Stampa il contesto previsto per l'esempio di frase
        #return predicted_context,predicted_context2[0]
        return predicted_context2[0]
    
ml = ML()