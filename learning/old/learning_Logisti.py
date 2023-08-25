import numpy as np
from pandas import concat, read_csv
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils import shuffle
from sklearn.feature_extraction.text import TfidfVectorizer

class ML:
    def __init__(self) -> None:
        input_dirty = "data/data_test/Dataset_LearningW.csv"
        testing_dirty = "data/data_test/Dataset_Testing.csv"
        
        csv_altro = read_csv("data/data_final/altro.csv",delimiter=";")
        csv_eventi = read_csv("data/data_final/eventi.csv",delimiter=";")
        csv_giorno_della_settimana = read_csv("data/data_final/giorni_della_settimana.csv",delimiter=";")
        csv_manca = read_csv("data/data_final/manca.csv",delimiter=";")
        csv_meteo = read_csv("data/data_final/meteo.csv",delimiter=";")
        csv_musica = read_csv("data/data_final/musica.csv",delimiter=";")
        csv_news = read_csv("data/data_final/news.csv",delimiter=";")
        csv_time = read_csv("data/data_final/time.csv",delimiter=";")
        csv_timer = read_csv("data/data_final/timer.csv",delimiter=";")
        csv_volume = read_csv("data/data_final/volume.csv",delimiter=";")
                
        # Carica i dataset
        self.datasetTraining = concat([
            csv_altro, csv_eventi, csv_manca,csv_giorno_della_settimana,
            csv_meteo,csv_meteo,csv_musica,csv_news,csv_time,
            csv_timer,csv_volume], ignore_index=True)
        # Carica i dataset
        #self.datasetTraing = read_csv(input_dirty, sep=';', encoding='utf-8')
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

        # Addestra il modello LogisticRegression
        self.lr_model = LogisticRegression(max_iter=1000, random_state=42)
        self.lr_model.fit(self.train_features, self.train_labels)

        # Valuta le prestazioni del modello sul set di test
        self.predictions = self.lr_model.predict(self.test_features)
        self.accuracy = accuracy_score(self.test_labels, self.predictions)
        print("Test Accuracy:", self.accuracy)

        # Stampa il classification report e la matrice di confusione
        print("Classification Report:")
        print(classification_report(self.test_labels, self.predictions, zero_division=1))  # Imposta zero_division=1
        print("Confusion Matrix:")
        print(confusion_matrix(self.test_labels, self.predictions))

    def prevision(self, command: str):
        
        # Esempio di frase da testare
        example_sentence = command
        # Trasforma l'esempio di frase in una rappresentazione TF-IDF
        example_features = self.tfidf_vectorizer.transform([example_sentence])
        # Fai una previsione utilizzando il modello
        predicted_context2 = self.lr_model.predict(example_features)
        # Fai una previsione utilizzando il modello LogisticRegression
        predicted_probs = self.lr_model.predict_proba(example_features)[0]
        
        # Stampa le probabilità previste per ciascuna classe
        possible_value = ["OR","MT","GDS","MC","NW","EV","TM","VL","MU","AL"]
        for value, prob in zip(possible_value, predicted_probs):
            print(f"Value {value}: Probability {prob:.4f}")
        
        # Trova l'indice della classe con la massima probabilità
        predicted_class_idx = np.argmax(predicted_probs)
        predicted_context = possible_value[predicted_class_idx]
            
        # Stampa il contesto previsto per l'esempio di frase
        #return predicted_context, predicted_context2[0]
        return predicted_context2[0]

ml = ML()
print(ml.prevision("Ciao"))