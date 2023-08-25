from pandas import concat, read_csv
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import nltk
from nltk.corpus import stopwords
import spacy

class ML():
    def __init__(self):
        # Carica i dati dal file CSV
        '''
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
        datasetTraining = concat([
            csv_altro, csv_eventi, csv_manca,csv_giorno_della_settimana,
            csv_meteo,csv_meteo,csv_musica,csv_news,csv_time,
            csv_timer,csv_volume], ignore_index=True)'''
        
        datasetTraining = read_csv("data/data_test/Dataset_Learning.csv",delimiter=";")
        self.sentences = datasetTraining['Frasi'].tolist()
        self.labels = datasetTraining['Contesto'].tolist()


        nltk.download('stopwords')
        nlp = spacy.load("it_core_news_sm")
        stop_words = set(stopwords.words("italian"))

        self.processed_data = []
        for sentence in self.sentences:
            tokens = [token.lemma_ for token in nlp(sentence) if token.text.lower() not in stop_words]
            self.processed_data.append(" ".join(tokens))
            #self.processed_data.append(sentence)
        
        # Mappa etichette testuali a valori numerici
        self.label_mapping = {
            "OR": 1,
            "MT": 2,
            "NW": 3,
            "EV": 4,
            "TM": 5,
            "VL": 6,
            "MU": 7,
            "MC": 8,
            "GDS": 9,
            "AL": 0,
        }

        labels = [self.label_mapping[label_text] for label_text in self.labels]

        # Creazione del tokenizer
        self.tokenizer = Tokenizer()
        self.tokenizer.fit_on_texts(self.processed_data)
        self.word_index = self.tokenizer.word_index
        self.vocab_size = len(self.word_index) + 1

        # Conversione delle frasi in sequenze di interi
        self.sequences = self.tokenizer.texts_to_sequences(self.processed_data)

        # Padding delle sequenze per ottenere lunghezza uniforme
        self.max_sequence_length = max([len(seq) for seq in self.sequences])
        self.padded_sequences = pad_sequences(self.sequences, maxlen=self.max_sequence_length, padding='post')

        # Creazione del modello RNN con strato LSTM
        self.model = tf.keras.Sequential([
            tf.keras.layers.Embedding(self.vocab_size, 50, input_length=self.max_sequence_length),
            tf.keras.layers.LSTM(64),
            tf.keras.layers.Dense(len(self.label_mapping), activation='softmax')  # Cambio di attivazione e output units
        ])

        self.model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        # Addestramento del modello
        self.model.fit(self.padded_sequences, np.array(labels), epochs=10)

    def prevision(self,command:str):
        # Esempio di nuova frase da valutare
        self.new_sentence = [command]

        # Converto la nuova frase in sequenza di interi e la paddo
        self.new_sequence = self.tokenizer.texts_to_sequences(self.new_sentence)
        self.new_padded_sequence = pad_sequences(self.new_sequence, maxlen=self.max_sequence_length, padding='post')

        # Effettuo la predizione del contesto
        self.prediction = self.model.predict(self.new_padded_sequence)

        self.predicted_label = np.argmax(self.prediction)  # Trova l'indice dell'etichetta predetta
        for label, value in self.label_mapping.items():
            if value == self.predicted_label:
                print(f"Prevista etichetta: {label}")
                return label




