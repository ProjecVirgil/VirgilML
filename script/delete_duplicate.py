# Carica il dataset di apprendimento italiano
import pandas as pd
dataset_learning_ita = pd.read_csv("appoggio.csv", delimiter=';').drop_duplicates()
dataset_learning_ita.to_csv("appoggio.csv", sep=";", index=False, encoding='utf-8')
print(len(dataset_learning_ita))