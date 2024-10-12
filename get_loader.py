import os  # Per gestire i percorsi dei file
import pandas as pd  # Per caricare e manipolare il file delle annotazioni
import spacy  # Per la tokenizzazione del testo
import torch
from PIL import Image  # Per caricare le immagini
from torch.nn.utils.rnn import pad_sequence  # Per il padding delle sequenze nei batch
from torch.utils.data import DataLoader, Dataset  # Dataset e DataLoader di PyTorch

# Carica il tokenizer di spaCy per l'inglese
spacy_eng = spacy.load('en_core_web_sm')



class Vocabulary:
    def __init__(self, freq_threshold):
        # Dizionari per la mappatura tra ID e token e viceversa
        self.itos = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}  # ID a token
        self.stoi = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2, "<UNK>": 3}  # Token a ID
        self.freq_threshold = freq_threshold  # Soglia di frequenza per includere una parola nel vocabolario

    def __len__(self):
        # Restituisce il numero totale di parole nel vocabolario
        return len(self.itos)

    @staticmethod
    def tokenizer_eng(text):
        # Tokenizza il testo utilizzando spaCy e restituisce una lista di parole in minuscolo
        return [tok.text.lower() for tok in spacy_eng.tokenizer(text)]

    def build_vocabulary(self, sentence_list):
        # Costruisce il vocabolario basato sulle frasi fornite
        frequencies = {}  # Dizionario per tenere traccia della frequenza di ciascuna parola
        idx = 4  # Inizia a contare le parole dal primo indice disponibile dopo <PAD>, <SOS>, <EOS>, <UNK>

        # Conta le frequenze di tutte le parole
        for sentence in sentence_list:
            for word in self.tokenizer_eng(sentence):
                frequencies[word] = frequencies.get(word, 0) + 1  # Aumenta il contatore per ogni parola

        # Aggiungi al vocabolario solo le parole che superano la soglia di frequenza
        for word, freq in frequencies.items():
            if freq >= self.freq_threshold:  # Verifica se la frequenza supera la soglia
                self.stoi[word] = idx  # Assegna un nuovo ID alla parola
                self.itos[idx] = word  # Aggiungi la parola al vocabolario inverso
                idx += 1  # Incrementa l'indice per la parola successiva

    def numericalize(self, text):
        # Converte il testo in una sequenza di ID numerici basati sul vocabolario
        tokenized_text = self.tokenizer_eng(text)  # Tokenizza il testo
        return [
            self.stoi[token] if token in self.stoi else self.stoi["<UNK>"]  # Usa <UNK> se la parola non è nel vocabolario
            for token in tokenized_text
        ]


class FlickrDataset(Dataset):
    def __init__(self, root_dir, captions_file, transform=None, freq_threshold=5):
        # Inizializza il dataset di Flickr
        self.root_dir = root_dir  # Directory delle immagini
        self.df = pd.read_csv(captions_file)  # Carica il file CSV delle annotazioni
        self.transform = transform  # Trasformazioni da applicare alle immagini

        # Ottieni colonne delle immagini e delle didascalie
        self.imgs = self.df["image"]  # Nomi dei file immagine
        self.captions = self.df["caption"]  # Didascalie

        # Inizializza il vocabolario e costruiscilo
        self.vocab = Vocabulary(freq_threshold)  # Istanzia il vocabolario con la soglia di frequenza
        self.vocab.build_vocabulary(self.captions.tolist())  # Costruisce il vocabolario dalle didascalie

    def __len__(self):
        # Restituisce il numero di elementi nel dataset
        return len(self.df)

    def __getitem__(self, index):
        # Ottiene l'immagine e la didascalia per un dato indice
        caption = self.captions[index]  # Didascalia associata all'immagine
        img_id = self.imgs[index]  # Nome del file immagine
        img = Image.open(os.path.join(self.root_dir, img_id)).convert("RGB")  # Carica l'immagine e la converte in RGB

        if self.transform is not None:
            img = self.transform(img)  # Applica le trasformazioni all'immagine (se specificate)

        # Converte la didascalia in una sequenza di ID numerici, aggiungendo i token <SOS> e <EOS>
        numericalized_caption = [self.vocab.stoi["<SOS>"]]  # Inizia la didascalia con il token <SOS>
        numericalized_caption += self.vocab.numericalize(caption)  # Numerizza la didascalia
        numericalized_caption.append(self.vocab.stoi["<EOS>"])  # Aggiunge il token <EOS> alla fine

        return img, torch.tensor(numericalized_caption)  # Restituisce l'immagine e la didascalia numerica


class MyCollate:
    def __init__(self, pad_idx):
        # Inizializza l'indice di padding
        self.pad_idx = pad_idx

    def __call__(self, batch):
        # Aggrega un batch di immagini e didascalie
        imgs = [item[0].unsqueeze(0) for item in batch]  # Aggiungi una dimensione batch alle immagini
        imgs = torch.cat(imgs, dim=0)  # Concatenate le immagini in un unico tensor
        targets = [item[1] for item in batch]  # Estrai le didascalie numeriche
        targets = pad_sequence(targets, batch_first=True, padding_value=self.pad_idx)  # Padding delle didascalie

        return imgs, targets  # Restituisce il batch di immagini e didascalie con padding


def get_loader(root_folder,annotation_file,transform,batch_size=64,num_workers=8,shuffle=True,pin_memory=True,):
    # Crea un DataLoader per il dataset di Flickr
    dataset = FlickrDataset(root_folder, annotation_file, transform=transform)  # Istanzia il dataset

    pad_idx = dataset.vocab.stoi["<PAD>"]  # Ottieni l'indice di padding dal vocabolario

    # Definisce il DataLoader con i parametri specificati
    loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,  # Dimensione del batch
        num_workers=num_workers,  # Numero di worker per il caricamento dei dati
        shuffle=shuffle,  # Mescola i dati ad ogni epoch
        pin_memory=pin_memory,  # Ottimizzazione della memoria
        collate_fn=MyCollate(pad_idx=pad_idx),  # Usa la funzione di collate personalizzata per il padding
    )

    return loader, dataset  # Restituisce il DataLoader e il dataset
