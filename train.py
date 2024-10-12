import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import random_split, DataLoader
from tqdm import tqdm  # Barra di progresso per il monitoraggio dell'addestramento

from get_loader import get_loader  # Funzione per caricare il dataset e il vocabolario
from model import CNNtoRNN  # Modello CNNtoRNN per generare le didascalie

def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    """Salva lo stato attuale del modello e dell'ottimizzatore."""
    print("=> Saving checkpoint")
    torch.save(state, filename)


def load_checkpoint(checkpoint, model, optimizer):
    """Carica lo stato salvato del modello e dell'ottimizzatore."""
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])  # Carica i pesi del modello
    optimizer.load_state_dict(checkpoint["optimizer"])  # Carica lo stato dell'ottimizzatore
    step = checkpoint["step"]  # Carica il passo (numero di batch processati)
    return step


def train():
    """Funzione principale per l'addestramento del modello."""

    # Definizione delle trasformazioni da applicare alle immagini
    transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(224),  # Ridimensiona casualmente l'immagine
            transforms.RandomHorizontalFlip(),  # Capovolge orizzontalmente l'immagine con probabilità 0.5
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),  # Modifica casualmente i parametri di luminosità, contrasto, ecc.
            transforms.ToTensor(),  # Converte l'immagine in un tensore PyTorch
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalizza l'immagine con i valori medi e deviazione standard di ImageNet
        ]
    )

    # Caricamento del dataset e creazione del DataLoader
    loader, dataset = get_loader(
        root_folder="flickr8k_images/Images",  # Percorso della cartella delle immagini
        annotation_file="flickr8k_images/captions.txt",  # File delle annotazioni (didascalie)
        transform=transform,  # Trasformazioni da applicare alle immagini
        batch_size=64,  # Dimensione del batch
        num_workers=8,  # Numero di worker per il caricamento dei dati
        shuffle=True,  # Mescola i dati a ogni epoca
        pin_memory=True,  # Mantiene in memoria il caricamento dei dati (utile per la GPU)
    )

    # Suddivisione del dataset in training e validation set
    train_size = int(0.8 * len(dataset))  # 80% dei dati per il training
    val_size = len(dataset) - train_size  # 20% dei dati per la validazione
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])  # Divide il dataset

    # Creazione dei DataLoader per il training e la validazione
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=8, collate_fn=loader.collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=8, collate_fn=loader.collate_fn)

    # Configurazione del dispositivo (GPU se disponibile, altrimenti CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)  # Stampa il dispositivo utilizzato
    load_model = False  # Se True, carica un modello pre-addestrato
    save_model = True   # Se True, salva il modello dopo ogni epoca

    # Hyperparametri
    embed_size = 256  # Dimensione dell'embedding
    hidden_size = 256  # Dimensione dello stato nascosto dell'LSTM
    vocab_size = len(dataset.vocab)  # Dimensione del vocabolario (numero di parole nel vocabolario)
    num_layers = 1  # Numero di strati LSTM
    learning_rate = 0.001  # Tasso di apprendimento
    num_epochs = 10  # Numero di epoche di addestramento
    step = 0  # Passo per tracciare il numero di batch processati

    # Parametri per l'early stopping
    patience = 3  # Numero massimo di epoche senza miglioramento prima di interrompere l'addestramento
    best_val_loss = float('inf')  # Migliore perdita di validazione registrata
    epochs_without_improvement = 0  # Contatore delle epoche senza miglioramento

    # Inizializzazione del modello, della funzione di perdita e dell'ottimizzatore
    model = CNNtoRNN(embed_size, hidden_size, vocab_size, num_layers)  # Crea il modello
    model = model.to(device)  # Sposta il modello sul dispositivo (GPU/CPU)

    criterion = nn.CrossEntropyLoss(ignore_index=dataset.vocab.stoi["<PAD>"])  # Funzione di perdita CrossEntropy, ignorando il token di padding
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)  # Ottimizzatore AdamW
    scaler = torch.cuda.amp.GradScaler()  # Scaler per la precisione mista (Mixed Precision)

    # Carica un checkpoint se richiesto
    if load_model:
        step = load_checkpoint(torch.load("my_checkpoint.pth.tar"), model, optimizer)

    model.train()  # Imposta il modello in modalità addestramento

    # Apri un file per salvare le perdite di training e validazione
    with open('training_validation_losses.txt', 'w') as f:
        f.write("Epoch,Training Loss,Validation Loss\n")  # Scrive l'intestazione nel file

        for epoch in range(num_epochs):  # Ciclo su tutte le epoche
            # Fase di training
            model.train()  # Assicura che il modello sia in modalità training
            train_loss = 0  # Inizializza la perdita di training

            for idx, (imgs, captions) in tqdm(enumerate(train_loader), total=len(train_loader), leave=True):
                imgs = imgs.to(device)  # Sposta le immagini sul dispositivo
                captions = captions.to(device)  # Sposta le didascalie sul dispositivo

                optimizer.zero_grad()  # Azzeramento dei gradienti dell'ottimizzatore
                with torch.cuda.amp.autocast():
                    outputs = model(imgs, captions[:-1])  # Passa le immagini e le didascalie (escludendo l'ultimo token) attraverso il modello
                    loss = criterion(outputs.reshape(-1, outputs.shape[2]), captions.reshape(-1))  # Calcola la perdita

                scaler.scale(loss).backward()  # Esegue il backpropagation con lo scaling della perdita
                scaler.step(optimizer)  # Aggiorna i pesi del modello
                scaler.update()  # Aggiorna lo scaler

                train_loss += loss.item()  # Accumula la perdita di training
                step += 1  # Incrementa il contatore dei passi

            train_loss /= len(train_loader)  # Calcola la perdita media di training
            print(f"Epoch [{epoch + 1}/{num_epochs}], Training Loss: {train_loss:.4f}")

            # Fase di validazione
            model.eval()  # Imposta il modello in modalità validazione
            val_loss = 0  # Inizializza la perdita di validazione

            with torch.no_grad():  # Disabilita il calcolo dei gradienti durante la validazione
                for imgs, captions in tqdm(val_loader, total=len(val_loader), leave=True):
                    imgs = imgs.to(device)  # Sposta le immagini sul dispositivo
                    captions = captions.to(device)  # Sposta le didascalie sul dispositivo

                    with torch.cuda.amp.autocast():
                        outputs = model(imgs, captions[:-1])  # Passa le immagini e le didascalie attraverso il modello
                        loss = criterion(outputs.reshape(-1, outputs.shape[2]), captions.reshape(-1))  # Calcola la perdita

                    val_loss += loss.item()  # Accumula la perdita di validazione

            val_loss /= len(val_loader)  # Calcola la perdita media di validazione
            print(f"Epoch [{epoch + 1}/{num_epochs}], Validation Loss: {val_loss:.4f}")

            # Salva le perdite di training e validazione nel file di testo
            f.write(f"{epoch + 1},{train_loss:.4f},{val_loss:.4f}\n")

            # Controlla l'early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss  # Aggiorna la migliore perdita di validazione
                epochs_without_improvement = 0  # Reset del contatore di epoche senza miglioramento
                if save_model:
                    checkpoint = {
                        "state_dict": model.state_dict(),  # Stato del modello
                        "optimizer": optimizer.state_dict(),  # Stato dell'ottimizzatore
                        "step": step,  # Numero di passi
                    }
                    save_checkpoint(checkpoint)  # Salva il checkpoint del modello
            else:
                epochs_without_improvement += 1  # Incrementa il contatore di epoche senza miglioramento

            # Interrompe l'addestramento se il numero massimo di epoche senza miglioramento è stato raggiunto
            if epochs_without_improvement >= patience:
                print(f"Early stopping triggered after {epoch + 1} epochs")
                break

    # Salva il modello finale se richiesto
    if save_model:
        save_checkpoint({
            "state_dict": model.state_dict(),  # Stato finale del modello
            "optimizer": optimizer.state_dict(),  # Stato finale dell'ottimizzatore
            "step": step,  # Numero finale di passi
        })


if __name__ == "__main__":
    train()  # Avvia l'addestramento
