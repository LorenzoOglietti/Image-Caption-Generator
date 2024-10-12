import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import EfficientNet_B0_Weights


# Definizione della classe EncoderCNN, che estende nn.Module
class EncoderCNN(nn.Module):
    # Inizializzazione dell'encoder con il parametro embed_size
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()  # Chiama il costruttore della classe base nn.Module

        # Carica il modello EfficientNet pre-addestrato con i pesi di ImageNet
        self.efficientnet = models.efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)

        # Modifica l'ultimo livello della rete (il classificatore) per produrre embed_size output.
        # Questo output sarà utilizzato come input per il Decoder RNN.
        self.efficientnet.classifier[1] = nn.Linear(self.efficientnet.classifier[1].in_features, embed_size)

        # Congela i pesi del resto della rete EfficientNet, evitando che vengano aggiornati durante il training.
        for param in self.efficientnet.parameters():
            param.requires_grad = False

    # Definizione del metodo forward per il passaggio in avanti del modello
    def forward(self, images):
        features = self.efficientnet(images)  # Estrae le feature dalle immagini utilizzando EfficientNet
        return features  # Restituisce le feature estratte


# Definizione della classe DecoderRNN, che estende nn.Module
class DecoderRNN(nn.Module):
    # Inizializzazione del decoder con i parametri embed_size, hidden_size, vocab_size, num_layers
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers):
        super(DecoderRNN, self).__init__()  # Chiama il costruttore della classe base nn.Module

        # Crea uno strato di embedding che converte le parole in vettori di dimensione embed_size
        self.embed = nn.Embedding(vocab_size, embed_size)

        # Definisce un'architettura LSTM che ha embed_size input, hidden_size stato nascosto, e num_layers livelli.
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers)

        # Definisce uno strato lineare che converte l'output della LSTM in vocab_size,
        # corrispondente alla dimensione del vocabolario (cioè, il numero di parole possibili).
        self.linear = nn.Linear(hidden_size, vocab_size)

    # Definizione del metodo forward per il passaggio in avanti del modello
    def forward(self, features, captions):
        embeddings = self.embed(captions)  # Applica l'embedding sulle didascalie
        embeddings = torch.cat((features.unsqueeze(0), embeddings), dim=0)  # Concatenazione delle feature con gli embedding delle didascalie
        hiddens, _ = self.lstm(embeddings)  # Passa gli embedding attraverso l'LSTM
        outputs = self.linear(hiddens)  # Passa l'output della LSTM attraverso lo strato lineare per predire le parole
        return outputs  # Restituisce le predizioni


# Definizione della classe CNNtoRNN, che combina l'Encoder CNN e il Decoder RNN
class CNNtoRNN(nn.Module):
    # Inizializzazione con embed_size, hidden_size, vocab_size e num_layers
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers):
        super(CNNtoRNN, self).__init__()  # Chiama il costruttore della classe base nn.Module

        # Inizializza l'encoder CNN con embed_size
        self.encoderCNN = EncoderCNN(embed_size)

        # Inizializza il decoder RNN con embed_size, hidden_size, vocab_size e num_layers
        self.decoderRNN = DecoderRNN(embed_size, hidden_size, vocab_size, num_layers)

    # Definizione del metodo forward per passare immagini e didascalie attraverso il modello
    def forward(self, images, captions):
        features = self.encoderCNN(images)  # Estrae le feature dalle immagini utilizzando l'Encoder CNN
        outputs = self.decoderRNN(features, captions)  # Passa le feature e le didascalie attraverso il Decoder RNN
        return outputs  # Restituisce l'output finale (predizioni delle parole)

    # Metodo per generare una didascalia a partire da un'immagine
    def caption_image(self, image, vocabulary, max_length=50):
        result_caption = []  # Lista per salvare la didascalia generata

        # Disattiva il calcolo del gradiente poiché non è richiesto durante la generazione
        with torch.no_grad():
            x = self.encoderCNN(image).unsqueeze(0)  # Estrae le feature dall'immagine e aggiunge una dimensione batch
            states = None  # Inizializza gli stati nascosti dell'LSTM come None

            # Ciclo per generare la didascalia, un token alla volta, fino a max_length
            for _ in range(max_length):
                outputs, states = self.decoderRNN.lstm(x, states)  # Passa le feature attraverso l'LSTM e ottiene l'output e lo stato nascosto
                output = self.decoderRNN.linear(outputs.squeeze(0))  # Passa l'output della LSTM attraverso lo strato lineare per ottenere le predizioni
                predicted = output.argmax(1)  # Seleziona la parola con la massima probabilità (argmax)
                result_caption.append(predicted.item())  # Aggiunge la parola alla lista della didascalia
                x = self.decoderRNN.embed(predicted).unsqueeze(0)  # Usa la parola predetta come input per il passo successivo

                # Interrompe il ciclo se la parola predetta è il token di fine frase "<EOS>"
                if vocabulary.itos[predicted.item()] == "<EOS>":
                    break

        # Restituisce la didascalia finale come una lista di parole
        return [vocabulary.itos[idx] for idx in result_caption]
