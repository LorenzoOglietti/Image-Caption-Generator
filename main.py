import os  # Per gestire i percorsi dei file
import torch
import tkinter as tk  # Libreria per la creazione di interfacce grafiche
from tkinter import filedialog  # Finestra di dialogo per la selezione di file
from PIL import Image, ImageTk  # Per la manipolazione delle immagini e la visualizzazione nella GUI
import torchvision.transforms as transforms  # Per le trasformazioni delle immagini
from model import CNNtoRNN  # Importa il modello CNNtoRNN dal file model.py
from get_loader import get_loader  # Importa la funzione get_loader per caricare il dataset e il vocabolario

# Trasformazioni per l'immagine
transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),  # Ridimensiona l'immagine a 224x224 pixel
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),  # Modifica casualmente luminosità, contrasto, saturazione e tono
        transforms.ToTensor(),  # Converte l'immagine in un tensore PyTorch
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalizza il tensore secondo i valori di ImageNet
    ]
)


def preprocess_image(image_path):
    """Preprocessa l'immagine: apre, converte in RGB, applica le trasformazioni e aggiunge una dimensione batch."""
    image = Image.open(image_path).convert("RGB")  # Carica e converte l'immagine in RGB
    return transform(image).unsqueeze(0)  # Applica le trasformazioni e aggiunge una dimensione batch


def select_image():
    """Permette all'utente di selezionare un'immagine e genera la didascalia usando il modello."""
    # Apre una finestra di dialogo per selezionare un'immagine
    file_path = filedialog.askopenfilename(
        filetypes=[("Image Files", ".jpg;.jpeg;*.png")],  # Tipi di file accettati
        title="Select an Image"  # Titolo della finestra di dialogo
    )

    if file_path:
        # Estrai il nome del file dall'intero percorso
        file_name = os.path.basename(file_path)

        # Carica e preprocessa l'immagine
        image_tensor = preprocess_image(file_path)

        # Carica l'immagine per la visualizzazione nella GUI
        image_pil = Image.open(file_path).resize((300, 300))  # Ridimensiona l'immagine per la visualizzazione nella GUI

        model.eval()  # Imposta il modello in modalità valutazione (evaluation mode)

        with torch.no_grad():  # Disabilita il calcolo dei gradienti durante l'inferenza
            caption = model.caption_image(image_tensor.to(device), dataset.vocab)  # Genera la didascalia per l'immagine
        caption_text = f"Image: {file_name}\n\nCaption:\n{' '.join(caption)}"  # Format del testo della didascalia

        # Mostra l'immagine e la didascalia nella GUI
        img_tk = ImageTk.PhotoImage(image_pil)  # Converti l'immagine per l'uso in Tkinter
        image_label.config(image=img_tk)  # Mostra l'immagine nella label
        image_label.image = img_tk  # Salva un riferimento all'immagine per evitare che venga eliminata dal garbage collector
        caption_label.config(text=caption_text)  # Aggiorna il testo della didascalia nella GUI

        # Cambia lo sfondo delle label per un tocco estetico
        caption_label.config(bg="#F0F0F0", font=("Arial", 12, "italic"))  # Cambia lo stile della didascalia
        image_label.config(bg="white", relief="solid", bd=2)  # Aggiunge un bordo bianco all'immagine


# Configura la finestra principale dell'interfaccia grafica
root = tk.Tk()  # Inizializza la finestra principale
root.title("Image Caption Generator")  # Titolo della finestra
root.geometry("500x600")  # Imposta le dimensioni della finestra

# Aggiungi uno sfondo e un titolo per la GUI
root.configure(bg="#ECECEC")  # Imposta il colore di sfondo della finestra
title_label = tk.Label(root, text="Image Caption Generator", font=("Helvetica", 16, "bold"), bg="#ECECEC")
title_label.pack(pady=20)  # Posiziona il titolo con un po' di margine

# Aggiungi un pulsante per selezionare un'immagine
button = tk.Button(root, text="Select an Image", command=select_image, font=("Helvetica", 12), bg="#4CAF50", fg="white",
                   padx=10, pady=5)  # Pulsante per caricare un'immagine
button.pack(pady=10)  # Posiziona il pulsante

# Aggiungi una label per visualizzare l'immagine
image_label = tk.Label(root, bg="#ECECEC")  # Label per mostrare l'immagine
image_label.pack(pady=10)

# Aggiungi una label per visualizzare la didascalia
caption_label = tk.Label(root, text="", wraplength=450, justify="left", bg="#ECECEC", font=("Arial", 12), anchor="w")
caption_label.pack(pady=10, padx=10)  # Posiziona la label della didascalia

# Configura il dispositivo (GPU se disponibile, altrimenti CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Usa GPU se disponibile, altrimenti CPU

# Inizializza e carica il modello
embed_size = 256  # Dimensione dell'embedding
hidden_size = 256  # Dimensione dello stato nascosto del modello RNN
num_layers = 1  # Numero di livelli dell'RNN

# Carica il dataset per accedere al vocabolario
_, dataset = get_loader(
    root_folder="flickr8k_images/Images",  # Cartella delle immagini
    annotation_file="flickr8k_images/captions.txt",  # File delle annotazioni con le didascalie
    transform=transform,  # Trasformazioni da applicare alle immagini
    num_workers=8,  # Numero di worker per il caricamento dei dati
)

# Inizializza il modello CNNtoRNN
model = CNNtoRNN(embed_size, hidden_size, len(dataset.vocab), num_layers).to(device)

# Carica i pesi del modello addestrato
checkpoint = torch.load("my_checkpoint.pth.tar", map_location=device, weights_only=True)  # Carica il checkpoint con i pesi del modello
model.load_state_dict(checkpoint['state_dict'])  # Carica lo stato del modello dal checkpoint

# Avvia la GUI
root.mainloop()  # Esegue il loop principale della finestra Tkinter
