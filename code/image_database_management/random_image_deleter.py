import os
import random

def delete_random_images(folder, fraction=0.1):
    # Liste tous les fichiers dans le dossier
    files = [os.path.join(folder, f) for f in os.listdir(folder) if
             f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff'))]
    # Calcule le nombre de fichiers à supprimer
    num_to_delete = int(len(files) * fraction)

    # Sélectionne aléatoirement des fichiers à supprimer
    files_to_delete = random.sample(files, num_to_delete)

    # Supprime les fichiers sélectionnés
    for file in files_to_delete:
        os.remove(file)
        print(f"Supprimé: {file}")  # Affiche le nom du fichier supprimé


# Utilisation de la fonction
folder = 'n'  # Chemin vers le dossier contenant les images
delete_random_images(folder)
