import cv2
import os
from pathlib import Path

def resize_image_with_padding(img, target_width=600, target_height=450):
    height, width = img.shape[:2]

    # Calculer le ratio de redimensionnement pour la largeur et la hauteur
    scaling_factor = min(target_width / width, target_height / height)

    # Nouvelles dimensions
    new_width = int(width * scaling_factor)
    new_height = int(height * scaling_factor)

    # Redimensionner l'image
    resized_image = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)

    # Créer une nouvelle image avec les dimensions cibles et un fond noir
    padded_image = cv2.copyMakeBorder(resized_image,
                                      top=(target_height - new_height) // 2,
                                      bottom=(target_height - new_height) - (target_height - new_height) // 2,
                                      left=(target_width - new_width) // 2,
                                      right=(target_width - new_width) - (target_width - new_width) // 2,
                                      borderType=cv2.BORDER_CONSTANT,
                                      value=[0, 0, 0])  # Couleur noire

    return padded_image

def process_images(source_folder, target_folder):
    # Assurer que le dossier cible existe
    Path(target_folder).mkdir(parents=True, exist_ok=True)

    valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp']

    # Parcourir les fichiers dans le dossier source
    for image_name in os.listdir(source_folder):
        image_path = os.path.join(source_folder, image_name)

        # Vérifier si le fichier est une image valide
        if not any(image_name.lower().endswith(ext) for ext in valid_extensions):
            print(f"Fichier ignoré (pas une image): {image_name}")
            continue

        try:
            img = cv2.imread(image_path)
            if img is None:
                print(f"Impossible de charger l'image: {image_path}")
                continue

            # Si l'image est déjà de la bonne taille, la sauvegarder directement
            height, width = img.shape[:2]
            if width == 24 and height == 24:
                target_path = os.path.join(target_folder, image_name)
                cv2.imwrite(target_path, img)
                print(f"Image déjà à la bonne taille: {image_path}")
                continue

            # Redimensionner et ajouter du padding
            padded_img = resize_image_with_padding(img)

            # Enregistrer l'image redimensionnée dans le dossier cible
            target_path = os.path.join(target_folder, image_name)
            cv2.imwrite(target_path, padded_img)
            print(f"Image redimensionnée avec padding et déplacée: {image_path} -> {target_path}")

        except Exception as e:
            print(f"Erreur lors du traitement de l'image {image_path}: {str(e)}")

# Chemins des dossiers
source_folder = 'n1'
target_folder = 'n'

# Appel de la fonction
process_images(source_folder, target_folder)
