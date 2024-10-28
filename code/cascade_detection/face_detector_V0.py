import cv2
import os

# Charger le modèle en cascade formé
cascade_path = "cascade.xml"  # Le fichier cascade est dans le même dossier que le script
face_cascade = cv2.CascadeClassifier(cascade_path)

# Dossier contenant les images de test
image_folder = "image_test"  # Assurez-vous que ce dossier existe et contient les images

# Parcourir chaque image du dossier
for image_name in os.listdir(image_folder):
    image_path = os.path.join(image_folder, image_name)
    image = cv2.imread(image_path)

    # Vérifier si l'image est correctement chargée
    if image is None:
        print(f"Erreur lors du chargement de l'image: {image_path}")
        continue

    # Convertir l'image en niveaux de gris
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Effectuer la détection d'objets
    detected_objects = face_cascade.detectMultiScale(
        gray_image,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )

    # Afficher le nombre d'objets détectés
    print(f"{len(detected_objects)} objets détectés dans {image_name}")

    # Dessiner des rectangles autour des objets détectés
    for (x, y, w, h) in detected_objects:
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Afficher l'image avec les objets détectés
    cv2.imshow(f'Image: {image_name}', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
