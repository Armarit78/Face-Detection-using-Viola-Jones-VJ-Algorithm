import cv2

# Charger le modèle en cascade formé
cascade_path = "cascade.xml"  # Le fichier cascade est dans le même dossier que le script
face_cascade = cv2.CascadeClassifier(cascade_path)

# Initialiser la capture vidéo depuis la webcam (index 0 correspond à la première webcam)
video_capture = cv2.VideoCapture(0)

if not video_capture.isOpened():
    print("Erreur : Impossible d'accéder à la webcam.")
else:
    while True:
        # Capturer une image depuis la webcam
        ret, frame = video_capture.read()

        if not ret:
            print("Erreur lors de la capture de l'image.")
            break

        # Convertir l'image en niveaux de gris
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Effectuer la détection d'objets
        detected_objects = face_cascade.detectMultiScale(
            gray_frame,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )

        # Dessiner des rectangles autour des objets détectés
        for (x, y, w, h) in detected_objects:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Afficher le flux vidéo avec les objets détectés
        cv2.imshow('Détection avec Webcam', frame)

        # Sortir de la boucle en appuyant sur 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Libérer la capture vidéo et fermer les fenêtres
video_capture.release()
cv2.destroyAllWindows()
