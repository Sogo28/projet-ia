import cv2
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import pickle
import os

class HOGFaceRecognition:
    def __init__(self):
        # Initialisation du détecteur de visages
        self.face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        # Initialisation des paramètres HOG
        self.hog = cv2.HOGDescriptor()
        # Initialisation du classifieur
        self.classifier = SVC(kernel='linear', probability=True)

    def extract_hog_features(self, image):
        """Extrait les caractéristiques HOG d'une image."""
        # Conversion en niveaux de gris
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Détection des visages
        faces = self.face_detector.detectMultiScale(gray, 1.1, 4)
        if len(faces) == 0:
            return None
        
        # Pour le premier visage détecté
        x, y, w, h = faces[0]
        face_roi = gray[y:y+h, x:x+w]
        
        # Redimensionnement pour normalisation
        face_roi = cv2.resize(face_roi, (64, 128))
        
        # Calcul des caractéristiques HOG
        hog_features = self.hog.compute(face_roi)
        
        return hog_features.flatten()

    def train(self, dataset_path):
        """Entraîne le système sur un jeu de données."""
        # Vérifier si les caractéristiques ont déjà été sauvegardées
        if os.path.exists('features.pkl'):
            print("Chargement des caractéristiques extraites depuis le fichier sauvegardé...")
            with open('features.pkl', 'rb') as f:
                X, y = pickle.load(f)
        else:
            X = []  # Caractéristiques
            y = []  # Étiquettes (noms des personnes)
            # Parcours du dataset et traitement des images comme dans l'exemple précédent
        
            # Parcours du dataset
            for person_name in os.listdir(dataset_path):
                person_dir = os.path.join(dataset_path, person_name)
                if not os.path.isdir(person_dir):
                    continue
                
                print(f"Traitement des images de {person_name}...")
            
                # Traitement de chaque image
                for image_name in os.listdir(person_dir):
                    image_path = os.path.join(person_dir, image_name)
                    image = cv2.imread(image_path)
                    if image is None:
                        continue
                    
                    # Extraction des caractéristiques HOG
                    features = self.extract_hog_features(image)
                    if features is not None:
                        X.append(features)
                        y.append(person_name)

            # Sauvegarder les caractéristiques extraites
            with open('features.pkl', 'wb') as f:
                pickle.dump((X, y), f)
        
        # Division en ensembles d'entraînement et de test
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Entraînement du classifieur
        self.classifier.fit(X_train, y_train)
        
        # Évaluation
        y_pred = self.classifier.predict(X_test)
        
        # Calcul et affichage des métriques
        print("Rapport de classification :")
        print(classification_report(y_test, y_pred))
        
        print("\nMatrice de confusion :")
        print(confusion_matrix(y_test, y_pred))
        
        return X_test, y_test, y_pred

    def predict(self, image):
        """Prédit l'identité d'une personne sur une image."""
        features = self.extract_hog_features(image)
        if features is None:
            return None
            
        # Prédiction avec probabilités
        pred = self.classifier.predict([features])[0]
        proba = self.classifier.predict_proba([features])[0]
        confidence = max(proba)
        
        return pred, confidence

    def real_time_recognition(self):
        """Reconnaissance en temps réel via la webcam."""
        cap = cv2.VideoCapture(1)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Détection des visages
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_detector.detectMultiScale(gray, 1.1, 4)
            
            # Pour chaque visage détecté
            for (x, y, w, h) in faces:
                # Prédiction
                result = self.predict(frame)
                if result is not None:
                    name, confidence = result
                    
                    # Affichage
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    cv2.putText(frame, f"{name} ({confidence:.2f})", 
                              (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 
                              0.5, (0, 255, 0), 2)
            
            cv2.imshow('Reconnaissance Faciale HOG', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        cap.release()
        cv2.destroyAllWindows()

    def test_single_image(self, image_path):
        """Test de la reconnaissance faciale sur une seule image."""
        # Chargement de l'image
        image = cv2.imread(image_path)
        
        if image is None:
            print("L'image n'a pas pu être chargée.")
            return
        
        # Détection et prédiction sur l'image
        result = self.predict(image)
        
        if result is not None:
            name, confidence = result
            if confidence > 0.5:
                print(f"Prédiction : {name} avec une confiance de {confidence:.2f}")
            else:
                print("Aucune prédiction fiable.")
        else:
            print("Aucun visage détecté.")
        
if __name__ == "__main__":
    # Initialisation du système
    system = HOGFaceRecognition()
    
    # Entraînement
    X_test, y_test, y_pred = system.train("./dataset")
    
    # Lancement de la reconnaissance en temps réel
    # system.real_time_recognition()
    system.test_single_image("image.png")