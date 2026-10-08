# Reconnaissance des expressions faciales — Deep Learning

Projet de Deep Learning utilisant le dataset **FER2013** pour reconnaître 7 expressions faciales à l'aide d'un réseau de neurones convolutif (CNN).

Le projet comprend également :

* La détection de plusieurs visages avec YOLOv8n-Face.
* La classification des expressions faciales sur des images et des vidéos.
* Des expérimentations pour améliorer les performances du modèle.

**Résultat final : 59,57 % d'accuracy sur le jeu de test FER2013** (macro-F1 : 0,5406 ; weighted F1 : 0,5901).
Modèle retenu : CNN à 3 blocs Conv/Pool avec **data augmentation** (Exp. 1), choisi sur la validation (58,18 %) sans consulter le test.

## 🎬 Démonstration vidéo

Les vidéos ne sont pas hébergées sur GitHub en raison de leur taille.

* ▶️ [Vidéo originale](https://youtu.be/-KZI9_dtdz8)
* 🤖 [Vidéo analysée avec YOLO + CNN](https://youtu.be/qQCIpFGsurE)

Vidéo analysée : 19,6 s, 25 fps, 490 images, 1 723 instances de visages détectées (Joie : 1 239, Tristesse : 387, Neutre : 57, Colère : 40).

## 🛠️ Technologies

Python · TensorFlow/Keras · YOLOv8n-Face · OpenCV · Scikit-learn

## 📊 Dataset

**FER2013** — 35 887 images de visages en niveaux de gris (48×48), réparties en 7 classes : colère, dégoût, peur, joie, tristesse, surprise et neutre.

Source : challenge « Facial Expression Recognition » (2013), copie publique sur [Kaggle](https://www.kaggle.com/datasets/msambare/fer2013). Références d'origine : Pierre-Luc Carrier et Aaron Courville ; Goodfellow et al. (2013).

| Expression | Images |
|---|---|
| Colère (Angry) | 4 953 |
| Dégoût (Disgust) | 547 |
| Peur (Fear) | 5 121 |
| Joie (Happy) | 8 989 |
| Tristesse (Sad) | 6 077 |
| Surprise | 4 002 |
| Neutre | 6 198 |
| **Total** | **35 887** |

Le dataset est déséquilibré : *Disgust* ne représente que 1,52 % des images (rapport 16,4× avec la classe majoritaire *Happy*).

Split officiel : 28 709 images d'entraînement, 3 589 de validation (`PublicTest`) et 3 589 de test (`PrivateTest`, utilisé une seule fois après le choix du modèle). Prétraitement : conversion en tenseurs 48×48×1, normalisation entre 0 et 1, labels entiers 0–6.

## 🧠 Pipeline

Image / vidéo → YOLOv8n-Face (détection) → recadrage des visages (marge de 12 %) → prétraitement (gris, 48×48, /255) → CNN → expression + score Softmax

## 🏗️ Architecture du CNN de référence

Input 48×48×1 → [Conv 3×3 (32) → MaxPool] → [Conv 3×3 (64) → MaxPool] → [Conv 3×3 (128) → MaxPool] → Flatten (4 608) → Dense 128 (ReLU) → Dropout 0,30 → Dense 7 (Softmax)

683 527 paramètres. Optimiseur Adam (lr 1e-3), loss `sparse_categorical_crossentropy`, batch size 128, 40 époques max avec Early Stopping (patience 6), ReduceLROnPlateau (facteur 0,5) et ModelCheckpoint sur `val_loss`.

## 📁 Contenu du notebook

| Partie | Contenu |
|---|---|
| 1 | Étude, préparation et séparation des données |
| 2 | Modèle de référence (réseau dense) |
| 3 | Construction du CNN (3 blocs Conv/Pool) |
| 4 | Entraînement (Early Stopping, ReduceLROnPlateau, ModelCheckpoint) |
| 5 | Évaluation : matrice de confusion, analyse des erreurs |
| 6 | 3 expériences : data augmentation, dropout 0,5, learning rate 3e-4 |
| 7 | Transfer Learning (MobileNetV2) et fine-tuning |
| 8 | Détection multi-visages avec YOLO + classification |
| 9 | Analyse d'une vidéo |

## 📈 Résultats

| Modèle | Modification | Val accuracy |
|---|---|---|
| Baseline dense | Flatten + Dense 256, sans convolution | 0,3683 |
| CNN de référence | 3 blocs Conv/Pool, dropout 0,30, Adam 1e-3 | 0,5684 |
| **Exp. 1 — Data augmentation** | flip + rotation 0,05 + zoom 0,08 | **0,5818** |
| Exp. 2 — Dropout 0,50 | dropout 0,50 | 0,5486 |
| Exp. 3 — Learning rate 3e-4 | lr 3e-4 | 0,5606 |
| MobileNetV2 (backbone gelé) | entrée 96×96×3 | 0,4349 |
| MobileNetV2 + fine-tuning | 20 dernières couches dégelées | 0,4848 |

**Test final (modèle Exp. 1) : 59,57 % d'accuracy.**
Meilleurs rappels : Happy 0,8248 ; Neutral 0,6677 ; Surprise 0,6635. Rappel le plus faible : Disgust 0,2182.
Confusions dominantes : Sad → Neutral (139), Fear → Sad (97), Fear → Angry (95).

## 🔎 Détection de visages (YOLO)

Détecteur : YOLOv8n-Face (`yolov8n-face-lindevs.pt`, entraîné sur WIDERFace), seuil de confiance 0,35, taille d'inférence 640. Le mAP du détecteur n'a pas été calculé : FER2013 ne fournit pas de bounding boxes annotées permettant d'évaluer la détection.

## ▶️ Exécution

1. Ouvrir le notebook dans Google Colab.
2. Activer un runtime GPU (*Exécution → Modifier le type d'exécution*).
3. Lancer *Exécuter tout*. Le dataset et les poids YOLO sont téléchargés automatiquement.
4. Pour les démos, renseigner `DEMO_IMAGE_PATH` et `DEMO_VIDEO_PATH`.

## ⚠️ Limites

* Faible résolution (48×48), qui limite l'accès aux micro-expressions.
* Déséquilibre des classes, qui pénalise surtout *Disgust*.
* Annotations parfois ambiguës ou bruitées : Fear, Sad, Angry et Neutral sont souvent confondues.
* Le Transfer Learning (MobileNetV2) reste inférieur au CNN retenu.

Perspectives : mieux traiter le déséquilibre des classes, tester d'autres architectures, ajouter des données, améliorer la robustesse (pose, éclairage, temps réel).

## 👥 Auteurs

Anthusan SRIKARAN et Emine OULD AGATT — 9 octobre 2026
