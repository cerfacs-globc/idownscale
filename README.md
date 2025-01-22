# IRISCC

## Contexte du projet

Le projet **IRISCC** (nom complet à définir si nécessaire) a pour objectif de [description générale du projet, son but, et les problématiques qu'il adresse].

Il est conçu pour [type d'application ou d'utilisation, comme la classification d'images, le traitement du langage naturel, etc.]. 

Ce document fournit un aperçu de la structure du code, des commandes utiles pour manipuler le projet, et des informations nécessaires pour démarrer rapidement.

---

## Structure du code

Le projet est organisé comme suit :

```
IRISCC/
├── data/               # Dossier contenant les datasets
│   ├── raw/           # Données brutes
│   ├── processed/     # Données transformées et prêtes à l'utilisation
├── src/                # Code source principal
│   ├── preprocessing/ # Scripts de pré-traitement des données
│   ├── models/        # Définition des modèles
│   ├── training/      # Scripts pour l'entraînement
│   ├── evaluation/    # Scripts pour l'évaluation
├── notebooks/          # Jupyter Notebooks pour exploration et prototypage
├── scripts/            # Scripts utilitaires pour automatisation
├── logs/               # Logs générés pendant l'entraînement ou les tests
├── outputs/            # Résultats finaux (modèles entraînés, prédictions, etc.)
├── requirements.txt    # Dépendances Python
├── README.md           # Documentation du projet (vous y êtes !) 
```

---

## Commandes utiles

### Create Dataset

#### Description
Cette commande permet de créer un dataset en transformant les données brutes et en les sauvegardant dans le répertoire approprié.

#### Commande bash
```bash
python src/preprocessing/create_dataset.py --input data/raw --output data/processed
```

---

### Train

#### Description
L'entraînement du modèle à partir des données préparées.

#### Commande bash
```bash
python src/training/train.py --config configs/train_config.yaml
```

---

### Test

#### Description
Tester un modèle déjà entraîné pour évaluer ses performances.

#### Commande bash
```bash
python src/evaluation/test.py --model outputs/model.pth --data data/processed/test
```

---
