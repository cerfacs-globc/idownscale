# IRISCC

## Contexte du projet

Le projet **IRISCC** (nom complet à définir si nécessaire) a pour objectif de [description générale du projet, son but, et les problématiques qu'il adresse].

Il est conçu pour [type d'application ou d'utilisation, comme la classification d'images, le traitement du langage naturel, etc.]. 

Ce document fournit un aperçu de la structure du code, des commandes utiles pour manipuler le projet, et des informations nécessaires pour démarrer rapidement.

---

## Structure du code

Le projet est organisé comme suit :

```
iriscc/
├── bin/               # Dossier contenant les datasets
│   ├── preprocessing/           # Données brutes
│   │   ├── safran_reformat.py    # Scripts pour l'évaluation
│   │   ├── build_dataset.py    # Scripts pour l'évaluation
│   │   ├── compute_statistics.py    # Scripts pour l'évaluation
│   ├── training/     # Données transformées et prêtes à l'utilisation
│   │   ├── train.py    # Scripts pour l'évaluation
│   │   ├── predict.py    # Scripts pour l'évaluation
│   ├── evaluation/     # Données transformées et prêtes à l'utilisation
│   │
├── iriscc/                # Code source principal
│   ├── dataloaders.py # Scripts de pré-traitement des données
│   ├── hparams.py        # Définition des modèles
│   ├── settings.py    # Scripts pour l'évaluation
│   ├── lightning_module.py     # Scripts pour l'entraînement
│   ├── loss.py    # Scripts pour l'évaluation
│   ├── metrics.py    # Scripts pour l'évaluation
│   ├── plotutils.py    # Scripts pour l'évaluation
│   ├── datautils.py    # Scripts pour l'évaluation
│   ├── transforms.py    # Scripts pour l'évaluation
│   ├── unet.py    # Scripts pour l'évaluation
│   ├── swin2sr.py    # Scripts pour l'évaluation

scratch/globc/garcia/
├── datasets/               # Dossier contenant les datasets
├── graphs/                # Code source principal
├── rawdata/                # Code source principal
├── runs/                # Code source principal

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
python3 bin/evaluation/compute_era5_test_metrics_daily.py exp1 swinunet_6mb_30y yes
```

---
