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

### Création des jeux de données

Cette commande permet de créer un jeu de données pour l'entraînement des réseaux de neurones avec des entrées x et des sorties y. Une interpolation conservative est appliquée aux entrées pour correspondre à la taille des données de sorties. Une liste d'exemple correspondant à un pas de temps (journalier) est stockée dans un répertoire `dataset` associé à l'expérience. La topographie de référence est ajoutée aux entrées. 

L'expérience 3 prend SAFRAN comme référence. Une interpolation bilinéaire est utilisée comme baseline.
```bash
python3 bin/preprocessing/build_dataset_exp3.py
```
```bash
python3 bin/preprocessing/build_dataset_exp3_baseline.py
```

L'expérience 4 prend E-OBS comme référence. Le domain comprend une partie de l'Europe. La selection du domaine est appliqué par lors de l'entrainement. Une interpolation bilinéaire est utilisée comme baseline.
```bash
python3 bin/preprocessing/build_dataset_exp4.py
```
```bash
python3 bin/preprocessing/build_dataset_exp4_baseline.py
```

Afin de normaliser les données, le script `compute_statistics.py --dataset_path` calcule les statistiques de chaque canal et les sauvegarde sous le nom de `statistics.json` dans le répertoire de l'expérience. 
ATTENTION : Si vous souhaitez appliquer un masque aux entrées, pensez à faire cette étape avant la normalisation.


---

### Entraînement

La classe IRISCCHyperParameters() regroupe tous les hyper-paramètres nécessaires à l'entraînement des réseaux de neurones. La commande pour entrainer le réseau est : 

```bash
python bin/training/train.py
```
Les résultats sont enregistrés dans le répertoire 'runs'. L'avancée des métriques peut être visualisée sur Tensorboard grace à la commande :
```bash
tensorboard --logdir='path-to-runs'
```
Le chemin vers les poids du modèle le mieux entrainé devra être renommé '{version_best}' pour le post-traitement.

---
### Correction de biais
Dans l'approche 'perfect prognosis' employée par [Soares et al. (2024)](https://gmd.copernicus.org/articles/17/229/2024/) et [Vrac et Vaittinada Ayar (2017)](https://journals.ametsoc.org/view/journals/apme/56/1/jamc-d-16-0079.1.xml), le réseau de neurone apprend la relation de desente d'échelle entre les réanalyses et les observations avant d'appliquer les poids à des données simulées. Les données simulées sont corrigées par rapport aux réanalyses en pré-traitement afin de réduire le biais du modèle.

On utilise ici la méthode CDF-t [(P.-A. Michelangeli (2009))](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2009GL038401). Les données sont premièrement pré-traitée pour créer un jeu d'entraînement et deux jeux de données (historique et futur) à débaiser dont un servira pour l'évaluation de la méthode.
```bash
python bin/preprocessing/build_dataset_bc.py
```
Le script `bin/preprocessing/bias_correction_ibicus.py` corrige, évalue et enregistre les données dans le même format que celle utilisée pour l'entraînement du réseau de neurone.

```bash
python bin/preprocessing/bias_correction_ibicus.py --exp exp3 --ssp ssp585 --simu rcm --var tas
```


### Prédiction
Un réseau de neurone pré-entraîné peuvent être utilisé pour prédire de nouvelles sorties à partir d'entrées jamais vues par le réseau. 
Un jeu de test permet de comparer la prédiction à la référence pour une date donnée. Ce même jeu de test est utilisé lors de l'entrainement  pour calculer des métriques d'évaluation. La prédiction est obtenue par :

```bash
python bin/training/predict.py --date 20121018 --exp exp3 --test-name unet
```
La commande suivante crée un fichier netCDF pour prédire une longue période sans avoir à comparer avec la référence (pour le futur par exemple) : 
```bash
python bin/training/predict_loop.py --exp exp3 --test-name unet 
```

Rq : L'option `--simu-test gcm` indique si les données en entrée sont des données ERA5 (no), CNRM-CM6-1 (gcm) ou CNRM-CM6-1 corrigées par rapport à ERA5 (gcm_bc), de même pour le RCM ALADIN. Les données sont ainsi récupérées dans les répertoires associés.

### Evaluation

Les prédictions du réseau de neurone sont comparées aux données de référence pour la période de test historique. 

#### Calcul des métriques
Pour les métriques journalières : 
```bash
python3 bin/evaluation/compute_test_metrics_day.py --exp exp3 --test-name unet 
```
```bash
python3 bin/evaluation/compute_test_metrics_day.py --exp exp3 --test-name baseline 
```
pour les métriques mensuelles :
```bash
python3 bin/evaluation/compute_test_metrics_month.py --exp exp3 --test-name unet 
```
Pour calculer les métriques avec les simulations débiaisées ou non, il faut également ajouter l'argument `--simu-test gcm`

#### Visualisation des métriques
```bash
python3 bin/evaluation/compare_test_metrics.py --exp exp3 --test-list unet_gcm,unet_gcm_bc --scale monthly --pp yes
```

#### Tendance future
La commande suivante crée une figure des changements de température entre les périodes futures et la période de référence 1980-2014. 
```bash
python3 bin/evaluation/evaluate_future_trend.py --exp exp3 --ssp ssp585
```

---