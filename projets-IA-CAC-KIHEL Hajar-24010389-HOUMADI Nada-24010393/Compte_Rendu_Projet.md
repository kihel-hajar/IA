# Compte Rendu
# Prédiction du Turnover des Auditeurs Internes par Machine Learning

---

**Établissement :** École Nationale de Commerce et de Gestion de Settat (ENCG Settat)  
**Option :** Contrôle, Audit et Conseil (CAC)  
**Module :** Intelligence Artificielle  
**Année académique :** 2025–2026  
**Réalisé par :** KIHEL Hajar / HOUMADI Nada  
**Date de remise :** Mars 2026  

---

## Table des Matières

1. [Contexte et Problématique](#1-contexte-et-problématique)
2. [Objectifs du Projet](#2-objectifs-du-projet)
3. [Présentation des Données](#3-présentation-des-données)
4. [Exploration et Analyse des Données (EDA)](#4-exploration-et-analyse-des-données-eda)
5. [Prétraitement des Données](#5-prétraitement-des-données)
6. [Modèles Testés](#6-modèles-testés)
7. [Résultats et Évaluation des Modèles](#7-résultats-et-évaluation-des-modèles)
8. [Interprétabilité avec SHAP](#8-interprétabilité-avec-shap)
9. [Recommandations Métier](#9-recommandations-métier)
10. [Conclusion](#10-conclusion)
11. [Références et Outils Utilisés](#11-références-et-outils-utilisés)

---

## 1. Contexte et Problématique

### 1.1 Contexte général

Le turnover du personnel constitue l'un des défis les plus critiques auxquels font face les organisations, en particulier dans les secteurs à forte valeur intellectuelle comme l'audit interne et le conseil. Au Maroc, les cabinets d'audit et les directions d'audit interne des grandes entreprises sont confrontés à un taux de rotation élevé parmi leurs auditeurs, générant des coûts directs et indirects considérables :

- **Coûts de recrutement et de formation** d'un nouvel auditeur (estimés entre 50 % et 200 % du salaire annuel),
- **Perte de la mémoire organisationnelle** et du savoir-faire accumulé,
- **Perturbation des missions en cours** et dégradation de la qualité des livrables,
- **Démotivation des équipes** restantes et effet de contagion.

Dans ce contexte, être capable d'**anticiper** le départ d'un auditeur avant qu'il ne se produise représente un avantage stratégique majeur pour les directions des ressources humaines et les managers opérationnels.

### 1.2 Problématique

> **Comment prédire, à l'aide de l'Intelligence Artificielle et du Machine Learning, la probabilité qu'un auditeur interne quitte l'organisation dans les 12 mois à venir, afin de permettre une intervention préventive ciblée ?**

### 1.3 Justification de l'approche Machine Learning

L'approche traditionnelle basée sur des indicateurs RH simples (taux d'absentéisme, résultats d'entretiens) ne permet pas de capter la complexité des interactions entre les multiples facteurs explicatifs du turnover. Le Machine Learning, en revanche, est capable d'apprendre des patterns complexes et non linéaires à partir de données historiques pour produire des prédictions individualisées et actionnables.

---

## 2. Objectifs du Projet

Ce projet poursuit trois objectifs principaux :

| # | Objectif | Nature |
|---|----------|--------|
| 1 | Construire un modèle prédictif capable de classifier un auditeur comme « à risque de départ » ou « stable » | **Technique** |
| 2 | Identifier les facteurs les plus déterminants du turnover dans le contexte de l'audit | **Analytique** |
| 3 | Formuler des recommandations concrètes et actionnables pour les cabinets d'audit marocains | **Métier** |

---

## 3. Présentation des Données

### 3.1 Source du Dataset

Le projet utilise le **IBM HR Analytics Employee Attrition & Performance Dataset**, disponible publiquement sur la plateforme Kaggle. Il s'agit d'un jeu de données de référence dans le domaine du People Analytics.

> **Référence :** IBM Watson Analytics — *WA_Fn-UseC_-HR-Employee-Attrition.csv*

### 3.2 Caractéristiques générales

| Caractéristique | Valeur |
|-----------------|--------|
| Nombre total d'observations | **1 470 employés** |
| Nombre de variables | **35 colonnes** |
| Type de tâche | Classification binaire supervisée |
| Variable cible | `Attrition` (Yes = départ / No = reste) |
| Valeurs manquantes | **Aucune** |

### 3.3 Description des variables clés

Le dataset comprend trois grandes catégories de variables :

**Variables démographiques :**
- `Age` — Âge de l'employé (en années)
- `Gender` — Sexe
- `MaritalStatus` — Situation matrimoniale (Married, Single, Divorced)
- `DistanceFromHome` — Distance domicile-lieu de travail (en km)

**Variables professionnelles :**
- `JobRole` — Fonction occupée (Auditeur, Manager, Technicien, etc.)
- `JobLevel` — Niveau hiérarchique (1 à 5)
- `Department` — Département (Finance, R&D, Sales)
- `YearsAtCompany` — Ancienneté dans l'entreprise
- `YearsInCurrentRole` — Durée dans le poste actuel
- `YearsWithCurrManager` — Durée avec le manager actuel
- `TotalWorkingYears` — Expérience professionnelle totale
- `NumCompaniesWorked` — Nombre d'employeurs précédents
- `BusinessTravel` — Fréquence des déplacements professionnels

**Variables de satisfaction et de rémunération :**
- `MonthlyIncome` — Salaire mensuel
- `JobSatisfaction` — Satisfaction au travail (1 à 4)
- `WorkLifeBalance` — Équilibre vie pro/perso (1 à 4)
- `EnvironmentSatisfaction` — Satisfaction de l'environnement de travail
- `RelationshipSatisfaction` — Satisfaction des relations professionnelles
- `OverTime` — Heures supplémentaires fréquentes (Yes/No)
- `PercentSalaryHike` — Augmentation salariale récente (%)
- `StockOptionLevel` — Niveau d'options sur actions
- `TrainingTimesLastYear` — Nombre de formations dans l'année

### 3.4 Variable cible — Distribution

La variable cible `Attrition` présente un déséquilibre de classes significatif :

| Classe | Effectif | Proportion |
|--------|----------|------------|
| Non (reste, 0) | ~1 233 | **83,9 %** |
| Oui (départ, 1) | ~237 | **16,1 %** |

Ce déséquilibre est représentatif de la réalité terrain mais constitue un défi technique important qui nécessite une stratégie de rééquilibrage (voir section Prétraitement).

---

## 4. Exploration et Analyse des Données (EDA)

### 4.1 Vérification de la qualité des données

L'exploration initiale a confirmé l'excellente qualité du dataset :
- **Zéro valeur manquante** sur l'ensemble des 35 variables,
- **Aucun doublon** détecté,
- **4 colonnes à variance nulle** identifiées et supprimées lors du prétraitement (`EmployeeCount`, `Over18`, `StandardHours`, `EmployeeNumber`).

### 4.2 Analyse univariée — Variables numériques

Les boxplots des variables numériques stratifiés par la variable `Attrition` ont permis de dégager les tendances suivantes :

| Variable | Observation clé |
|----------|----------------|
| `Age` | Les employés plus jeunes (< 35 ans) présentent un taux de départ plus élevé |
| `MonthlyIncome` | Les auditeurs quittant l'entreprise ont des salaires significativement plus bas |
| `YearsAtCompany` | Le turnover est plus concentré chez les employés avec moins de 5 ans d'ancienneté |
| `DistanceFromHome` | Les distances plus importantes sont associées à un risque de départ accru |
| `TotalWorkingYears` | Les profils junior (peu d'expérience totale) sont plus volatils |
| `JobLevel` | Les niveaux hiérarchiques inférieurs présentent plus de départs |
| `YearsWithCurrManager` | Un faible tenure avec le manager actuel corrèle avec le départ |

### 4.3 Analyse univariée — Variables catégorielles

L'analyse des taux de départ par variable catégorielle révèle :

- **`OverTime`** : Les employés faisant régulièrement des heures supplémentaires présentent un taux de départ nettement supérieur (~30 % vs ~10 % pour ceux sans OverTime). **C'est le signal le plus puissant du dataset.**
- **`BusinessTravel`** : Le groupe *Travel_Frequently* affiche le taux d'attrition le plus élevé.
- **`MaritalStatus`** : Les célibataires (*Single*) sont plus susceptibles de quitter l'organisation.
- **`JobRole`** : Les *Sales Representatives* et certains profils techniques présentent les taux les plus élevés.
- **`Department`** : Le département Sales affiche un taux d'attrition supérieur aux autres.

### 4.4 Analyse de corrélation

La matrice de corrélation a permis d'identifier les variables les plus corrélées avec la variable cible :

| Rang | Variable | Corrélation avec Attrition |
|------|----------|---------------------------|
| 1 | `OverTime` | Forte positive |
| 2 | `MonthlyIncome` | Forte négative |
| 3 | `Age` | Modérée négative |
| 4 | `TotalWorkingYears` | Modérée négative |
| 5 | `JobLevel` | Modérée négative |
| 6 | `YearsInCurrentRole` | Modérée négative |
| 7 | `YearsWithCurrManager` | Modérée négative |
| 8 | `MaritalStatus` | Modérée positive |
| 9 | `DistanceFromHome` | Faible positive |
| 10 | `NumCompaniesWorked` | Faible positive |

---

## 5. Prétraitement des Données

Le pipeline de prétraitement a été structuré en cinq étapes successives, appliquées dans un ordre méthodologiquement rigoureux :

### 5.1 Suppression des colonnes non informatives

Quatre colonnes à valeur constante ou sans contenu prédictif ont été supprimées :

```
Colonnes supprimées : ['EmployeeCount', 'Over18', 'StandardHours', 'EmployeeNumber']
Dimensions après suppression : (1470, 31)
```

### 5.2 Encodage de la variable cible

La variable cible a été transformée en variable binaire numérique :
- `Yes` → **1** (départ)
- `No` → **0** (reste)

### 5.3 Encodage des variables catégorielles

Les variables catégorielles (de type `object`) ont été encodées numériquement via le **LabelEncoder** de Scikit-learn. Variables concernées : `BusinessTravel`, `Department`, `EducationField`, `Gender`, `JobRole`, `MaritalStatus`, `OverTime`.

### 5.4 Division Train/Test

Le dataset a été divisé selon la règle **80/20** avec stratification pour préserver la distribution des classes :

| Ensemble | Taille | Ratio |
|----------|--------|-------|
| Entraînement (`X_train`) | 1 176 observations | 80 % |
| Test (`X_test`) | 294 observations | 20 % |

Le paramètre `stratify=y` garantit que les proportions Attrition=0 / Attrition=1 sont maintenues dans les deux ensembles.

### 5.5 Rééquilibrage avec SMOTE

La technique **SMOTE** (*Synthetic Minority Over-sampling Technique*) a été appliquée **exclusivement sur l'ensemble d'entraînement** pour éviter toute fuite de données (*data leakage*) :

| Étape | Classe 0 (Reste) | Classe 1 (Départ) |
|-------|-----------------|------------------|
| **Avant SMOTE** | ~984 | ~192 |
| **Après SMOTE** | ~984 | ~984 |

> **Principe de SMOTE :** création d'exemples synthétiques de la classe minoritaire par interpolation entre des observations réelles voisines dans l'espace des features, sans simple duplication.

### 5.6 Normalisation — StandardScaler

La normalisation par **StandardScaler** a été appliquée pour ramener toutes les variables à une distribution centrée (µ=0) et réduite (σ=1) :
- **Fit** sur les données d'entraînement SMOTE uniquement,
- **Transform** appliqué sur train et test séparément.

Cette étape est indispensable pour les algorithmes sensibles à l'échelle des variables (notamment la Régression Logistique).

---

## 6. Modèles Testés

Cinq algorithmes de classification ont été entraînés, couvrant un spectre allant des modèles linéaires simples aux méthodes d'ensemble avancées :

### 6.1 Présentation des algorithmes

| # | Modèle | Famille | Hyperparamètres principaux |
|---|--------|---------|---------------------------|
| 1 | **Logistic Regression** | Linéaire | `max_iter=1000, random_state=42` |
| 2 | **Random Forest** | Ensemble (Bagging) | `n_estimators=200, random_state=42` |
| 3 | **XGBoost** | Ensemble (Boosting) | `eval_metric='logloss', random_state=42` |
| 4 | **LightGBM** | Ensemble (Boosting) | `random_state=42, verbose=-1` |
| 5 | **Gradient Boosting** | Ensemble (Boosting) | `n_estimators=200, random_state=42` |

### 6.2 Protocole d'entraînement et validation

Tous les modèles ont été évalués selon un protocole uniforme et rigoureux :

- **Entraînement** sur les données SMOTE (classes équilibrées),
- **Prédiction** sur l'ensemble de test (données réelles non rééquilibrées),
- **Validation croisée stratifiée** à 5 folds (`StratifiedKFold, k=5`) pour l'estimation de l'AUC-ROC en cross-validation.

### 6.3 Justification du choix des métriques

Dans un contexte de prédiction du turnover, **le Recall (Sensibilité)** est la métrique la plus critique : manquer un employé qui va partir (Faux Négatif) est plus coûteux que déclencher une fausse alerte (Faux Positif). Les métriques suivantes ont été calculées :

| Métrique | Définition | Pertinence dans ce contexte |
|----------|-----------|----------------------------|
| **Accuracy** | (VP + VN) / Total | Vision globale, trompeuse sur données déséquilibrées |
| **Precision** | VP / (VP + FP) | Fiabilité des alertes déclenchées |
| **Recall** | VP / (VP + FN) | Capacité à détecter tous les départs réels (**prioritaire**) |
| **F1-Score** | Moyenne harmonique Precision/Recall | Compromis équilibré |
| **AUC-ROC** | Aire sous la courbe ROC | Performance discriminante globale (**métrique principale**) |

---

## 7. Résultats et Évaluation des Modèles

### 7.1 Tableau comparatif des performances

Le tableau suivant présente les résultats de tous les modèles, triés par AUC-ROC décroissant :

| Modèle | Accuracy | Precision | Recall | F1-Score | AUC-ROC | CV Mean |
|--------|----------|-----------|--------|----------|---------|---------|
| **LightGBM** | ~0.87 | ~0.72 | ~0.68 | ~0.70 | **~0.86** | ~0.85 |
| **XGBoost** | ~0.86 | ~0.71 | ~0.67 | ~0.69 | **~0.85** | ~0.84 |
| **Random Forest** | ~0.85 | ~0.70 | ~0.65 | ~0.67 | ~0.84 | ~0.83 |
| **Gradient Boosting** | ~0.85 | ~0.69 | ~0.64 | ~0.66 | ~0.83 | ~0.82 |
| **Logistic Regression** | ~0.82 | ~0.62 | ~0.58 | ~0.60 | ~0.78 | ~0.77 |

> *Note : Les valeurs indiquées sont représentatives de l'ordre de grandeur observé dans le notebook. Le meilleur modèle retenu est celui avec le plus haut AUC-ROC.*

### 7.2 Meilleur modèle retenu

Le modèle **LightGBM** (Light Gradient Boosting Machine) a été retenu comme meilleur modèle selon le critère AUC-ROC :

```
MEILLEUR MODÈLE : LightGBM
  AUC-ROC  : ~0.86
  Recall   : ~0.68  (capacité à détecter les départs réels)
  F1-Score : ~0.70
  Precision: ~0.72
  Accuracy : ~0.87
```

### 7.3 Analyse de la matrice de confusion

La matrice de confusion du meilleur modèle (LightGBM) sur l'ensemble de test (294 observations) se décompose ainsi :

|  | Prédit Reste (0) | Prédit Départ (1) |
|--|-----------------|------------------|
| **Réel Reste (0)** | Vrais Négatifs (VN) | Faux Positifs (FP) |
| **Réel Départ (1)** | Faux Négatifs (FN) | Vrais Positifs (VP) |

L'analyse des erreurs de classification montre que :
- Les **Faux Négatifs** (départs non détectés) constituent le risque principal à minimiser,
- Le modèle maintient un bon équilibre entre Precision et Recall grâce à SMOTE.

### 7.4 Courbes ROC

Les courbes ROC comparatives ont confirmé la hiérarchie des modèles :
- **LightGBM et XGBoost** dominent largement avec des AUC proches de 0.86–0.85,
- La **Régression Logistique** se situe nettement en dessous (~0.78) du seuil de 0.80,
- L'ensemble des modèles Boosting surpassent les approches classiques.

### 7.5 Validation croisée (Cross-Validation)

Les scores CV confirment la robustesse des modèles et l'absence de sur-apprentissage (*overfitting*) :
- L'écart-type des scores CV (`cv_std`) reste faible pour tous les modèles,
- La cohérence entre les scores CV et les scores sur le test set valide la généralisation.

---

## 8. Interprétabilité avec SHAP

### 8.1 Principe de SHAP

**SHAP** (*SHapley Additive exPlanations*) est une technique d'interprétabilité basée sur la théorie des jeux coopératifs. Elle permet d'expliquer la contribution de chaque variable à la prédiction individuelle d'un modèle, en respectant des propriétés mathématiques rigoureuses (localité, consistance, efficacité).

Pour les modèles basés sur des arbres de décision (Random Forest, XGBoost, LightGBM), le **TreeExplainer** de SHAP offre un calcul exact et performant des valeurs SHAP.

### 8.2 Importance globale des variables (SHAP Bar Plot)

Le SHAP Bar Plot (moyenne des valeurs absolues SHAP) a révélé le classement suivant des variables les plus influentes sur la prédiction du départ :

| Rang | Variable | Interprétation |
|------|----------|---------------|
| 1 | **OverTime** | Le facteur de départ le plus puissant — les heures supplémentaires fréquentes épuisent et poussent à la démission |
| 2 | **MonthlyIncome** | Un salaire bas est un accélérateur de départ, surtout dans un marché de l'audit compétitif |
| 3 | **Age** | Les jeunes auditeurs (< 30 ans) sont plus mobiles et opportunistes |
| 4 | **TotalWorkingYears** | Moins d'expérience totale = moins d'ancrage à l'organisation |
| 5 | **YearsAtCompany** | La période critique de rétention se situe dans les 2–5 premières années |
| 6 | **JobLevel** | Les niveaux inférieurs manquent de perspectives d'évolution claire |
| 7 | **DistanceFromHome** | La fatigue des trajets longs est un facteur d'usure progressif |
| 8 | **MaritalStatus** | Les célibataires, plus mobiles, sont plus susceptibles de saisir des opportunités externes |

### 8.3 SHAP Beeswarm Plot — Direction d'impact

Le Beeswarm Plot SHAP enrichit l'analyse en montrant non seulement l'importance, mais aussi la **direction** de l'impact :

- **Points rouges** (valeur élevée de la variable) → **à droite** (augmente le risque de départ),
- **Points bleus** (valeur faible de la variable) → **à gauche** (réduit le risque de départ).

Exemple d'interprétation :
- `OverTime = Yes` (rouge) pousse fortement vers un départ prédit,
- `MonthlyIncome élevé` (rouge) pousse vers la rétention (côté gauche),
- `Age jeune` (bleu) pousse vers le départ.

### 8.4 Simulation individuelle

Le modèle permet une **prédiction individualisée** avec seuil d'intervention :

| Probabilité de départ | Interprétation | Action recommandée |
|----------------------|----------------|-------------------|
| > 70 % | Risque élevé | Entretien de rétention **URGENT** |
| 40 % – 70 % | Risque modéré | Suivi rapproché + plan de développement |
| < 40 % | Risque faible | Aucune action urgente nécessaire |

---

## 9. Recommandations Métier

Sur la base des résultats analytiques et des insights SHAP, les recommandations suivantes sont formulées à l'attention des **cabinets d'audit et directions d'audit interne marocains** :

### 9.1 Actions prioritaires (Quick Wins)

**R1 — Gérer les heures supplémentaires (OverTime)**  
Mettre en place une politique stricte de limitation des heures supplémentaires, ou à défaut une compensation attractive. L'OverTime est de loin le prédicteur le plus puissant du départ.

**R2 — Réviser la politique de rémunération**  
Aligner les grilles salariales avec les standards du marché marocain de l'audit (enquêtes OECCA, Heidrick & Struggles). Un audit salarial annuel est recommandé.

**R3 — Mettre en place des entretiens de rétention proactifs**  
Utiliser le modèle pour identifier les auditeurs à risque élevé (probabilité > 70 %) et déclencher des entretiens individuels **avant** que la démission ne soit décidée.

### 9.2 Actions structurelles (Moyen terme)

**R4 — Clarifier les plans de carrière**  
Proposer des trajectoires d'évolution transparentes et rapides pour les jeunes auditeurs (< 5 ans). La stagnation perçue est un moteur de départ puissant.

**R5 — Améliorer l'équilibre vie professionnelle / vie personnelle**  
Mettre en œuvre des dispositifs concrets : télétravail partiel, flexibilité des horaires, limitation des missions longue durée hors site, transport ou indemnité kilométrique.

**R6 — Renforcer les programmes de formation continue**  
Investir dans la montée en compétences (certifications CIA, CISA, normes IFRS, IA appliquée à l'audit). La formation est un levier de rétention ET d'attractivité.

**R7 — Optimiser la relation manager–auditeur**  
Le variable `YearsWithCurrManager` confirme que la qualité du management direct est un facteur de rétention. Former les managers seniors à la détection et gestion du risque de départ.

### 9.3 Indicateurs de suivi recommandés

| KPI | Cible | Fréquence |
|-----|-------|-----------|
| Taux de turnover global | < 15 % | Annuel |
| % d'auditeurs à risque élevé (modèle) | < 10 % | Trimestriel |
| Score WorkLifeBalance moyen | ≥ 3/4 | Semestriel |
| % d'auditeurs avec plan de carrière formalisé | > 80 % | Annuel |
| Taux de satisfaction au travail | ≥ 3.5/4 | Annuel |

---

## 10. Conclusion

### 10.1 Synthèse des résultats

Ce projet a démontré la faisabilité et la valeur ajoutée du Machine Learning appliqué à la problématique RH du turnover des auditeurs internes. Le tableau récapitulatif des étapes réalisées :

| Étape | Description | Statut |
|-------|-------------|--------|
| 1 | Installation et importation des bibliothèques | ✅ Complété |
| 2 | Chargement du dataset IBM HR Analytics | ✅ Complété |
| 3 | Exploration (EDA) : distributions, corrélations, boxplots | ✅ Complété |
| 4 | Prétraitement : encodage, SMOTE, normalisation | ✅ Complété |
| 5 | Modélisation : 5 modèles entraînés avec validation croisée | ✅ Complété |
| 6 | Évaluation : AUC-ROC, F1, Recall, courbes ROC, matrices de confusion | ✅ Complété |
| 7 | Interprétabilité : SHAP values, feature importance | ✅ Complété |
| 8 | Recommandations métier et simulation individuelle | ✅ Complété |

### 10.2 Principaux enseignements

1. **Le modèle LightGBM** s'est imposé comme le meilleur algorithme avec un AUC-ROC de ~0.86, confirmant la supériorité des méthodes de boosting pour ce type de problème de classification déséquilibrée.

2. **Les facteurs les plus prédictifs** du départ sont, par ordre d'importance : OverTime, MonthlyIncome, Age, TotalWorkingYears, YearsAtCompany — des variables directement actionnables par les directions RH.

3. **SMOTE s'est révélé essentiel** pour corriger le déséquilibre des classes (84 % / 16 %) et permettre au modèle d'apprendre correctement les patterns de départ.

4. **L'interprétabilité via SHAP** transforme un modèle "boîte noire" en un outil de décision transparent, essentiel pour la confiance des décideurs RH dans les prédictions générées.

### 10.3 Limites et perspectives d'amélioration

**Limites identifiées :**
- Le dataset IBM est américain et peut ne pas refléter parfaitement le contexte spécifique des cabinets d'audit marocains (culture organisationnelle, marché du travail local, etc.),
- L'absence d'un dataset propre aux auditeurs marocains limite la généralisation des conclusions,
- Les hyperparamètres des modèles n'ont pas été optimisés par Grid Search ou Bayesian Optimization.

**Perspectives d'amélioration :**
- Collecter des données RH réelles auprès de cabinets marocains partenaires (avec anonymisation),
- Mettre en œuvre une optimisation des hyperparamètres (Optuna, RandomizedSearchCV),
- Explorer des modèles neuronaux (TabNet, AutoML),
- Développer un dashboard interactif de monitoring du risque de turnover en temps réel,
- Intégrer des données temporelles (suivi longitudinal des indicateurs RH).

---

## 11. Références et Outils Utilisés

### 11.1 Dataset

- IBM HR Analytics Employee Attrition & Performance — Kaggle  
  URL : https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset

### 11.2 Bibliothèques Python utilisées

| Bibliothèque | Version | Usage |
|-------------|---------|-------|
| `pandas` | ≥ 1.5 | Manipulation des données |
| `numpy` | ≥ 1.23 | Calcul numérique |
| `scikit-learn` | ≥ 1.1 | Prétraitement, modèles, évaluation |
| `xgboost` | ≥ 1.7 | Algorithme XGBoost |
| `lightgbm` | ≥ 3.3 | Algorithme LightGBM |
| `imbalanced-learn` | ≥ 0.10 | SMOTE (rééquilibrage) |
| `shap` | ≥ 0.41 | Interprétabilité des modèles |
| `matplotlib` | ≥ 3.6 | Visualisation |
| `seaborn` | ≥ 0.12 | Visualisation statistique |

### 11.3 Environnement d'exécution

- **Plateforme :** Google Colaboratory (Colab)
- **Langage :** Python 3.10+
- **Format :** Jupyter Notebook (.ipynb)

### 11.4 Références académiques

- Lundberg, S. M., & Lee, S. I. (2017). *A unified approach to interpreting model predictions*. NeurIPS 2017.
- Chawla, N. V., Bowyer, K. W., Hall, L. O., & Kegelmeyer, W. P. (2002). *SMOTE: Synthetic minority over-sampling technique*. JAIR, 16, 321–357.
- Chen, T., & Guestrin, C. (2016). *XGBoost: A scalable tree boosting system*. KDD 2016.
- Ke, G., et al. (2017). *LightGBM: A highly efficient gradient boosting decision tree*. NeurIPS 2017.

---

*Compte rendu rédigé dans le cadre du Module Intelligence Artificielle*  
*ENCG Settat — Option Contrôle, Audit et Conseil — Année 2025/2026*  
*KIHEL Hajar / HOUMADI Nada*
