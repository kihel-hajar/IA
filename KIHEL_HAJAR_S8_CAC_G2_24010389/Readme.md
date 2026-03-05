 <img src="photo-kihel hajar.jpeg" style="height:464px;margin-right:432px"/>

# KIHEL HAJAR

**Numéro d’étudiant** : 24010389
**Classe** : CAC2

---
# Apprentissage Supervisé : Implémentation et Évaluation de Modèles de Classification avec Scikit-Learn

---

## 1️⃣ Modifications techniques apportées

Les principales corrections réalisées dans le notebook concernent :

* **Correction des erreurs de syntaxe Python**

  * Parenthèses manquantes
  * Commentaires mal fermés
  * erreurs d’écriture dans certaines cellules.

* **Correction de la création des variables**

  * Définition correcte de la matrice des caractéristiques et des labels :

  ```python
  X = df.drop(columns=["target"])
  y = df["target"]
  ```

* **Correction des étapes du pipeline de Machine Learning**

  * séparation correcte des données avec `train_test_split`
  * normalisation avec `StandardScaler`
  * entraînement correct des modèles.

* **Correction des métriques d’évaluation**

  * ajout ou correction de :

  ```python
  confusion_matrix
  classification_report
  accuracy_score
  ```

* **Amélioration de l'affichage des résultats**

  * affichage de la matrice de confusion
  * affichage du rapport de classification.

* **Organisation plus cohérente des cellules**

  * ordre logique :
    **chargement → préparation → entraînement → évaluation**

---

# 2️⃣ Technologies & Librairies utilisées

Le notebook utilise principalement les bibliothèques Python suivantes :

### Manipulation de données

* **Pandas**
* **NumPy**

### Visualisation

* **Matplotlib**
* **Seaborn**

### Machine Learning (Scikit-learn)

Modèles utilisés :

* **Logistic Regression**
* **K-Nearest Neighbors (KNN)**
* **Support Vector Machine (SVM)**
* **Decision Tree**
* **Random Forest**

Outils de Scikit-learn :

* `train_test_split`
* `StandardScaler`
* `accuracy_score`
* `confusion_matrix`
* `classification_report`

Dataset utilisé :

* `load_breast_cancer` (dataset de classification binaire).

---

# 3️⃣ Structure du Notebook

Le notebook suit la structure classique d’un projet de **Machine Learning supervisé**.

### 1. Importation des librairies

Chargement de toutes les bibliothèques nécessaires.

---

### 2. Chargement du dataset

Utilisation du dataset :

```python
load_breast_cancer()
```

Conversion en **DataFrame Pandas**.

---

### 3. Exploration des données (EDA)

Analyse rapide du dataset :

* aperçu des données
* distribution des classes
* statistiques descriptives.

---

### 4. Préparation des données

* séparation **features / target**

```python
X = df.drop(columns=["target"])
y = df["target"]
```

* division des données :

```python
train_test_split()
```

* normalisation :

```python
StandardScaler
```

---

### 5. Entraînement des modèles

Plusieurs algorithmes sont testés :

* Logistic Regression
* KNN
* SVM
* Decision Tree
* Random Forest

---

### 6. Évaluation des performances

Utilisation de plusieurs métriques :

* Accuracy
* Matrice de confusion
* Rapport de classification.

---

### 7. Comparaison des modèles

Analyse des performances pour identifier **le meilleur modèle**.

---
 **Objectif du notebook :**
Apprendre les bases du **Machine Learning supervisé pour un problème de classification** en comparant plusieurs algorithmes.

---


