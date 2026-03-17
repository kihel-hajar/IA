# Compte Rendu : L'algorithme K-Means en Machine Learning

Ce rapport présente une analyse approfondie de l'algorithme de clustering **K-Means**, explorant ses fondements théoriques, son implémentation en Python, jusqu'à l'analyse critique de ses performances sur des données réelles.

---

## 1. Introduction

### Définition du clustering
Le **clustering** (ou apprentissage non supervisé) est une méthode d'analyse de données visant a regrouper un ensemble d'observations en différents sous-groupes (ou clusters). Contrairement à la classification ou à la régression, les données ne possèdent pas d'étiquettes préalables. L'objectif est de s'assurer que les éléments au sein d'un même cluster soient aussi similaires que possible, et que les éléments de clusters différents soient aussi dissemblables que possible.

### Présentation de K-Means
Le **K-Means** est l'algorithme de clustering le plus connu et le plus utilisé en Machine Learning en raison de sa simplicité et de sa rapidité. Il a pour but de partitionner *N* observations en *K* clusters, où *K* est un hyperparamètre définit au préalable par l'utilisateur. Chaque observation appartient au cluster dont elle est le plus proche de la moyenne (le **centroïde**).

### Cas d'utilisation
L'algorithme K-Means excelle dans divers domaines :
- **Segmentation client** : regrouper les clients selon leurs comportements d'achat pour des campagnes marketing ciblées.
- **Compression d'images** : réduire le nombre de couleurs d'une image en regroupant les pixels similaires.
- **Détection d'anomalies** : identifier un comportement inhabituel séparé des clusters normaux (fraude, défaillance réseau).
- **Classification automatique de documents** : grouper des articles par topics similaires.

---

## 2. Principe de l'algorithme

### Explication simple et mathématique
Le principe mathématique repose sur la minimisation de la variance intra-cluster, mathématiquement appelée **Inertie** ou **WCSS** (Within-Cluster Sum of Squares).
Pour *K* clusters $C = \{C_1, C_2, ..., C_K\}$ et leurs centroïdes respectifs $\mu = \{\mu_1, \mu_2, ..., \mu_K\}$, l'algorithme cherche à minimiser la fonction d'objectif suivante :

$$ WCSS = \sum_{j=1}^{K} \sum_{x \in C_j} || x - \mu_j ||^2 $$

Où :
- $x$ est un point de données appartenant au cluster $C_j$.
- $\mu_j$ est le centre (la moyenne) du cluster $C_j$.
- $|| x - \mu_j ||$ est la distance euclidienne entre le point $x$ et le centroïde $\mu_j$.

### Étapes de l'algorithme (Méthode de Lloyd)
L'algorithme procède de manière itérative jusqu'à atteindre un état de convergence (stabilité des clusters) :
1. **Initialisation** : Choisir *K* centroïdes initiaux de manière aléatoire (ou méthodiquement avec *K-Means++*).
2. **Assignation** : Assigner chaque point du dataset au centroïde le plus proche (en utilisant la distance euclidienne). Cela forme *K* clusters provisoires.
3. **Mise à jour** : Recalculer la position de chaque centroïde. Le nouveau centroïde devient le barycentre (la moyenne arithmétique) de tous les points nouvellement assignés à ce cluster.
4. **Répétition** : Répéter les étapes 2 et 3 jusqu'à ce que la position des centroïdes ne change plus (convergence) ou que le nombre maximal d'itérations soit atteint.

### Hypothèses
Pour que K-Means fonctionne de manière optimale, il suppose que :
- **Les clusters sont convexes et isotropes** (de forme plutôt sphérique).
- **Les clusters ont des variances (dispersions) similaires**.
- **Les clusters sont de taille comparable** (nombre d'éléments sensiblement égal).
Si le dataset s'écarte fortement de ces hypothèses géométriques (par ex. formes en lune, de densités différentes), d'autres algorithmes comme DBSCAN ou Spectral Clustering seront plus indiqués.

---

## 3. Préparation des données

Avant d'entraîner K-Means, une bonne préparation est essentielle. 
Nous utiliserons le célèbre **Dataset Iris** de Scikit-Learn.
1. La **normalisation** permet à ce que les variables sur des échelles numériques différentes ne biaisent pas le calcul des distances euclidiennes de l'algorithme.
2. La **PCA** (Analyse en Composantes Principales) permet de projeter l'espace original (4 dimensions pour Iris) en 2 dimensions pour la visualisation.

### Code Python : Chargement et Préparation

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# 1. Charger les données
iris = datasets.load_iris()
X = iris.data # Les features (longueurs et largeurs des sépales et pétales)
y_true = iris.target # Les vraies classes (pour comparaison ultérieure)

# 2. Normaliser les données 
# Les caractéristiques sont centrées autour d'une moyenne de 0 avec un écart-type de 1
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. Réduire en 2D avec PCA pour la visualisation
# On conserve 2 composantes principales
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)

print(f"Dimensions originales : {X_scaled.shape}")
print(f"Dimensions après PCA : {X_pca.shape}")
```

---

## 4. Implémentation de K-Means

L'implémentation standard s'effectue aisément via `KMeans` de la librairie **Scikit-Learn**. 
Sachant empiriquement qu'il y a 3 classes d'Iris sauvages, nous définissons `k=3`. Nous explicitons aussi l'initialisation `k-means++` pour accélérer et fiabiliser la convergence.

### Code Python : Entraînement et Prédiction

```python
from sklearn.cluster import KMeans

# Paramétrage de KMeans
k = 3
kmeans = KMeans(
    n_clusters=k, 
    init='k-means++', # Choisit les centres initiaux intelligemment
    n_init=10,        # Exécute l'algo 10 fois avec des centres de départ différents 
    max_iter=300, 
    random_state=42
)

# Entraîner le modèle et prédire directement sur quelles données
# Attention: on entraîne TOUJOURS sur les données dimensionnées (X_scaled) 
# et non sur les composantes de la PCA pour ne pas perdre d'informations lors du calcul des distances.
y_kmeans = kmeans.fit_predict(X_scaled)

# Récupération des centroïdes générés par le modèle KMeans (qui sont dans l'espace 4D)
centroids_original = kmeans.cluster_centers_

# Projection des centroïdes en 2D avec notre objet PCA précédemment ajusté, 
# pour pouvoir les afficher correctement par la suite.
centroids_pca = pca.transform(centroids_original)

print("Entraînement réussi. Centroïdes 2D :\n", centroids_pca)
```

---

## 5. Évaluation du modèle

Dans un contexte de clustering où nous n'avons théoriquement pas les "vraies" étiquettes (`y_true`), nous devons utiliser des métriques intrinsèques pour évaluer la qualité de notre partitionnement.

1. **Silhouette Score** : varie de -1 à +1. Une valeur proche de +1 indique que les points sont très proches au sein de leur cluster, et bien séparés des autres clusters.
2. **Inertie (WCSS)** : la somme des carrés des distances entre les points et le centre de leur cluster respectif. On cherche à la minimiser.

### Code Python : Métriques intrinsèques

```python
from sklearn.metrics import silhouette_score

# 1. Calcul du Silhouette Score
silhouette_avg = silhouette_score(X_scaled, y_kmeans)
print(f"Silhouette Score : {silhouette_avg:.3f}")

# 2. Récupération de l'Inertie (WCSS) directement calculée par l'objet kmeans
wcss = kmeans.inertia_
print(f"Inertie (WCSS) : {wcss:.2f}")
```

**Interprétation :**
- Un **Silhouette Score d'environ ~0.46** sur ce jeu de données (avec K=3) indique des clusters de qualité correcte, et distincts les uns des autres, même s'il peut y avoir quelques chevauchements.
- L'**Inertie (WCSS) approche 139.8**. Considérée seule, l'inertie absolue n'a pas beaucoup de sens mathématique (elle dépend de l'échelle des données), c'est pourquoi nous utilisons la méthode "Elbow" ci-dessous pour l'interpréter comparativement.

---

## 6. Visualisation

### Code Python : Graphiques (PCA et Méthode "Elbow")

```python
plt.figure(figsize=(14, 5))

# Graphique 1 : Affichage des clusters en 2D
plt.subplot(1, 2, 1)

# Couleurs pour chaque cluster
colors = ['purple', 'teal', 'gold']

# Affichage des points de données appartenant à chaque cluster
for i in range(k):
    plt.scatter(
        X_pca[y_kmeans == i, 0], X_pca[y_kmeans == i, 1], 
        s=50, c=colors[i], label=f'Cluster {i}', alpha=0.7, edgecolor='k'
    )

# Affichage des centroïdes
plt.scatter(
    centroids_pca[:, 0], centroids_pca[:, 1], 
    s=250, marker='X', c='red', edgecolor='black', label='Centroïdes'
)

plt.title("Clustering K-Means (Dataset Iris projeté via PCA)")
plt.xlabel('Composante Principale 1')
plt.ylabel('Composante Principale 2')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)

# Graphique 2 : Méthode Elbow (Coude)
plt.subplot(1, 2, 2)
inertias = []
K_range = range(1, 11)

# On recalcule KMeans pour chaque K de 1 à 10
for i in K_range:
    km = KMeans(n_clusters=i, init='k-means++', n_init=10, random_state=42)
    km.fit(X_scaled) # Toujours sur le jeu normalisé
    inertias.append(km.inertia_)

plt.plot(K_range, inertias, marker='o', color='blue', linestyle='dashed')
plt.title("Méthode de l'Elbow (Coude)")
plt.xlabel("Nombre de clusters (k)")
plt.ylabel("Inertie (WCSS)")
plt.xticks(K_range)
plt.grid(True, linestyle='--', alpha=0.5)

# Marqueur visuel du coude
plt.axvline(x=3, color='red', linestyle=':', label='Coude optimal (k=3)')
plt.legend()

plt.tight_layout()
plt.show()
```

### Description des graphiques générés
- Le **Graphique des clusters en 2D** met en exergue trois groupes distincts. Le premier groupe (violet) est particulièrement bien séparé des autres (il correspond à l'espèce de fleur *Iris Setosa*). Les deux autres groupes (bleu-vert et jaune) sont plus proches et présentent une légère zone de chevauchement à leur frontière commune. Les étoiles rouges, représentant les **centroïdes** de notre algorithme, se positionnent logiquement au barycentre de ces attroupements.
- La **Méthode Elbow (Coude)** vient certifier notre choix du paramètre hyperparamètre $K=3$. On constate que l'Inertie baisse fortement lorsqu'on passe de $K=1$ à $K=2$, puis à $K=3$. Cependant, après $K=3$, la pente s'assouplit massivement ; la fonction décrit un "coude" très net. Cela signifie qu'ajouter un quatrième cluster n'apporterait que des gains marginaux dans la minimisation de la variance. La valeur $K=3$ est bel et bien la configuration optimale.

---

## 7. Analyse des résultats

### Interprétation des clusters obtenus 
Le modèle K-Means a réussi à segmenter de manière non-supervisée le jeu de variables botaniques (sépales, pétales) en trois grands groupes. Le K-Means y décèle :
- **Un cluster très compact et isolé** : Caractéristiques morphologiques clairement tranchées par rapport au reste de la population (petals très courts). 
- **Deux autres clusters proches** : Présentent des frontières un peu floues, indiquant des variations intermédiaires.

### Comparaison avec les vraies classes (Iris)
Étant donné qu'Iris est fondamentalement un jeu de données classifié (Setosa, Versicolor, Virginica), nous avons l'opportunité de comparer qualitativement notre partitionnement K-Means à la réalité "terrain" :
- **Cluster 0 (Séparé)** : Identifie à 100% l'espèce *Iris Setosa* qui est physiquement dissimilaire des deux autres.
- **Clusters 1 et 2 (Proches)** : Reflètent respectivement *Iris Versicolor* et *Iris Virginica*. L'algorithme se trompe sur quelques échantillons situés à la frontière de ces deux espèces car leurs dimensions se chevauchent naturellement, rendant leur séparation via des frontières purement linéaires ou sphériques imparfaite.

### La qualité du clustering
La qualité globale est excellemment robuste, étayée par un paramétrage adéquat de l'initialisation (k-means++) et une mise à l'échelle rigoureuse (StandardScaler). Elle serait beaucoup plus dégradée sans l'étape de pré-réglage de nos Features à moyenne nue. 

---

## 8. Conclusion

### Résumé des performances
L'algorithme K-Means s'avère particulièrement efficace sur le dataset de référence abstrait `Iris`, démontrant une forte capacité de généralisation sous un score en Silhouette ~0.46, et une modélisation pertinente justifiée par l'approche de la variance (Elbow). Il sépare magistralement les cas de figure bien distincts des variances qui se chevauchent.

### Avantages de K-Means
- **Facile à comprendre** et tout aussi simple à implémenter.
- **Extrêmement rapide** et scalaire sur d'importants volumes de données (Complexité en $O(n.k.t.d)$).
- **Très robuste** lorsque les clusters sont convexes (sphériques).
- S'adapte facilement à des nouveaux points entrants.

### Limites de K-Means
- **L'hyperparamètre K doit être présélectionné à l'avance** (bien que la méthode Elbow pallie légèrement ce désagrément).
- Il reste **très sensible à l'initialisation des centres** et aux **Outliers** (les valeurs aberrantes, qui éloignent dramatiquement la moyenne).
- Exécrable face à des données de formes non-sphériques ou de densités très inégales.

### Cas recommandés
L'usage de la technique K-Means est **absolument recommandée** si vous disposez d'intuitions quantitatives sur le nombre probable de partitions à attendre, et que vous traitez de larges jeux de données où les variations restent statistiquement bornées. Idéal pour la segmentation clientèle standard, pour le prototypage rapide, et pour servir de point de repère (*Baseline model*) sur tout nouveau projet de Clustering.
