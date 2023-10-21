# Importations
import numpy as np
import matplotlib.pyplot as plt

# Définition des données
X = np.array([[0, 1, 0, 1, 1, 0], [0, 1, 1, 0, 1, 0]]).T

# Création de la matrice diagonale D des poids
D = np.diag([1/6]*6)

# Calcul des moyennes de X et Y
mean_X = np.mean(X[:, 0])
mean_Y = np.mean(X[:, 1])

# Création de la matrice de données centrées Z
Z = X - np.array([[mean_X, mean_Y]])

# Calcul de la matrice de covariance S
S = Z.T @ D @ Z

# Extraction des variances et de la covariance
var_X = S[0, 0]
var_Y = S[1, 1]
cov_XY = S[0, 1]

# Calcul des valeurs propres de S
eigenvalues = np.linalg.eigvals(S)

# Utilisation de la formule pour calculer les vecteurs propres pour les valeurs propres
u1_lambda1 = np.array([var_Y - eigenvalues[0], -cov_XY])
u1_lambda2 = np.array([var_Y - eigenvalues[1], -cov_XY])

# Sélection de deux vecteurs propres orthogonaux
eigenvector1 = -u1_lambda1
eigenvector2 = u1_lambda2

# calcul de Is(u) pour les deux vecteurs propres
Is_u1 = eigenvector1.T @ S @ eigenvector1
Is_u2 = eigenvector2.T @ S @ eigenvector2

# Vérification des relations
verification_Is_u1 = eigenvalues[0] * np.linalg.norm(eigenvector1)**2
verification_Is_u2 = eigenvalues[1] * np.linalg.norm(eigenvector2)**2

# Calcul de l'inertie totale IT
IT = np.trace(S)

# Normalisation des vecteurs propres
eigenvector1_normalized = eigenvector1 / np.linalg.norm(eigenvector1)
eigenvector2_normalized = eigenvector2 / np.linalg.norm(eigenvector2)

# calcul de Is(u) pour les vecteurs propres normalisés
Is_u1_normalized = eigenvector1_normalized.T @ S @ eigenvector1_normalized
Is_u2_normalized = eigenvector2_normalized.T @ S @ eigenvector2_normalized

# Somme des inerties portées par les 2 axes principaux
sum_Is_normalized = Is_u1_normalized + Is_u2_normalized

# Création du graphique 
plt.figure(figsize=(8, 8))
plt.scatter(X[:, 0], X[:, 1], color='blue', label='Données')
plt.quiver(mean_X, mean_Y, 0.5*eigenvector1_normalized[0], 0.5*eigenvector1_normalized[1], angles='xy', scale_units='xy', scale=1, color='red', label='Axe principal D1')
plt.quiver(mean_X, mean_Y, 0.5*eigenvector2_normalized[0], 0.5*eigenvector2_normalized[1], angles='xy', scale_units='xy', scale=1, color='green', label='Axe principal D2')
plt.axhline(0, color='black',linewidth=0.5)
plt.axvline(0, color='black',linewidth=0.5)
plt.grid(color = 'gray', linestyle = '--', linewidth = 0.5)
plt.title('Graphique des 2 axes principaux D1 et D2 prolongés')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.axis('equal')
plt.show()

# Calcul du taux d'inertie expliqué pour chaque composante
explained_inertia_rate_1 = eigenvalues[0] / np.sum(eigenvalues)
explained_inertia_rate_2 = eigenvalues[1] / np.sum(eigenvalues)

#on a bien deux fois plus de données sur un axe que sur l'autre

# Utilisation de la matrice des coordonnées V des vecteurs propres
V = np.column_stack((eigenvector1_normalized, eigenvector2_normalized))

# Calcul des coordonnées factorielles pour chaque point
factorial_coordinates = Z @ V

#les vecteurs sont tous sur les axes portés par les eigenvectors

# Calcul de la qualité des projections pour chaque individu
quality_projections_axis1 = (factorial_coordinates[:, 0]**2) / eigenvalues[0]
quality_projections_axis2 = (factorial_coordinates[:, 1]**2) / eigenvalues[1]

quality_projections = np.column_stack((quality_projections_axis1, quality_projections_axis2))

# résultats montrent, par exemple, que les individus 1, 2, 5, et 6 sont principalement représentés par l'axe u1, tandis que les individus 3 et 4 sont principalement représentés par l'axe u2

# Calcul de la contribution des individus pour chaque axe
n = len(X)  # Nombre total d'individus
contribution_axis1 = (factorial_coordinates[:, 0]**2) / (n * eigenvalues[0])
contribution_axis2 = (factorial_coordinates[:, 1]**2) / (n * eigenvalues[1])

contributions = np.column_stack((contribution_axis1, contribution_axis2))

# ces resultats montre que la cotribution de chaque vecteur est égal sur chacun des axes (1, 2, 5, et 6 pour le premier et le reste pour le deuxième)
#Pour cet ensemble de données, les individus 1, 2, 5 et 6 sont les mieux représentés et contribuent le plus à l'axe u1, tandis que les individus 3 et 4 contribuent à l'axe u2.