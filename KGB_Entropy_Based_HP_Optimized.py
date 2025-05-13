
"""
OptimizedParameterKGB.py
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import re
import time
import logging
import concurrent.futures
from datetime import datetime
from copy import deepcopy
from functools import lru_cache
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
from tqdm import tqdm

# Configuration du logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s')
logger = logging.getLogger("kgb_optimizer")

class EnhancedKGB:
    """
    Détection d'anomalies dans les flux réseau via une analyse en composantes principales.
    """
    def __init__(self,
                 sample_size=1000000,
                 n_major_components=18,
                 threshold_PCA=0.9925744596009977,
                 threshold_major=0.9997463594911226,
                 threshold_minor=0.9819175395244663,
                 random_state=42,
                 verbose=False):
        self.sample_size = sample_size
        self.n_major_components = n_major_components
        self.threshold_PCA = threshold_PCA
        self.percentil_threshold_major = threshold_major * 100
        self.percentil_threshold_minor = threshold_minor * 100
        self.verbose = verbose
        self.random_state = random_state
        np.random.seed(self.random_state)

        self.scaler = StandardScaler()
        self.feature_names = [
            'timestamp', 'duration', 'protocol', 'src_ip', 'src_port',
            'direction', 'dst_ip', 'dst_port', 'state', 'sTos', 'dTos',
            'total_packets', 'total_bytes', 'src_bytes'
        ]

        self.df = None
        self.X_entropy_scaled = None
        self.y_true = None
        self.pca_model = None
        self.feature_importances_df = None
        self.f_major = None
        self.f_minor = None
        self.entropy_df = None
        self.entropy_values = None
        self.src_ips = None

        # Création d'un dossier de sortie unique
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = os.path.join("kgb_results", self.run_id)
        os.makedirs(self.output_dir, exist_ok=True)
        if verbose:
            logger.info(f"Modèle initialisé => output_dir = {self.output_dir}")

    @staticmethod
    def compute_entropy(series):
        """Calcul de l'entropie de Shannon pour une série pandas."""
        counts = series.value_counts()
        probabilities = counts / counts.sum()
        # On ajoute une très petite constante pour éviter log(0)
        entropy = -np.sum(probabilities * np.log(probabilities + 1e-10))
        return entropy

    def load_and_preprocess_data(self, file_path):
        if not os.path.exists(file_path):
            logger.error("Fichier CSV non trouvé: %s", file_path)
            raise FileNotFoundError(f"CSV non trouvé: {file_path}")

        try:
            # Lecture du CSV en prenant les premiers 'sample_size' échantillons
            self.df = pd.read_csv(file_path, header=0, engine='python', nrows=self.sample_size)
            # Renommage des colonnes selon self.feature_names et ajout de 'flow'
            columns = self.feature_names.copy()
            columns.append('flow')
            self.df.columns = columns

            if self.verbose:
                logger.info(f"CSV chargé (premiers {self.sample_size} échantillons). Forme: {self.df.shape}")
        except Exception as e:
            logger.error("Erreur lors de la lecture du CSV: %s", e)
            raise

        if self.df.shape[1] < 15:
            raise ValueError("Le CSV doit contenir au moins 15 colonnes.")

        # Conversion de la colonne 'timestamp' en valeurs numériques
        try:
            self.df['timestamp'] = pd.to_datetime(self.df['timestamp'], errors='coerce')
            self.df['timestamp'] = self.df['timestamp'].fillna(pd.Timestamp('1970-01-01'))
            self.df['timestamp'] = self.df['timestamp'].astype('int64') // 10**9
        except Exception as e:
            logger.error("Erreur lors de la conversion de 'timestamp': %s", e)
            raise

        # Génération de l'étiquette (Label) à partir de la colonne 'flow'
        suspicious_keywords = ['from-botnet', 'botnet', 'attack', 'ddos', 'malicious',
                              'trojan', 'worm', 'exploit', 'phishing', 'backdoor']
        pattern = re.compile(r'(' + '|'.join(suspicious_keywords) + r')', re.IGNORECASE)
        self.df['Label'] = self.df['flow'].astype(str).apply(lambda x: 1 if pattern.search(x) else 0)

        if self.verbose:
            positive_count = self.df['Label'].sum()
            total_samples = self.df.shape[0]
            logger.info(f"Prétraitement => {self.df.shape[1]} features, {total_samples} échantillons, positifs: {positive_count} ({(positive_count/total_samples)*100:.2f}%).")

        self.y_true = self.df['Label'].values
        return self.df

    def compute_entropies(self):
        """Calcule les entropies pour tous les groupes IP source."""
        if self.df is None:
            raise ValueError("Les données doivent être chargées avant de calculer les entropies.")

        grouped = self.df.groupby('src_ip')
        entropy_list = []
        self.src_ips = []

        # Calcul des entropies pour chaque groupe (adresse IP source)
        for group_id, group in tqdm(grouped, desc="Calcul des entropies", disable=not self.verbose):
            if group.empty:
                continue
            entropies = group[self.feature_names].apply(self.compute_entropy)
            entropy_list.append(entropies.tolist())
            self.src_ips.append(group_id)

        # Construction du DataFrame d'entropies
        self.entropy_df = pd.DataFrame(entropy_list, columns=[f'H_{feat}' for feat in self.feature_names])
        self.entropy_df.insert(0, 'src_ip', self.src_ips)

        # Standardisation des entropies
        self.entropy_values = self.entropy_df[[f'H_{feat}' for feat in self.feature_names]].values
        self.X_entropy_scaled = self.scaler.fit_transform(self.entropy_values)

        return self.X_entropy_scaled

    def detect_anomalies(self, n_major_components=None, threshold_PCA=None, threshold_major=None, threshold_minor=None):
        """
        Détection des anomalies basée sur PCA et entropies.
        Permet de spécifier de nouveaux hyperparamètres sans recréer l'objet.
        """
        # Mise à jour des hyperparamètres si spécifiés
        if n_major_components is not None:
            self.n_major_components = n_major_components
        if threshold_PCA is not None:
            self.threshold_PCA = threshold_PCA
        if threshold_major is not None:
            self.percentil_threshold_major = threshold_major * 100
        if threshold_minor is not None:
            self.percentil_threshold_minor = threshold_minor * 100

        # Vérifier si les entropies ont été calculées
        if self.X_entropy_scaled is None:
            self.compute_entropies()

        # Application de la PCA
        self.pca_model = PCA(n_components=self.threshold_PCA, whiten=False, random_state=self.random_state)
        principal_components = self.pca_model.fit_transform(self.X_entropy_scaled)
        eigenvalues = self.pca_model.explained_variance_
        n_components = principal_components.shape[1]

        # Calcul des scores d'anomalie
        f_major = np.sum((principal_components[:, :self.n_major_components]**2) / (eigenvalues[:self.n_major_components]**2), axis=1)
        if n_components > self.n_major_components:
            f_minor = np.sum((principal_components[:, self.n_major_components:]**2) / (eigenvalues[self.n_major_components:]**2), axis=1)
        else:
            f_minor = np.zeros_like(f_major)

        # Sauvegarde des scores
        self.f_major = f_major
        self.f_minor = f_minor

        # Calcul des seuils
        threshold_major = np.percentile(f_major, self.percentil_threshold_major)
        threshold_minor = np.percentile(f_minor, self.percentil_threshold_minor)

        anomaly_mask = (f_major > threshold_major) | (f_minor > threshold_minor)
        anomaly_indices = np.where(anomaly_mask)[0]

        # Utiliser les indices d'anomalies pour créer le DataFrame d'anomalies
        anomalies_df = self.entropy_df.loc[anomaly_indices].copy() if len(anomaly_indices) > 0 else pd.DataFrame(columns=['src_ip'])
        if len(anomaly_indices) > 0:
            anomalies_df['f_major'] = self.f_major[anomaly_indices]
            anomalies_df['f_minor'] = self.f_minor[anomaly_indices]

        # Protection contre les erreurs 'src_ip'
        try:
            # Copie des IP sources anomales pour les rechercher dans le DataFrame
            anomaly_ips = set(anomalies_df['src_ip'].values) if not anomalies_df.empty else set()

            # Marquer chaque paquet comme anomalie si son IP source est dans les anomalies
            if 'src_ip' in self.df.columns:
                self.df['predictions'] = self.df['src_ip'].apply(lambda x: 1 if x in anomaly_ips else 0)
            else:
                self.df['predictions'] = 0
                logger.error("Colonne 'src_ip' introuvable dans le DataFrame")
        except Exception as e:
            logger.error(f"Erreur lors de la détection des anomalies: {e}")
            self.df['predictions'] = 0

        return self.df

    def compute_confusion_matrix(self, predictions_col="predictions"):
        """Calcul de la matrice de confusion: (TN, FP, FN, TP)."""
        y_true = self.df['Label'].values
        y_pred = self.df[predictions_col].values
        if len(np.unique(y_true)) == 2:
            cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
            tn, fp, fn, tp = cm.ravel()
            if self.verbose:
                logger.info(f"Matrice de confusion => TN={tn}, FP={fp}, FN={fn}, TP={tp}")
            return tn, fp, fn, tp
        else:
            logger.info("Classe positive absente dans y_true.")
            return (0, 0, 0, 0)


# Fonction d'évaluation définie au niveau du module pour être picklable
def evaluate_individual(individual_and_data):
    """
    Fonction d'évaluation globale pour ProcessPoolExecutor.
    Prend un tuple (individual, df, X_entropy_scaled, entropy_df, src_ips, sample_size)
    """
    (individual, df, X_entropy_scaled, entropy_df, src_ips, sample_size) = individual_and_data

    start_time = time.time()

    try:
        # Recréation d'un modèle pour le processus
        model_clone = EnhancedKGB(
            sample_size=sample_size,
            n_major_components=int(individual["n_major_components"]),
            threshold_PCA=float(individual["threshold_PCA"]),
            threshold_major=float(individual["threshold_major"]),
            threshold_minor=float(individual["threshold_minor"]),
            verbose=False,
            random_state=42
        )

        # Réutilisation des données prétraitées
        model_clone.df = df.copy()
        model_clone.X_entropy_scaled = X_entropy_scaled.copy()
        model_clone.entropy_df = entropy_df.copy()
        model_clone.src_ips = src_ips.copy()

        # Détection des anomalies
        model_clone.detect_anomalies()

        # Calcul de la matrice de confusion
        tn, fp, fn, tp = model_clone.compute_confusion_matrix()

        # Calcul des métriques
        results = {
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "tp": tp,
            "total": tn + fp + fn + tp,
            "time": time.time() - start_time
        }

        # Pénalité extrême pour tout FN
        if fn > 0:
            fn_penalty = 1e15 * (fn + 1)
        else:
            fn_penalty = 0

        # Pénalité progressive pour FP
        if fp <= 30000:
            fp_penalty = fp / 30000.0
        else:
            excess = fp - 30000
            fp_penalty = 1.0 + (excess / 10000.0) ** 1.5  # Plus agressive

        # Fitness combiné
        results["fitness"] = fn_penalty + fp_penalty

        return (results, individual)

    except Exception as e:
        logger.error(f"Erreur lors de l'évaluation: {str(e)}")
        # En cas d'erreur, retourner un fitness très élevé
        return ({
            "tn": 0, "fp": 0, "fn": 0, "tp": 0, "total": 0,
            "time": time.time() - start_time,
            "fitness": 1e12,
            "error": str(e)
        }, individual)