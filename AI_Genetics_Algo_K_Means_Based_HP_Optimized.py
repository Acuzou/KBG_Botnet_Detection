"""
K_means_Genetics_IA_Optimized.py
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

class GeneticOptimizer:
    """
    Optimisateur génétique des hyperparamètres pour le modèle EnhancedKGB.
    """
    def __init__(self,
                 file_path,
                 sample_size=1000000,
                 population_size=40,
                 generations=200,
                 mutation_factor_initial=0.05,  # Réduit encore pour des mutations plus fines
                 mutation_factor_final=0.005,   # Mutations encore plus fines en fin de parcours
                 crossover_prob=0.7,
                 elite_size=2,
                 stagnation_limit=15,  # Augmenté pour permettre plus de temps de convergence
                 diversity_threshold=0.01,  # Réduit pour être plus sensible à la perte de diversité
                 improvement_window=5,  # Fenêtre pour mesurer les améliorations
                 n_workers=None,
                 verbose=True):

        self.file_path = file_path
        self.sample_size = sample_size
        self.population_size = population_size
        self.generations = generations
        self.mutation_factor_initial = mutation_factor_initial
        self.mutation_factor_final = mutation_factor_final
        self.crossover_prob = crossover_prob
        self.elite_size = elite_size
        self.stagnation_limit = stagnation_limit
        self.diversity_threshold = diversity_threshold
        self.improvement_window = improvement_window
        self.n_workers = n_workers if n_workers is not None else os.cpu_count()
        self.verbose = verbose

        # Valeurs optimales déjà identifiées
        self.best_known = {
            "n_major_components": 16,
            "threshold_PCA": 0.992570308853951,
            "threshold_major": 0.9996810217716001,
            "threshold_minor": 0.981087081050228
        }

        # Plages de paramètres très réduites autour des meilleures valeurs connues
        self.param_ranges = {
            "n_major_components": {"min": 14, "max": 18, "type": "int", "focus": self.best_known["n_major_components"]},
            "threshold_PCA": {"min": 0.990, "max": 0.995, "type": "float", "focus": self.best_known["threshold_PCA"]},
            "threshold_major": {"min": 0.9994, "max": 0.9999, "type": "float", "focus": self.best_known["threshold_major"]},
            "threshold_minor": {"min": 0.979, "max": 0.983, "type": "float", "focus": self.best_known["threshold_minor"]}
        }

        # Pour suivre la progression
        self.best_solutions = []
        self.all_results = []
        self.best_no_fn_solution = None
        self.min_fp = float('inf')
        self.stagnation_counter = 0  # Nouveau: compteur pour early stopping
        self.last_best_fitness = float('inf')  # Nouveau: pour suivre l'amélioration

        # Création d'un dossier pour les résultats
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = os.path.join("genetic_optimizer_results", self.run_id)
        os.makedirs(self.output_dir, exist_ok=True)
        logger.info(f"Optimisateur initialisé => output_dir = {self.output_dir}, n_workers = {self.n_workers}")

        # Initialisation du modèle et chargement des données une seule fois
        self.model = EnhancedKGB(
            sample_size=self.sample_size,
            verbose=self.verbose,
            random_state=42
        )
        logger.info(f"Chargement et prétraitement des données...")
        self.model.load_and_preprocess_data(self.file_path)
        logger.info(f"Calcul des entropies...")
        self.model.compute_entropies()
        logger.info(f"Données prêtes pour l'optimisation")

        # Préchargement des données pour éviter les problèmes de parallélisation
        self.df = self.model.df
        self.X_entropy_scaled = self.model.X_entropy_scaled
        self.entropy_df = self.model.entropy_df
        self.src_ips = self.model.src_ips

    def initialize_population(self):
        """Initialise la population avec un focus précis sur les meilleures valeurs connues."""
        population = []

        # 1. Ajouter la meilleure solution connue
        population.append(self.best_known.copy())

        # 2. Micro-variations très fines autour de la meilleure solution (70% de la population)
        micro_variations_count = int(self.population_size * 0.7)
        for i in range(micro_variations_count):
            variation = {}
            for param, range_info in self.param_ranges.items():
                focus_value = range_info["focus"]
                param_range = range_info["max"] - range_info["min"]

                # Facteur de mutation extrêmement petit (0.05% à 0.5% de la plage)
                mutation_scale = param_range * np.random.uniform(0.0005, 0.005)

                if range_info["type"] == "int":
                    # Pour n_major_components, rester très proche de la valeur optimale
                    if np.random.random() < 0.7:  # 70% de chance de garder la valeur exacte
                        new_value = int(focus_value)
                    else:
                        delta = np.random.choice([-1, 1])  # Seulement +/-1
                        new_value = int(focus_value + delta)
                else:
                    # Pour les flottants, perturbation gaussienne très fine
                    new_value = focus_value + np.random.normal(0, mutation_scale)

                # Assurer que la valeur reste dans les limites
                new_value = max(range_info["min"], min(range_info["max"], new_value))

                # Conversion au type approprié
                if range_info["type"] == "int":
                    new_value = int(round(new_value))

                variation[param] = new_value

            population.append(variation)

        # 3. Variations un peu plus larges (20% de la population)
        medium_variations_count = int(self.population_size * 0.2)
        for i in range(medium_variations_count):
            variation = {}
            for param, range_info in self.param_ranges.items():
                focus_value = range_info["focus"]
                param_range = range_info["max"] - range_info["min"]

                # Facteur de mutation modéré (0.5% à 2% de la plage)
                mutation_scale = param_range * np.random.uniform(0.005, 0.02)

                if range_info["type"] == "int":
                    delta = np.random.choice([-2, -1, 0, 1, 2], p=[0.1, 0.2, 0.4, 0.2, 0.1])
                    new_value = int(focus_value + delta)
                else:
                    new_value = focus_value + np.random.normal(0, mutation_scale)

                # Assurer que la valeur reste dans les limites
                new_value = max(range_info["min"], min(range_info["max"], new_value))

                # Conversion au type approprié
                if range_info["type"] == "int":
                    new_value = int(round(new_value))

                variation[param] = new_value

            population.append(variation)

        # 4. Explorer les limites de l'espace (10% de la population)
        # Quelques solutions aux "extrêmes" des plages réduites pour l'exploration
        num_extreme = self.population_size - len(population)
        for i in range(num_extreme):
            variation = {}
            for param, range_info in self.param_ranges.items():
                # Alternatif entre min, max et valeurs aléatoires
                choice = i % 3
                if choice == 0:
                    # Valeur proche du minimum
                    if range_info["type"] == "int":
                        new_value = range_info["min"] + np.random.randint(0, 2)
                    else:
                        new_value = range_info["min"] + np.random.uniform(0, 0.1 * (range_info["max"] - range_info["min"]))
                elif choice == 1:
                    # Valeur proche du maximum
                    if range_info["type"] == "int":
                        new_value = range_info["max"] - np.random.randint(0, 2)
                    else:
                        new_value = range_info["max"] - np.random.uniform(0, 0.1 * (range_info["max"] - range_info["min"]))
                else:
                    # Valeur aléatoire dans la plage
                    if range_info["type"] == "int":
                        new_value = np.random.randint(range_info["min"], range_info["max"] + 1)
                    else:
                        new_value = np.random.uniform(range_info["min"], range_info["max"])

                # Assurer que la valeur reste dans les limites
                new_value = max(range_info["min"], min(range_info["max"], new_value))

                # Conversion au type approprié
                if range_info["type"] == "int":
                    new_value = int(round(new_value))

                variation[param] = new_value

            population.append(variation)

        return population

    def evaluate_population_batch(self, population, executor):
        """Évalue toute une population en utilisant l'exécuteur donné."""
        try:
            # Préparation des arguments pour la fonction d'évaluation
            eval_args = []
            for ind in population:
                eval_args.append((ind, self.df, self.X_entropy_scaled, self.entropy_df, self.src_ips, self.sample_size))

            # Utilisation de map pour évaluer
            results = list(executor.map(evaluate_individual, eval_args))

            # Traiter les résultats
            fitness_results = []
            for result, individual in results:
                result["individual"] = individual.copy()
                fitness_results.append(result)

            return fitness_results
        except Exception as e:
            logger.warning(f"Erreur avec executor: {str(e)}. Utilisation de l'évaluation séquentielle.")
            # Fallback: évaluation séquentielle en cas d'échec
            fitness_results = []
            for ind in tqdm(population, desc="Évaluation séquentielle", disable=not self.verbose):
                result, individual = evaluate_individual((ind, self.df, self.X_entropy_scaled,
                                                        self.entropy_df, self.src_ips, self.sample_size))
                result["individual"] = individual.copy()
                fitness_results.append(result)
            return fitness_results

    def crossover(self, parent1, parent2):
        """
        Effectue un croisement avancé entre deux parents avec plusieurs stratégies:
        1. Échange de paramètres pondéré
        2. Croisement arithmétique pour les paramètres numériques
        3. Croisement préservant les groupes de paramètres corrélés
        """
        if np.random.random() < self.crossover_prob:
            child = {}

            # Décider de la stratégie de croisement
            crossover_strategy = np.random.choice([
                'parameter_exchange',    # Échange simple de paramètres
                'weighted_average',      # Moyenne pondérée des valeurs
                'intelligent_grouping',  # Croisement préservant les groupes de paramètres corrélés
            ], p=[0.5, 0.3, 0.2])        # Probabilités pour chaque stratégie

            if crossover_strategy == 'parameter_exchange':
                # Stratégie 1: Échange de paramètres pondéré
                for param in self.param_ranges.keys():
                    # Mélange intelligent - plus de chance de prendre du meilleur parent
                    # Si parent1 est supposé meilleur (passé en premier)
                    if np.random.random() < 0.7:  # 70% de chance de prendre du parent1
                        child[param] = parent1[param]
                    else:
                        child[param] = parent2[param]

            elif crossover_strategy == 'weighted_average':
                # Stratégie 2: Croisement arithmétique (moyenne pondérée)
                # Meilleur pour exploration fine entre deux bonnes solutions
                for param, range_info in self.param_ranges.items():
                    # Poids aléatoire pour la moyenne
                    weight = np.random.beta(2, 2)  # Distribution Beta centrée sur 0.5 mais variable

                    # Moyenne pondérée
                    if range_info["type"] == "int":
                        # Pour les entiers, arrondir la moyenne pondérée
                        weighted_avg = weight * parent1[param] + (1 - weight) * parent2[param]
                        child[param] = int(round(weighted_avg))
                    else:
                        # Pour les flottants, moyenne pondérée directe
                        child[param] = weight * parent1[param] + (1 - weight) * parent2[param]

            else:  # intelligent_grouping
                # Stratégie 3: Préserver les groupes de paramètres corrélés
                # Les paramètres sont regroupés selon leur corrélation fonctionnelle

                # Groupe 1: n_major_components et threshold_PCA sont liés
                if np.random.random() < 0.5:
                    # Prendre ce groupe du parent 1
                    child["n_major_components"] = parent1["n_major_components"]
                    child["threshold_PCA"] = parent1["threshold_PCA"]
                else:
                    # Prendre ce groupe du parent 2
                    child["n_major_components"] = parent2["n_major_components"]
                    child["threshold_PCA"] = parent2["threshold_PCA"]

                # Groupe 2: threshold_major et threshold_minor sont liés
                if np.random.random() < 0.5:
                    # Prendre ce groupe du parent 1
                    child["threshold_major"] = parent1["threshold_major"]
                    child["threshold_minor"] = parent1["threshold_minor"]
                else:
                    # Prendre ce groupe du parent 2
                    child["threshold_major"] = parent2["threshold_major"]
                    child["threshold_minor"] = parent2["threshold_minor"]

            # Vérification finale et ajustements pour assurer la validité
            for param, range_info in self.param_ranges.items():
                # Assurer que les valeurs restent dans les limites
                child[param] = max(range_info["min"], min(range_info["max"], child[param]))

                # Conversion au type approprié
                if range_info["type"] == "int":
                    child[param] = int(round(child[param]))

            return child
        else:
            # Si pas de croisement, retourner une copie du premier parent
            return deepcopy(parent1)

    def mutate(self, individual, mutation_factor, population_diversity, improvement_trend=0, is_elite=False):
        """
        Mutation hautement adaptative basée sur multiples facteurs:
        - Diversité de la population
        - Tendance d'amélioration
        - Statut d'élite de l'individu
        - Distance par rapport à la meilleure solution connue
        """
        mutated = deepcopy(individual)

        # 1. Calcul du facteur de mutation de base
        base_factor = mutation_factor

        # 2. Ajustement en fonction de la diversité
        if population_diversity < self.diversity_threshold:
            # Augmenter le facteur si la diversité est faible pour explorer davantage
            diversity_adjustment = min(0.05, 0.05 - population_diversity)
            base_factor += diversity_adjustment

        # 3. Ajustement en fonction de la tendance d'amélioration
        # Si l'amélioration stagne (improvement_trend proche de 0), augmenter le facteur
        if abs(improvement_trend) < 0.001:
            base_factor *= 1.5
        # Si l'amélioration est bonne, réduire le facteur pour affiner
        elif improvement_trend > 0.01:
            base_factor *= 0.5

        # 4. Élites ont des mutations plus subtiles
        if is_elite:
            base_factor *= 0.3

        # 5. Probabilité de mutation adaptative pour chaque paramètre
        for param, range_info in self.param_ranges.items():
            # Distance à la meilleure valeur connue pour ce paramètre
            focus_value = range_info["focus"]
            param_range = range_info["max"] - range_info["min"]

            # Normaliser la distance actuelle par rapport à la plage
            if range_info["type"] == "int":
                normalized_distance = abs(individual[param] - focus_value) / param_range
            else:
                normalized_distance = abs(individual[param] - focus_value) / param_range

            # Probabilité de mutation inversement proportionnelle à la distance
            # Paramètres éloignés de la meilleure valeur connue mutent plus souvent
            mutation_prob = 0.2 + 0.3 * normalized_distance

            if np.random.random() < mutation_prob:
                # Amplitude de mutation adaptative
                # Plus on est loin de la valeur focus, plus les mutations sont dirigées vers elle
                attraction_to_focus = 0.0

                if normalized_distance > 0.1:  # Si on est loin du focus
                    attraction_to_focus = 0.3 * normalized_distance  # Force d'attraction vers le focus

                # Déterminer la direction de la mutation (vers focus ou aléatoire)
                if np.random.random() < attraction_to_focus:
                    # Mutation dirigée vers la valeur focus
                    direction = 1 if focus_value > individual[param] else -1
                    if range_info["type"] == "int":
                        # Pour les entiers, on fait un pas dans la bonne direction
                        step_size = max(1, int(base_factor * param_range * normalized_distance))
                        new_value = individual[param] + direction * step_size
                    else:
                        # Pour les flottants, on se rapproche proportionnellement
                        step_size = base_factor * param_range * normalized_distance
                        new_value = individual[param] + direction * step_size
                else:
                    # Mutation gaussienne standard, centrée sur la valeur actuelle
                    mutation_scale = base_factor * param_range

                    if range_info["type"] == "int":
                        # Pour les entiers, on utilise une distribution discrète
                        # Plus le facteur de mutation est grand, plus la plage de valeurs est large
                        width = max(1, int(mutation_scale * 5))
                        delta = np.random.randint(-width, width+1)
                        new_value = individual[param] + delta
                    else:
                        # Pour les flottants, petite perturbation gaussienne
                        new_value = individual[param] + np.random.normal(0, mutation_scale)

                # Assurer que la valeur reste dans les limites
                new_value = max(range_info["min"], min(range_info["max"], new_value))

                # Conversion au type approprié
                if range_info["type"] == "int":
                    new_value = int(round(new_value))

                mutated[param] = new_value

        return mutated

    def calculate_population_diversity(self, population):
        """Calcule la diversité de la population basée sur la variance des paramètres."""
        if not population:
            return 0

        # Extraire les valeurs de chaque paramètre
        param_values = {param: [] for param in self.param_ranges.keys()}
        for individual in population:
            for param in self.param_ranges.keys():
                param_values[param].append(individual[param])

        # Calculer la variance normalisée pour chaque paramètre
        normalized_variances = []
        for param, values in param_values.items():
            range_info = self.param_ranges[param]
            range_width = range_info["max"] - range_info["min"]
            if range_width > 0:
                variance = np.var(values) / (range_width ** 2)
                normalized_variances.append(variance)

        # Moyenne des variances normalisées comme mesure de diversité
        if normalized_variances:
            return np.mean(normalized_variances)
        return 0

    def select_parents(self, population, fitness_scores):
        """Sélectionne des parents en utilisant la méthode du tournoi."""
        tournament_size = 3
        parents = []

        for _ in range(2):  # Sélectionner deux parents
            tournament_indices = np.random.choice(len(population), tournament_size, replace=False)
            tournament_fitness = [fitness_scores[i] for i in tournament_indices]
            # Sélectionner le meilleur (celui avec le fitness le plus bas)
            winner_idx = tournament_indices[np.argmin(tournament_fitness)]
            parents.append(population[winner_idx])

        return parents

    def optimize(self):
        """Exécute le processus d'optimisation génétique avec stratégie adaptative avancée."""
        logger.info("Démarrage de l'optimisation génétique avec stratégie adaptative")

        # Initialisation de la population
        population = self.initialize_population()

        # Historique des améliorations pour l'analyse de tendance
        fitness_history = []
        improvement_rates = []

        for generation in range(self.generations):
            generation_start_time = time.time()

            # Calcul du facteur de mutation actuel (diminue avec les générations)
            mutation_factor = self.mutation_factor_initial - (
                (self.mutation_factor_initial - self.mutation_factor_final) *
                generation / (self.generations - 1)
            ) if self.generations > 1 else self.mutation_factor_final

            logger.info(f"Génération {generation+1}/{self.generations} - Facteur de mutation: {mutation_factor:.3f}")

            # Essayer ThreadPoolExecutor
            try:
                with concurrent.futures.ThreadPoolExecutor(max_workers=self.n_workers) as executor:
                    fitness_results = self.evaluate_population_batch(population, executor)
            except Exception as e:
                logger.warning(f"ThreadPoolExecutor a échoué ({str(e)}), utilisation d'évaluation séquentielle")
                fitness_results = []
                for ind in tqdm(population, desc="Évaluation séquentielle", disable=not self.verbose):
                    result, individual = evaluate_individual((ind, self.df, self.X_entropy_scaled,
                                                            self.entropy_df, self.src_ips, self.sample_size))
                    result["individual"] = individual.copy()
                    fitness_results.append(result)

            # Tri des individus par fitness (du meilleur au pire)
            fitness_results.sort(key=lambda x: x["fitness"])

            # Sauvegarde du meilleur individu de cette génération
            best_of_gen = fitness_results[0]
            self.best_solutions.append(best_of_gen)
            self.all_results.extend(fitness_results)

            # Calcul du taux d'amélioration relatif
            current_fitness = best_of_gen["fitness"]
            fitness_history.append(current_fitness)

            # Calcul de la tendance d'amélioration sur une fenêtre glissante
            improvement_trend = 0
            if len(fitness_history) >= self.improvement_window:
                window_start = max(0, len(fitness_history) - self.improvement_window)
                window_fitnesses = fitness_history[window_start:]

                if window_fitnesses[0] > 0:  # Éviter division par zéro
                    improvement_trend = (window_fitnesses[0] - window_fitnesses[-1]) / window_fitnesses[0]

                improvement_rates.append(improvement_trend)
                logger.info(f"Tendance d'amélioration sur {self.improvement_window} générations: {improvement_trend:.6f}")

            # Vérification et mise à jour de la meilleure solution sans FN
            for result in fitness_results:
                if result.get('fn', float('inf')) == 0 and result.get('fp', float('inf')) < self.min_fp:
                    self.min_fp = result.get('fp', float('inf'))
                    self.best_no_fn_solution = result
                    logger.info(f"Nouvelle meilleure solution sans FN: FP={self.min_fp}")

            # Affichage des résultats de la génération
            logger.info(f"Meilleur individu: {best_of_gen['individual']}")
            logger.info(f"Fitness: {best_of_gen['fitness']:.6f}, FN: {best_of_gen.get('fn', 'N/A')}, FP: {best_of_gen.get('fp', 'N/A')}")

            # Vérification d'early stopping améliorée
            if abs(self.last_best_fitness - best_of_gen["fitness"]) < 1e-6:
                self.stagnation_counter += 1
                logger.info(f"Stagnation détectée: {self.stagnation_counter}/{self.stagnation_limit}")
            else:
                # Réinitialiser le compteur seulement si l'amélioration est significative
                if (self.last_best_fitness - best_of_gen["fitness"]) / max(1e-10, self.last_best_fitness) > 0.001:
                    self.stagnation_counter = 0
                else:
                    # Incrémenter plus lentement pour les petites améliorations
                    self.stagnation_counter = max(0, self.stagnation_counter - 1)

                self.last_best_fitness = best_of_gen["fitness"]

            # Early stopping intelligent - considère à la fois stagnation et tendance d'amélioration
            if self.stagnation_counter >= self.stagnation_limit:
                if len(improvement_rates) > 5 and np.mean(improvement_rates[-5:]) < 0.0001:
                    logger.info(f"Early stopping activé après {generation+1} générations sans amélioration significative")
                    break
                else:
                    # Donner une chance supplémentaire même en cas de stagnation si la tendance est positive
                    logger.info(f"Stagnation détectée mais poursuite de l'optimisation (tendance positive)")

            # Affichage de la meilleure solution sans FN
            if self.best_no_fn_solution:
                logger.info(f"Meilleure solution sans FN: FP={self.best_no_fn_solution['fp']}")

            # Arrêt précoce si objectif atteint
            if best_of_gen.get('fn', float('inf')) == 0 and best_of_gen.get('fp', float('inf')) < 30000:
                logger.info(f"OBJECTIF ATTEINT à la génération {generation+1}! FN=0, FP={best_of_gen.get('fp')}")
                print(f"✅ OBJECTIF ATTEINT à la génération {generation+1}! FN=0, FP={best_of_gen.get('fp')}")
                break

            # Création de la nouvelle population
            new_population = []

            # Élitisme: conserver les meilleurs individus
            elites = [res["individual"] for res in fitness_results[:self.elite_size]]
            new_population.extend(elites)

            # Ajouter toujours la meilleure solution sans FN si elle existe et n'est pas déjà présente
            if self.best_no_fn_solution:
                best_no_fn_individual = self.best_no_fn_solution["individual"]
                if best_no_fn_individual not in new_population and len(new_population) < self.population_size:
                    new_population.append(best_no_fn_individual)

            # Calcul de la diversité pour ajuster les mutations
            population_diversity = self.calculate_population_diversity(population)
            logger.info(f"Diversité de la population: {population_diversity:.6f}")

            # Générer le reste de la population par sélection, croisement et mutation
            fitness_scores = [res["fitness"] for res in fitness_results]

            # Mode d'exploration ou exploitation basé sur les résultats actuels
            exploration_mode = population_diversity < self.diversity_threshold or self.stagnation_counter > 5

            while len(new_population) < self.population_size:
                # Stratégies différentes selon le mode
                if exploration_mode and np.random.random() < 0.3:  # 30% de chance en mode exploration
                    # Générer un individu complètement nouveau
                    new_individual = {}
                    for param, range_info in self.param_ranges.items():
                        if range_info["type"] == "int":
                            value = np.random.randint(range_info["min"], range_info["max"] + 1)
                        else:  # float
                            value = np.random.uniform(range_info["min"], range_info["max"])
                        new_individual[param] = value
                    new_population.append(new_individual)
                else:
                    # Sélection des parents - préférer les bons individus mais maintenir la diversité
                    if np.random.random() < 0.8:  # 80% du temps, parent1 dans le top 20%
                        idx1 = np.random.randint(0, max(1, int(len(population) * 0.2)))
                        parent1 = population[idx1]
                    else:  # 20% du temps, parent1 aléatoire dans toute la population
                        idx1 = np.random.randint(0, len(population))
                        parent1 = population[idx1]

                    # Pour parent2, favoriser la diversité
                    if np.random.random() < 0.6:  # 60% du temps, parent2 dans la moitié inférieure
                        idx2 = np.random.randint(len(population)//2, len(population))
                        parent2 = population[idx2]
                    else:  # 40% du temps, parent2 aléatoire
                        idx2 = np.random.randint(0, len(population))
                        parent2 = population[idx2]

                    # Croisement
                    child = self.crossover(parent1, parent2)

                    # Mutation adaptative avec la tendance d'amélioration
                    is_elite = child in elites
                    child = self.mutate(child, mutation_factor, population_diversity,
                                       improvement_trend=improvement_trend if improvement_rates else 0,
                                       is_elite=is_elite)

                    new_population.append(child)

            # Mise à jour de la population
            population = new_population

            # Ajustement adaptatif - si stagnation prolongée, réinitialiser partiellement la population
            if self.stagnation_counter > self.stagnation_limit // 2:
                # Conserver les meilleurs individus, réinitialiser une partie des autres
                num_to_keep = max(self.elite_size + 1, int(self.population_size * 0.3))
                keep_population = population[:num_to_keep]

                # Réinitialiser progressivement le reste
                fresh_population = self.initialize_population()
                reset_population = fresh_population[:(self.population_size - num_to_keep)]

                # Combiner
                population = keep_population + reset_population
                logger.info(f"Stagnation prolongée: réinitialisation partielle de la population ({num_to_keep} conservés)")

            generation_time = time.time() - generation_start_time
            logger.info(f"Temps de la génération: {generation_time:.2f} secondes")

            # Sauvegarde intermédiaire des résultats
            if generation % 5 == 0:  # Sauvegarder tous les 5 générations pour éviter trop d'I/O
                self.save_results()

        # Sauvegarde finale
        self.save_results()

        # Analyse des performances
        logger.info("=== Analyse finale des performances ===")
        if len(fitness_history) > 1:
            overall_improvement = (fitness_history[0] - fitness_history[-1]) / fitness_history[0] if fitness_history[0] > 0 else 0
            logger.info(f"Amélioration globale: {overall_improvement:.2%}")

        # Retourner la meilleure solution sans FN si elle existe, sinon la meilleure globale
        if self.best_no_fn_solution:
            logger.info(f"Meilleure solution finale (sans FN): FP={self.best_no_fn_solution['fp']}")
            return self.best_no_fn_solution
        else:
            # Tri final des meilleurs résultats
            self.best_solutions.sort(key=lambda x: x["fitness"])
            return self.best_solutions[0]

    def save_results(self):
        """Sauvegarde les résultats de l'optimisation."""
        # Chemin pour les résultats
        results_path = os.path.join(self.output_dir, "optimization_results.csv")
        best_path = os.path.join(self.output_dir, "best_solutions.csv")

        # Préparation des données pour les meilleurs résultats
        best_df = pd.DataFrame([
            {
                **res["individual"],
                "fitness": res.get("fitness", float('inf')),
                "fn": res.get("fn", float('nan')),
                "fp": res.get("fp", float('nan')),
                "tn": res.get("tn", float('nan')),
                "tp": res.get("tp", float('nan')),
                "time": res.get("time", float('nan'))
            }
            for res in self.best_solutions
        ])

        # Préparation des données pour tous les résultats
        all_df = pd.DataFrame([
            {
                **res.get("individual", {}),
                "fitness": res.get("fitness", float('inf')),
                "fn": res.get("fn", float('nan')),
                "fp": res.get("fp", float('nan')),
                "tn": res.get("tn", float('nan')),
                "tp": res.get("tp", float('nan')),
                "time": res.get("time", float('nan')),
                "error": res.get("error", "")
            }
            for res in self.all_results
        ])

        # Sauvegarde
        best_df.to_csv(best_path, index=False)
        all_df.to_csv(results_path, index=False)

        logger.info(f"Résultats sauvegardés dans {self.output_dir}")

    def validate_best_solution(self, best_solution):
        """Valide le meilleur hyperparamètre sur l'ensemble du jeu de données."""
        logger.info("Validation de la meilleure solution sur l'ensemble du jeu de données...")

        full_size = self.sample_size * 5  # Utiliser un échantillon plus grand pour validation

        try:
            model = EnhancedKGB(
                sample_size=full_size,
                n_major_components=int(best_solution["individual"]["n_major_components"]),
                threshold_PCA=float(best_solution["individual"]["threshold_PCA"]),
                threshold_major=float(best_solution["individual"]["threshold_major"]),
                threshold_minor=float(best_solution["individual"]["threshold_minor"]),
                verbose=True,
                random_state=42
            )

            model.load_and_preprocess_data(self.file_path)
            model.compute_entropies()
            model.detect_anomalies()
            tn, fp, fn, tp = model.compute_confusion_matrix()

            logger.info(f"Validation finale => TN={tn}, FP={fp}, FN={fn}, TP={tp}")

            # Sauvegarde des résultats de validation
            validation_path = os.path.join(self.output_dir, "validation_results.txt")
            with open(validation_path, 'w') as f:
                f.write(f"Meilleurs hyperparamètres:\n")
                for param, value in best_solution["individual"].items():
                    f.write(f"{param}: {value}\n")
                f.write(f"\nRésultats de validation:\n")
                f.write(f"TN: {tn}\n")
                f.write(f"FP: {fp}\n")
                f.write(f"FN: {fn}\n")
                f.write(f"TP: {tp}\n")

                if fn == 0:
                    f.write("\nObjectif atteint: Aucun faux négatif!\n")
                    if fp <= 30000:
                        f.write(f"✅ OBJECTIF ATTEINT: Faux positifs ({fp}) inférieurs à 100 000!\n")
                    else:
                        f.write(f"⚠️ Faux positifs ({fp}) supérieurs à l'objectif de 100 000\n")
                else:
                    f.write(f"\nAttention: {fn} faux négatifs détectés.\n")

            return {"tn": tn, "fp": fp, "fn": fn, "tp": tp}

        except Exception as e:
            logger.error(f"Erreur lors de la validation: {e}")
            return {"error": str(e)}

    def create_pareto_front(self):
        """Visualise le front de Pareto entre les faux positifs et faux négatifs."""
        try:
            results_path = os.path.join(self.output_dir, "optimization_results.csv")
            if not os.path.exists(results_path):
                logger.info(f"Fichier non trouvé: {results_path}")
                return

            df = pd.read_csv(results_path)

            # Vérifier si les colonnes nécessaires existent
            if not all(col in df.columns for col in ["fn", "fp"]):
                logger.info("Colonnes fn et/ou fp manquantes.")
                return

            # Filtrer les lignes valides
            df = df[~df['fn'].isna() & ~df['fp'].isna()]

            plt.figure(figsize=(12, 10))

            # Scatter plot des solutions
            plt.scatter(df['fp'], df['fn'], s=50, alpha=0.7, c=df['fitness'], cmap='viridis',
                       label="Solutions évaluées")

            # Marquer les solutions avec FN = 0 (notre objectif principal)
            zero_fn = df[df['fn'] == 0]
            if not zero_fn.empty:
                best_zero_fn = zero_fn.loc[zero_fn['fp'].idxmin()]
                plt.scatter(zero_fn['fp'], zero_fn['fn'], s=100, color='green', marker='*',
                           label="Solutions avec FN = 0")

                plt.scatter([best_zero_fn['fp']], [best_zero_fn['fn']], s=200, color='blue', marker='X',
                           label=f"Meilleure solution (FP={best_zero_fn['fp']}, FN=0)")

            # Ajouter des lignes pour les objectifs
            plt.axhline(y=0, color='green', linestyle='--', alpha=0.7, label="Objectif FN = 0")
            plt.axvline(x=30000, color='red', linestyle='--', alpha=0.7, label="Objectif FP ≤ 100 000")

            # Région idéale
            plt.fill_between([0, 30000], [0, 0], color='green', alpha=0.1, label="Zone objectif")

            plt.xlabel("Faux Positifs (FP)")
            plt.ylabel("Faux Négatifs (FN)")
            plt.title("Front de Pareto - Compromis entre Faux Positifs et Faux Négatifs")
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.colorbar(label="Fitness (plus bas = meilleur)")

            output_path = os.path.join(self.output_dir, "pareto_front.png")
            plt.savefig(output_path, dpi=150)
            logger.info(f"Front de Pareto sauvegardé: {output_path}")
            plt.close()

            # Créer un graphique d'évolution des FP/FN
            plt.figure(figsize=(12, 6))
            best_df = pd.DataFrame([res for res in self.best_solutions if 'fp' in res and 'fn' in res])
            generations = range(1, len(best_df) + 1)

            if not best_df.empty:
                plt.subplot(1, 2, 1)
                plt.plot(generations, [res.get('fp', float('nan')) for res in best_df], 'b-', label='Faux Positifs')
                plt.axhline(y=30000, color='r', linestyle='--', label='Objectif FP ≤ 30 000')
                plt.xlabel('Génération')
                plt.ylabel('Faux Positifs')
                plt.grid(True, alpha=0.3)
                plt.legend()

                plt.subplot(1, 2, 2)
                plt.plot(generations, [res.get('fn', float('nan')) for res in best_df], 'r-', label='Faux Négatifs')
                plt.axhline(y=0, color='g', linestyle='--', label='Objectif FN = 0')
                plt.xlabel('Génération')
                plt.ylabel('Faux Négatifs')
                plt.grid(True, alpha=0.3)
                plt.legend()

                plt.tight_layout()
                evolution_path = os.path.join(self.output_dir, "error_evolution.png")
                plt.savefig(evolution_path, dpi=150)
                plt.close()

        except Exception as e:
            logger.error(f"Erreur lors de la création du front de Pareto: {e}")


def run_genetic_optimization(file_path, sample_size=1000000, population_size=80, generations=400, n_workers=None, verbose=True):
    """Fonction principale pour exécuter l'optimisation génétique avec paramètres optimisés."""
    start_time = time.time()

    # Utiliser tous les CPUs disponibles si n_workers n'est pas spécifié
    if n_workers is None:
        n_workers = os.cpu_count()

    logger.info(f"Démarrage de l'optimisation avec fichier={file_path}, sample_size={sample_size}, n_workers={n_workers}")
    print(f"=== OPTIMISATION DES HYPERPARAMÈTRES ===")
    print(f"Fichier: {file_path}")
    print(f"Taille de l'échantillon: {sample_size}")
    print(f"Taille de la population: {population_size}")
    print(f"Nombre de générations: {generations}")
    print(f"Nombre de workers: {n_workers}")
    print(f"Meilleure solution de départ: n_major_components=16, threshold_PCA=0.99257, threshold_major=0.99968, threshold_minor=0.98109")
    print(f"Plages de recherche très resserrées autour des meilleures valeurs connues")

    optimizer = GeneticOptimizer(
        file_path=file_path,
        sample_size=sample_size,
        population_size=population_size,
        generations=generations,
        mutation_factor_initial=0.05,    # Mutations très fines
        mutation_factor_final=0.005,     # Mutations extrêmement fines à la fin
        crossover_prob=0.8,             # Favoriser le croisement
        elite_size=4,                   # Préserver plus d'élites
        stagnation_limit=20,            # Plus de patience avant d'arrêter
        diversity_threshold=0.01,       # Seuil de diversité plus sensible
        improvement_window=5,           # Analyser l'amélioration sur 5 générations
        n_workers=n_workers,
        verbose=verbose
    )

    # Exécution de l'optimisation
    best_solution = optimizer.optimize()

    # Validation de la meilleure solution
    validation_results = optimizer.validate_best_solution(best_solution)

    # Générer le front de Pareto
    optimizer.create_pareto_front()

    # Affichage des résultats finaux
    logger.info("Optimisation terminée!")
    total_time = time.time() - start_time

    print("\n=== OPTIMISATION TERMINÉE ===")
    print(f"Durée totale: {total_time/60:.1f} minutes")
    print(f"Meilleurs hyperparamètres trouvés:")
    for param, value in best_solution["individual"].items():
        print(f"  {param}: {value}")

    print("\nRésultats de la validation finale:")
    if "error" not in validation_results:
        print(f"  Vrais Négatifs (TN): {validation_results['tn']}")
        print(f"  Faux Positifs (FP): {validation_results['fp']}")
        print(f"  Faux Négatifs (FN): {validation_results['fn']}")
        print(f"  Vrais Positifs (TP): {validation_results['tp']}")

        # Vérification de l'objectif principal
        if validation_results['fn'] == 0:
            print("\n✅ OBJECTIF ATTEINT: Aucun faux négatif!")
            if validation_results['fp'] <= 30000:
                print(f"✅ OBJECTIF ATTEINT: Faux positifs ({validation_results['fp']}) inférieurs à 30 000!")
            else:
                print(f"⚠️ Faux positifs ({validation_results['fp']}) supérieurs à l'objectif de 30 000")
        else:
            print(f"\n❌ ATTENTION: {validation_results['fn']} faux négatifs détectés.")
    else:
        print(f"  Erreur: {validation_results['error']}")

    print(f"\nTous les résultats ont été sauvegardés dans: {optimizer.output_dir}")

    return best_solution, optimizer.output_dir


if __name__ == "__main__":
    # Fichier de données (remplacer par votre chemin)
    file_path = "network_packets.csv"

    # Exécution de l'optimisation avec utilisation maximale du CPU
    best_solution, output_dir = run_genetic_optimization(
        file_path=file_path,
        sample_size=1000000,    # Échantillon pour l'optimisation
        population_size=50,     # Population plus large
        generations=300,        # Beaucoup plus de générations
        n_workers=None,         # Utilise tous les CPUs disponibles
        verbose=True
    )