

from typing import Dict, Tuple, Optional
from enum import Enum
import math


# ============================================================================
# ÉNUMÉRATIONS ET CONSTANTES
# ============================================================================

class Gender(Enum):
    """Énumération pour le genre"""
    MALE = "male"
    FEMALE = "female"


class ActivityLevel(Enum):
    """Niveaux d'activité physique avec facteurs TDEE"""
    SEDENTARY = ("sedentary", 1.2, "Peu ou pas d'exercice")
    LIGHTLY_ACTIVE = ("lightly_active", 1.375, "Exercice léger 1-3 jours/semaine")
    MODERATELY_ACTIVE = ("moderately_active", 1.55, "Exercice modéré 3-5 jours/semaine")
    VERY_ACTIVE = ("very_active", 1.725, "Exercice intense 6-7 jours/semaine")
    EXTRA_ACTIVE = ("extra_active", 1.9, "Exercice très intense, travail physique")
    
    def __init__(self, key, factor, description):
        self.key = key
        self.factor = factor
        self.description = description


class FitnessGoal(Enum):
    """Objectifs de fitness avec ajustements caloriques"""
    WEIGHT_LOSS = ("weight_loss", -500, "Perte de poids")
    MODERATE_WEIGHT_LOSS = ("moderate_weight_loss", -250, "Perte de poids modérée")
    MAINTENANCE = ("maintenance", 0, "Maintien du poids")
    MUSCLE_GAIN = ("muscle_gain", 300, "Prise de masse musculaire")
    BULKING = ("bulking", 500, "Prise de masse importante")
    
    def __init__(self, key, calorie_adjustment, description):
        self.key = key
        self.calorie_adjustment = calorie_adjustment
        self.description = description


class BMICategory(Enum):
    """Catégories IMC selon l'OMS"""
    SEVERELY_UNDERWEIGHT = ("severely_underweight", 0, 16, "Maigreur sévère", "🔴")
    UNDERWEIGHT = ("underweight", 16, 18.5, "Insuffisance pondérale", "🟡")
    NORMAL = ("normal", 18.5, 25, "Corpulence normale", "🟢")
    OVERWEIGHT = ("overweight", 25, 30, "Surpoids", "🟡")
    OBESE_CLASS_I = ("obese_1", 30, 35, "Obésité modérée", "🟠")
    OBESE_CLASS_II = ("obese_2", 35, 40, "Obésité sévère", "🔴")
    OBESE_CLASS_III = ("obese_3", 40, 100, "Obésité morbide", "🔴")
    
    def __init__(self, key, min_bmi, max_bmi, description, indicator):
        self.key = key
        self.min_bmi = min_bmi
        self.max_bmi = max_bmi
        self.description = description
        self.indicator = indicator


# ============================================================================
# CLASSE PRINCIPALE DE CALCULS
# ============================================================================

class PhysiologicalCalculator:
    """
    Calculateur de métriques physiologiques pour le fitness.
    
    Cette classe encapsule tous les calculs nécessaires pour évaluer
    les besoins énergétiques et la santé d'un individu.
    """
    
    # Constantes de validation
    MIN_AGE = 15
    MAX_AGE = 100
    MIN_WEIGHT = 30  # kg
    MAX_WEIGHT = 300  # kg
    MIN_HEIGHT = 1.20  # m
    MAX_HEIGHT = 2.50  # m
    
    def __init__(self):
        """Initialise le calculateur"""
        pass
    
    # ========================================================================
    # VALIDATION DES ENTRÉES
    # ========================================================================
    
    @staticmethod
    def validate_age(age: int) -> Tuple[bool, str]:
        """
        Valide l'âge.
        
        Args:
            age: Âge en années
            
        Returns:
            Tuple (est_valide, message_erreur)
        """
        if not isinstance(age, (int, float)):
            return False, "L'âge doit être un nombre"
        
        if age < PhysiologicalCalculator.MIN_AGE:
            return False, f"L'âge doit être au moins {PhysiologicalCalculator.MIN_AGE} ans"
        
        if age > PhysiologicalCalculator.MAX_AGE:
            return False, f"L'âge ne peut pas dépasser {PhysiologicalCalculator.MAX_AGE} ans"
        
        return True, ""
    
    @staticmethod
    def validate_weight(weight: float) -> Tuple[bool, str]:
        """
        Valide le poids.
        
        Args:
            weight: Poids en kg
            
        Returns:
            Tuple (est_valide, message_erreur)
        """
        if not isinstance(weight, (int, float)):
            return False, "Le poids doit être un nombre"
        
        if weight < PhysiologicalCalculator.MIN_WEIGHT:
            return False, f"Le poids doit être au moins {PhysiologicalCalculator.MIN_WEIGHT} kg"
        
        if weight > PhysiologicalCalculator.MAX_WEIGHT:
            return False, f"Le poids ne peut pas dépasser {PhysiologicalCalculator.MAX_WEIGHT} kg"
        
        return True, ""
    
    @staticmethod
    def validate_height(height: float) -> Tuple[bool, str]:
        """
        Valide la taille.
        
        Args:
            height: Taille en mètres
            
        Returns:
            Tuple (est_valide, message_erreur)
        """
        if not isinstance(height, (int, float)):
            return False, "La taille doit être un nombre"
        
        if height < PhysiologicalCalculator.MIN_HEIGHT:
            return False, f"La taille doit être au moins {PhysiologicalCalculator.MIN_HEIGHT} m"
        
        if height > PhysiologicalCalculator.MAX_HEIGHT:
            return False, f"La taille ne peut pas dépasser {PhysiologicalCalculator.MAX_HEIGHT} m"
        
        return True, ""
    
    @staticmethod
    def validate_gender(gender: str) -> Tuple[bool, str]:
        """
        Valide le genre.
        
        Args:
            gender: Genre ("male" ou "female")
            
        Returns:
            Tuple (est_valide, message_erreur)
        """
        valid_genders = [g.value for g in Gender]
        gender_lower = gender.lower()
        
        if gender_lower not in valid_genders:
            return False, f"Le genre doit être 'male' ou 'female'"
        
        return True, ""
    
    # ========================================================================
    # CALCULS IMC
    # ========================================================================
    
    @staticmethod
    def calculate_bmi(weight: float, height: float) -> float:
        """
        Calcule l'Indice de Masse Corporelle (IMC).
        
        Formule: IMC = Poids(kg) / (Taille(m))²
        
        Args:
            weight: Poids en kg
            height: Taille en mètres
            
        Returns:
            IMC arrondi à 2 décimales
            
        Raises:
            ValueError: Si les valeurs ne sont pas valides
        """
        # Validation
        valid_weight, msg_weight = PhysiologicalCalculator.validate_weight(weight)
        if not valid_weight:
            raise ValueError(msg_weight)
        
        valid_height, msg_height = PhysiologicalCalculator.validate_height(height)
        if not valid_height:
            raise ValueError(msg_height)
        
        # Calcul
        bmi = weight / (height ** 2)
        return round(bmi, 2)
    
    @staticmethod
    def get_bmi_category(bmi: float) -> BMICategory:
        """
        Détermine la catégorie IMC selon l'OMS.
        
        Args:
            bmi: Valeur de l'IMC
            
        Returns:
            BMICategory correspondante
        """
        for category in BMICategory:
            if category.min_bmi <= bmi < category.max_bmi:
                return category
        
        # Par défaut, retourner la dernière catégorie
        return BMICategory.OBESE_CLASS_III
    
    @staticmethod
    def get_bmi_interpretation(bmi: float) -> Dict:
        """
        Retourne une interprétation complète de l'IMC.
        
        Args:
            bmi: Valeur de l'IMC
            
        Returns:
            Dictionnaire avec catégorie, description, recommandations
        """
        category = PhysiologicalCalculator.get_bmi_category(bmi)
        
        # Recommandations selon la catégorie
        recommendations = {
            BMICategory.SEVERELY_UNDERWEIGHT: "Consultez un professionnel de santé. Risque élevé pour la santé.",
            BMICategory.UNDERWEIGHT: "Envisagez d'augmenter votre apport calorique de manière saine.",
            BMICategory.NORMAL: "Continuez vos bonnes habitudes! Maintenez un mode de vie équilibré.",
            BMICategory.OVERWEIGHT: "Adoptez une alimentation équilibrée et augmentez votre activité physique.",
            BMICategory.OBESE_CLASS_I: "Consultez un nutritionniste. Commencez un programme de perte de poids.",
            BMICategory.OBESE_CLASS_II: "Consultation médicale recommandée. Programme de perte de poids supervisé.",
            BMICategory.OBESE_CLASS_III: "Consultation médicale urgente. Risque sérieux pour la santé."
        }
        
        return {
            "bmi": bmi,
            "category": category.description,
            "indicator": category.indicator,
            "range": f"{category.min_bmi} - {category.max_bmi}",
            "recommendation": recommendations[category]
        }
    
    # ========================================================================
    # CALCULS BMR (BASAL METABOLIC RATE)
    # ========================================================================
    
    @staticmethod
    def calculate_bmr(weight: float, height: float, age: int, gender: str) -> float:
        """
        Calcule le Métabolisme de Base (BMR) selon la formule Mifflin-St Jeor.
        
        Formules:
        - Homme: BMR = 10 × Poids(kg) + 6.25 × Taille(cm) - 5 × Âge + 5
        - Femme: BMR = 10 × Poids(kg) + 6.25 × Taille(cm) - 5 × Âge - 161
        
        Args:
            weight: Poids en kg
            height: Taille en mètres
            age: Âge en années
            gender: Genre ("male" ou "female")
            
        Returns:
            BMR en calories/jour, arrondi à 0 décimales
            
        Raises:
            ValueError: Si les valeurs ne sont pas valides
        """
        # Validations
        valid_weight, msg_weight = PhysiologicalCalculator.validate_weight(weight)
        if not valid_weight:
            raise ValueError(msg_weight)
        
        valid_height, msg_height = PhysiologicalCalculator.validate_height(height)
        if not valid_height:
            raise ValueError(msg_height)
        
        valid_age, msg_age = PhysiologicalCalculator.validate_age(age)
        if not valid_age:
            raise ValueError(msg_age)
        
        valid_gender, msg_gender = PhysiologicalCalculator.validate_gender(gender)
        if not valid_gender:
            raise ValueError(msg_gender)
        
        # Conversion taille en cm
        height_cm = height * 100
        
        # Calcul selon le genre
        gender_lower = gender.lower()
        
        if gender_lower == Gender.MALE.value:
            bmr = (10 * weight) + (6.25 * height_cm) - (5 * age) + 5
        else:  # female
            bmr = (10 * weight) + (6.25 * height_cm) - (5 * age) - 161
        
        return round(bmr, 0)
    
    # ========================================================================
    # CALCULS TDEE (TOTAL DAILY ENERGY EXPENDITURE)
    # ========================================================================
    
    @staticmethod
    def calculate_tdee(bmr: float, activity_level: str) -> float:
        """
        Calcule la Dépense Énergétique Journalière Totale (TDEE).
        
        TDEE = BMR × Facteur d'activité
        
        Args:
            bmr: Métabolisme de base (BMR)
            activity_level: Niveau d'activité (voir ActivityLevel)
            
        Returns:
            TDEE en calories/jour, arrondi à 0 décimales
            
        Raises:
            ValueError: Si le niveau d'activité est invalide
        """
        # Trouver le niveau d'activité correspondant
        activity_level_lower = activity_level.lower()
        
        for level in ActivityLevel:
            if level.key == activity_level_lower:
                tdee = bmr * level.factor
                return round(tdee, 0)
        
        # Si niveau non trouvé
        valid_levels = [level.key for level in ActivityLevel]
        raise ValueError(f"Niveau d'activité invalide. Valeurs valides: {valid_levels}")
    
    @staticmethod
    def get_activity_level_info() -> Dict:
        """
        Retourne les informations sur tous les niveaux d'activité.
        
        Returns:
            Dictionnaire avec toutes les infos des niveaux d'activité
        """
        return {
            level.key: {
                "factor": level.factor,
                "description": level.description
            }
            for level in ActivityLevel
        }
    
    # ========================================================================
    # RECOMMANDATIONS CALORIQUES SELON OBJECTIFS
    # ========================================================================
    
    @staticmethod
    def calculate_target_calories(tdee: float, goal: str) -> Dict:
        """
        Calcule les calories cibles selon l'objectif fitness.
        
        Args:
            tdee: Dépense énergétique totale
            goal: Objectif (voir FitnessGoal)
            
        Returns:
            Dictionnaire avec calories cibles et macronutriments
            
        Raises:
            ValueError: Si l'objectif est invalide
        """
        # Trouver l'objectif correspondant
        goal_lower = goal.lower()
        
        for fitness_goal in FitnessGoal:
            if fitness_goal.key == goal_lower:
                target_calories = tdee + fitness_goal.calorie_adjustment
                
                # Calcul des macronutriments (protéines, glucides, lipides)
                # Ratios standards selon l'objectif
                if fitness_goal in [FitnessGoal.WEIGHT_LOSS, FitnessGoal.MODERATE_WEIGHT_LOSS]:
                    # 40% protéines, 30% glucides, 30% lipides
                    protein_ratio, carbs_ratio, fat_ratio = 0.40, 0.30, 0.30
                elif fitness_goal in [FitnessGoal.MUSCLE_GAIN, FitnessGoal.BULKING]:
                    # 30% protéines, 45% glucides, 25% lipides
                    protein_ratio, carbs_ratio, fat_ratio = 0.30, 0.45, 0.25
                else:  # MAINTENANCE
                    # 30% protéines, 40% glucides, 30% lipides
                    protein_ratio, carbs_ratio, fat_ratio = 0.30, 0.40, 0.30
                
                # Conversion en grammes (1g protéine = 4 cal, 1g glucides = 4 cal, 1g lipides = 9 cal)
                protein_g = round((target_calories * protein_ratio) / 4, 0)
                carbs_g = round((target_calories * carbs_ratio) / 4, 0)
                fat_g = round((target_calories * fat_ratio) / 9, 0)
                
                return {
                    "goal": fitness_goal.description,
                    "tdee": round(tdee, 0),
                    "adjustment": fitness_goal.calorie_adjustment,
                    "target_calories": round(target_calories, 0),
                    "macros": {
                        "protein_g": protein_g,
                        "carbs_g": carbs_g,
                        "fat_g": fat_g,
                        "protein_percent": round(protein_ratio * 100, 0),
                        "carbs_percent": round(carbs_ratio * 100, 0),
                        "fat_percent": round(fat_ratio * 100, 0)
                    }
                }
        
        # Si objectif non trouvé
        valid_goals = [goal.key for goal in FitnessGoal]
        raise ValueError(f"Objectif invalide. Valeurs valides: {valid_goals}")
    
    # ========================================================================
    # FONCTION COMPLÈTE - PROFIL UTILISATEUR
    # ========================================================================
    
    @staticmethod
    def calculate_complete_profile(
        age: int,
        gender: str,
        weight: float,
        height: float,
        activity_level: str,
        goal: str
    ) -> Dict:
        """
        Calcule un profil physiologique complet pour l'utilisateur.
        
        Args:
            age: Âge en années
            gender: Genre ("male" ou "female")
            weight: Poids en kg
            height: Taille en mètres
            activity_level: Niveau d'activité
            goal: Objectif fitness
            
        Returns:
            Dictionnaire complet avec tous les calculs et recommandations
            
        Raises:
            ValueError: Si une des valeurs est invalide
        """
        calc = PhysiologicalCalculator()
        
        # Calculs de base
        bmi = calc.calculate_bmi(weight, height)
        bmi_info = calc.get_bmi_interpretation(bmi)
        
        bmr = calc.calculate_bmr(weight, height, age, gender)
        
        tdee = calc.calculate_tdee(bmr, activity_level)
        
        calories_info = calc.calculate_target_calories(tdee, goal)
        
        # Poids idéal (pour IMC = 22, milieu de la zone normale)
        ideal_bmi = 22
        ideal_weight = round(ideal_bmi * (height ** 2), 1)
        weight_difference = round(weight - ideal_weight, 1)
        
        # Assemblage du profil complet
        profile = {
            "user_info": {
                "age": age,
                "gender": gender.capitalize(),
                "weight": weight,
                "height": height,
                "activity_level": activity_level,
                "goal": goal
            },
            "bmi": bmi_info,
            "bmr": {
                "value": bmr,
                "description": "Métabolisme de base (calories au repos)"
            },
            "tdee": {
                "value": tdee,
                "description": "Dépense énergétique journalière totale"
            },
            "nutrition": calories_info,
            "weight_analysis": {
                "current": weight,
                "ideal": ideal_weight,
                "difference": weight_difference,
                "status": "au dessus" if weight_difference > 0 else "en dessous" if weight_difference < 0 else "idéal"
            }
        }
        
        return profile
    
    @staticmethod
    def format_profile_report(profile: Dict) -> str:
        """
        Formate le profil en un rapport texte lisible.
        
        Args:
            profile: Profil calculé par calculate_complete_profile
            
        Returns:
            Rapport formaté en texte
        """
        report = []
        report.append("=" * 60)
        report.append("🏋️  PROFIL PHYSIOLOGIQUE FITBOX")
        report.append("=" * 60)
        
        # Informations utilisateur
        user = profile["user_info"]
        report.append(f"\n👤 INFORMATIONS UTILISATEUR")
        report.append(f"   Âge: {user['age']} ans")
        report.append(f"   Genre: {user['gender']}")
        report.append(f"   Poids: {user['weight']} kg")
        report.append(f"   Taille: {user['height']} m")
        report.append(f"   Niveau d'activité: {user['activity_level']}")
        report.append(f"   Objectif: {user['goal']}")
        
        # IMC
        bmi = profile["bmi"]
        report.append(f"\n📊 INDICE DE MASSE CORPORELLE (IMC)")
        report.append(f"   Valeur: {bmi['bmi']} {bmi['indicator']}")
        report.append(f"   Catégorie: {bmi['category']}")
        report.append(f"   Plage normale: {bmi['range']}")
        report.append(f"   💡 {bmi['recommendation']}")
        
        # Analyse du poids
        weight = profile["weight_analysis"]
        report.append(f"\n⚖️  ANALYSE DU POIDS")
        report.append(f"   Poids actuel: {weight['current']} kg")
        report.append(f"   Poids idéal: {weight['ideal']} kg")
        report.append(f"   Différence: {abs(weight['difference'])} kg ({weight['status']})")
        
        # Métabolisme
        bmr = profile["bmr"]
        tdee = profile["tdee"]
        report.append(f"\n🔥 MÉTABOLISME")
        report.append(f"   BMR: {bmr['value']} cal/jour")
        report.append(f"   ({bmr['description']})")
        report.append(f"   TDEE: {tdee['value']} cal/jour")
        report.append(f"   ({tdee['description']})")
        
        # Nutrition
        nutrition = profile["nutrition"]
        report.append(f"\n🍽️  PLAN NUTRITIONNEL - {nutrition['goal'].upper()}")
        report.append(f"   Calories cibles: {nutrition['target_calories']} cal/jour")
        report.append(f"   Ajustement: {nutrition['adjustment']:+d} cal")
        
        macros = nutrition["macros"]
        report.append(f"\n   📈 MACRONUTRIMENTS:")
        report.append(f"      Protéines: {macros['protein_g']}g ({macros['protein_percent']}%)")
        report.append(f"      Glucides: {macros['carbs_g']}g ({macros['carbs_percent']}%)")
        report.append(f"      Lipides: {macros['fat_g']}g ({macros['fat_percent']}%)")
        
        report.append("\n" + "=" * 60)
        
        return "\n".join(report)


# ============================================================================
# FONCTIONS UTILITAIRES
# ============================================================================

def get_available_activity_levels() -> list:
    """Retourne la liste des niveaux d'activité disponibles"""
    return [(level.key, level.description) for level in ActivityLevel]


def get_available_goals() -> list:
    """Retourne la liste des objectifs disponibles"""
    return [(goal.key, goal.description) for goal in FitnessGoal]


# ============================================================================
# EXEMPLE D'UTILISATION
# ============================================================================

if __name__ == "__main__":
    print("\n🏋️  FitBox - Module de Calculs Physiologiques\n")
    
    # Exemple d'utilisation
    calc = PhysiologicalCalculator()
    
    # Calcul d'un profil complet
    try:
        profile = calc.calculate_complete_profile(
            age=25,
            gender="male",
            weight=75,
            height=1.75,
            activity_level="moderately_active",
            goal="muscle_gain"
        )
        
        # Afficher le rapport
        report = calc.format_profile_report(profile)
        print(report)
        
        # Afficher aussi les niveaux d'activité disponibles
        print("\n📋 NIVEAUX D'ACTIVITÉ DISPONIBLES:")
        for key, desc in get_available_activity_levels():
            print(f"   - {key}: {desc}")
        
        print("\n🎯 OBJECTIFS DISPONIBLES:")
        for key, desc in get_available_goals():
            print(f"   - {key}: {desc}")
        
    except ValueError as e:
        print(f"❌ Erreur: {e}")