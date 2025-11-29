"""
FitBox - Gestion des Templates de Prompts
Phase 5 - Étape 5.3
"""

from typing import Dict, List, Optional
from enum import Enum


class PromptType(Enum):
    """Types de prompts disponibles"""
    WORKOUT_PLAN = "workout_plan"
    NUTRITION_PLAN = "nutrition_plan"
    GENERAL_ADVICE = "general_advice"
    MOTIVATION = "motivation"
    EXERCISE_FORM = "exercise_form"
    INJURY_PREVENTION = "injury_prevention"
    PROGRESS_TRACKING = "progress_tracking"


class PromptTemplateManager:
    """
    Gestionnaire de templates de prompts pour FitBox.
    Fournit des prompts structurés et personnalisés selon le contexte.
    """
    
    SYSTEM_MESSAGE = """Tu es FitBox, un coach sportif et nutritionniste expert virtuel certifié.
Ta mission est d'aider les utilisateurs à atteindre leurs objectifs fitness de manière saine et durable.

TES PRINCIPES:
- Basé sur la science du sport et de la nutrition
- Personnalisé selon le profil de l'utilisateur
- Motivant et encourageant
- Pratique et actionable
- Sûr et respectueux des limitations physiques

STYLE DE RÉPONSE:
- Clair et concis
- Structuré avec des émojis appropriés
- Exemples concrets
- Pas de jargon inutile"""
    
    @staticmethod
    def format_user_context(user_data: dict, profile: dict) -> str:
        """
        Formate le contexte utilisateur pour le prompt.
        
        Args:
            user_data: Données utilisateur
            profile: Profil physiologique calculé
            
        Returns:
            Contexte formaté
        """
        
        # Niveau d'expérience
        experience_map = {
            1: "Débutant",
            2: "Intermédiaire",
            3: "Avancé"
        }
        experience = user_data.get('experience_level', 1)
        experience_label = experience_map.get(experience, "Non spécifié")
        
        # Objectif en français
        goal_map = {
            "weight_loss": "Perte de poids",
            "moderate_weight_loss": "Perte de poids modérée",
            "maintenance": "Maintien du poids",
            "muscle_gain": "Prise de masse musculaire",
            "bulking": "Prise de masse importante"
        }
        goal = user_data.get('goal', 'maintenance')
        goal_label = goal_map.get(goal, goal)
        
        # Niveau d'activité
        activity_map = {
            "sedentary": "Sédentaire",
            "lightly_active": "Légèrement actif",
            "moderately_active": "Modérément actif",
            "very_active": "Très actif",
            "extra_active": "Extrêmement actif"
        }
        activity = user_data.get('activity_level', 'moderately_active')
        activity_label = activity_map.get(activity, activity)
        
        context = f"""📋 PROFIL UTILISATEUR:
👤 Informations de base:
   - Âge: {user_data['age']} ans
   - Genre: {user_data['gender'].capitalize()}
   - Poids: {user_data['weight']} kg
   - Taille: {user_data['height']} m
   - Niveau: {experience_label}
   - Activité: {activity_label}
   
🎯 Objectif: {goal_label}

📊 DONNÉES PHYSIOLOGIQUES:
   - IMC: {profile['bmi']['bmi']} ({profile['bmi']['category']}) {profile['bmi']['indicator']}
   - BMR (Métabolisme de base): {profile['bmr']['value']:.0f} cal/jour
   - TDEE (Dépense totale): {profile['tdee']['value']:.0f} cal/jour
   - Calories cibles: {profile['nutrition']['target_calories']:.0f} cal/jour
   
🍽️ BESOINS NUTRITIONNELS:
   - Protéines: {profile['nutrition']['macros']['protein_g']:.0f}g/jour ({profile['nutrition']['macros']['protein_percent']:.0f}%)
   - Glucides: {profile['nutrition']['macros']['carbs_g']:.0f}g/jour ({profile['nutrition']['macros']['carbs_percent']:.0f}%)
   - Lipides: {profile['nutrition']['macros']['fat_g']:.0f}g/jour ({profile['nutrition']['macros']['fat_percent']:.0f}%)

⚖️ ANALYSE DU POIDS:
   - Poids actuel: {profile['weight_analysis']['current']} kg
   - Poids idéal: {profile['weight_analysis']['ideal']} kg
   - Différence: {abs(profile['weight_analysis']['difference']):.1f} kg ({profile['weight_analysis']['status']})"""
        
        return context
    
    @staticmethod
    def create_workout_plan_prompt(
        user_data: dict,
        profile: dict,
        workout_type: Optional[str] = None,
        duration_weeks: int = 1
    ) -> str:
        """
        Crée un prompt pour générer un programme d'entraînement.
        
        Args:
            user_data: Données utilisateur
            profile: Profil physiologique
            workout_type: Type d'entraînement spécifique (optionnel)
            duration_weeks: Durée du programme en semaines
            
        Returns:
            Prompt complet formaté
        """
        
        context = PromptTemplateManager.format_user_context(user_data, profile)
        
        workout_spec = ""
        if workout_type:
            workout_spec = f" de type {workout_type}"
        
        user_request = f"""Crée-moi un programme d'entraînement{workout_spec} personnalisé pour {duration_weeks} semaine(s).

STRUCTURE ATTENDUE:
📅 Programme sur {duration_weeks} semaine(s)

Pour chaque séance, inclus:
1. 🏋️ Type d'entraînement
2. ⏱️ Durée recommandée
3. 💪 Exercices principaux (3-5 exercices)
4. 📈 Séries et répétitions
5. 💡 Conseils de progression

CONSIDÈRE:
- Mon niveau actuel
- Mon objectif spécifique
- Mes capacités physiques
- La progression graduelle
- La récupération nécessaire"""
        
        prompt = f"""<|system|>
{PromptTemplateManager.SYSTEM_MESSAGE}<|end|>
<|user|>
{context}

{user_request}<|end|>
<|assistant|>
"""
        
        return prompt
    
    @staticmethod
    def create_nutrition_plan_prompt(
        user_data: dict,
        profile: dict,
        meal_count: int = 4,
        dietary_restrictions: Optional[List[str]] = None
    ) -> str:
        """
        Crée un prompt pour générer un plan nutritionnel.
        
        Args:
            user_data: Données utilisateur
            profile: Profil physiologique
            meal_count: Nombre de repas par jour
            dietary_restrictions: Restrictions alimentaires (optionnel)
            
        Returns:
            Prompt complet formaté
        """
        
        context = PromptTemplateManager.format_user_context(user_data, profile)
        
        restrictions_text = ""
        if dietary_restrictions:
            restrictions_text = f"\n\n⚠️ RESTRICTIONS ALIMENTAIRES:\n" + "\n".join(
                f"   - {r}" for r in dietary_restrictions
            )
        
        calories_per_meal = profile['nutrition']['target_calories'] / meal_count
        
        user_request = f"""Crée-moi un plan alimentaire détaillé pour une journée type avec {meal_count} repas.
{restrictions_text}

STRUCTURE ATTENDUE:
🍽️ PLAN NUTRITIONNEL JOURNALIER ({profile['nutrition']['target_calories']:.0f} calories)

Pour chaque repas (~{calories_per_meal:.0f} cal):
1. 🕐 Moment de la journée
2. 🍴 Composition du repas
3. 📊 Répartition des macros
4. 📝 Exemple de repas concret
5. 💡 Alternatives possibles

ASSURE-TOI DE:
- Respecter mes macros totales
- Proposer des aliments accessibles
- Varier les sources de nutriments
- Inclure des collations si nécessaire
- Donner des portions précises"""
        
        prompt = f"""<|system|>
{PromptTemplateManager.SYSTEM_MESSAGE}<|end|>
<|user|>
{context}

{user_request}<|end|>
<|assistant|>
"""
        
        return prompt
    
    @staticmethod
    def create_general_advice_prompt(
        user_data: dict,
        profile: dict,
        question: str,
        conversation_history: Optional[List[dict]] = None
    ) -> str:
        """
        Crée un prompt pour des conseils généraux.
        
        Args:
            user_data: Données utilisateur
            profile: Profil physiologique
            question: Question de l'utilisateur
            conversation_history: Historique de conversation
            
        Returns:
            Prompt complet formaté
        """
        
        context = PromptTemplateManager.format_user_context(user_data, profile)
        
        # Historique
        history_text = ""
        if conversation_history and len(conversation_history) > 0:
            history_text = "\n\n💬 HISTORIQUE DE CONVERSATION:\n"
            for i, item in enumerate(conversation_history[-3:], 1):
                history_text += f"\n{i}. User: {item['user']}\n   Assistant: {item['assistant'][:100]}...\n"
        
        prompt = f"""<|system|>
{PromptTemplateManager.SYSTEM_MESSAGE}

Tu réponds de manière conversationnelle tout en restant professionnel.
Adapte tes conseils au contexte de la conversation.<|end|>
<|user|>
{context}
{history_text}

❓ QUESTION:
{question}<|end|>
<|assistant|>
"""
        
        return prompt
    
    @staticmethod
    def create_motivation_prompt(
        user_data: dict,
        profile: dict,
        context_type: str = "general"
    ) -> str:
        """
        Crée un prompt pour générer de la motivation.
        
        Args:
            user_data: Données utilisateur
            profile: Profil physiologique
            context_type: Type de contexte (general, plateau, setback)
            
        Returns:
            Prompt complet formaté
        """
        
        context = PromptTemplateManager.format_user_context(user_data, profile)
        
        context_messages = {
            "general": "Donne-moi un message motivant pour continuer mes efforts.",
            "plateau": "Je stagne dans mes progrès, comment rester motivé?",
            "setback": "J'ai manqué plusieurs séances, comment me remotiver?"
        }
        
        user_request = context_messages.get(context_type, context_messages["general"])
        
        prompt = f"""<|system|>
{PromptTemplateManager.SYSTEM_MESSAGE}

En plus de tes compétences techniques, tu es un excellent motivateur.
Fournis un message inspirant et encourageant, adapté à la situation de l'utilisateur.<|end|>
<|user|>
{context}

{user_request}

INCLUS:
💪 Message motivant personnalisé
🎯 Rappel des objectifs
📊 Progrès déjà accomplis
🚀 Prochaines étapes concrètes<|end|>
<|assistant|>
"""
        
        return prompt
    
    @staticmethod
    def create_exercise_form_prompt(
        user_data: dict,
        profile: dict,
        exercise_name: str
    ) -> str:
        """
        Crée un prompt pour expliquer la forme d'un exercice.
        
        Args:
            user_data: Données utilisateur
            profile: Profil physiologique
            exercise_name: Nom de l'exercice
            
        Returns:
            Prompt complet formaté
        """
        
        # Contexte simplifié pour ce type de requête
        basic_context = f"""👤 Utilisateur: {user_data['age']} ans, {user_data['gender']}, Niveau: {user_data.get('experience_level', 'intermédiaire')}"""
        
        prompt = f"""<|system|>
{PromptTemplateManager.SYSTEM_MESSAGE}

Tu es spécialisé dans l'enseignement de la technique d'exercices.
Explique clairement et de manière sécuritaire.<|end|>
<|user|>
{basic_context}

Explique-moi comment réaliser correctement l'exercice: {exercise_name}

STRUCTURE ATTENDUE:
🏋️ {exercise_name.upper()}

1. 📝 Description de l'exercice
2. 🎯 Muscles ciblés
3. 📋 Étapes détaillées d'exécution
4. ✅ Points clés à respecter
5. ❌ Erreurs communes à éviter
6. 💡 Variations selon le niveau
7. ⚠️ Précautions de sécurité<|end|>
<|assistant|>
"""
        
        return prompt
    
    @staticmethod
    def create_progress_tracking_prompt(
        user_data: dict,
        profile: dict,
        progress_data: dict
    ) -> str:
        """
        Crée un prompt pour analyser les progrès.
        
        Args:
            user_data: Données utilisateur
            profile: Profil physiologique
            progress_data: Données de progression (poids, performances, etc.)
            
        Returns:
            Prompt complet formaté
        """
        
        context = PromptTemplateManager.format_user_context(user_data, profile)
        
        # Formater les données de progression
        progress_text = "📈 DONNÉES DE PROGRESSION:\n"
        if 'weight_history' in progress_data:
            progress_text += f"   Poids: {progress_data['weight_history']}\n"
        if 'performance_metrics' in progress_data:
            progress_text += f"   Performances: {progress_data['performance_metrics']}\n"
        if 'adherence_rate' in progress_data:
            progress_text += f"   Taux de suivi: {progress_data['adherence_rate']}%\n"
        
        prompt = f"""<|system|>
{PromptTemplateManager.SYSTEM_MESSAGE}

Tu analyses les données de progression de manière objective et constructive.<|end|>
<|user|>
{context}

{progress_text}

Analyse mes progrès et donne-moi un retour constructif.

INCLUS:
📊 Analyse des progrès
✅ Points positifs
⚠️ Points à améliorer
🎯 Recommandations d'ajustement
🚀 Objectifs pour les prochaines semaines<|end|>
<|assistant|>
"""
        
        return prompt
    
    @staticmethod
    def get_template_by_type(
        prompt_type: PromptType,
        user_data: dict,
        profile: dict,
        **kwargs
    ) -> str:
        """
        Récupère un template de prompt selon le type.
        
        Args:
            prompt_type: Type de prompt désiré
            user_data: Données utilisateur
            profile: Profil physiologique
            **kwargs: Arguments supplémentaires selon le type
            
        Returns:
            Prompt formaté
        """
        
        if prompt_type == PromptType.WORKOUT_PLAN:
            return PromptTemplateManager.create_workout_plan_prompt(
                user_data, profile, **kwargs
            )
        
        elif prompt_type == PromptType.NUTRITION_PLAN:
            return PromptTemplateManager.create_nutrition_plan_prompt(
                user_data, profile, **kwargs
            )
        
        elif prompt_type == PromptType.GENERAL_ADVICE:
            return PromptTemplateManager.create_general_advice_prompt(
                user_data, profile, **kwargs
            )
        
        elif prompt_type == PromptType.MOTIVATION:
            return PromptTemplateManager.create_motivation_prompt(
                user_data, profile, **kwargs
            )
        
        elif prompt_type == PromptType.EXERCISE_FORM:
            return PromptTemplateManager.create_exercise_form_prompt(
                user_data, profile, **kwargs
            )
        
        elif prompt_type == PromptType.PROGRESS_TRACKING:
            return PromptTemplateManager.create_progress_tracking_prompt(
                user_data, profile, **kwargs
            )
        
        else:
            raise ValueError(f"Type de prompt non supporté: {prompt_type}")


# ============================================================================
# EXEMPLES D'UTILISATION
# ============================================================================

def demonstrate_templates():
    """Démontre l'utilisation des différents templates"""
    
    # Données de test
    user_data = {
        'age': 25,
        'gender': 'male',
        'weight': 75,
        'height': 1.75,
        'activity_level': 'moderately_active',
        'goal': 'muscle_gain',
        'experience_level': 2
    }
    
    # Profil simulé
    profile = {
        'bmi': {'bmi': 24.5, 'category': 'Normal', 'indicator': '🟢'},
        'bmr': {'value': 1669},
        'tdee': {'value': 2587},
        'nutrition': {
            'target_calories': 2887,
            'macros': {
                'protein_g': 216,
                'carbs_g': 325,
                'fat_g': 80,
                'protein_percent': 30,
                'carbs_percent': 45,
                'fat_percent': 25
            }
        },
        'weight_analysis': {
            'current': 75,
            'ideal': 67.4,
            'difference': 7.6,
            'status': 'au dessus'
        }
    }
    
    print("="*60)
    print("📝 DÉMONSTRATION DES TEMPLATES DE PROMPTS")
    print("="*60)
    
    # 1. Programme d'entraînement
    print("\n1️⃣ PROMPT: Programme d'entraînement")
    print("-"*60)
    workout_prompt = PromptTemplateManager.create_workout_plan_prompt(
        user_data, profile, workout_type="musculation", duration_weeks=2
    )
    print(workout_prompt[:500] + "...\n")
    
    # 2. Plan nutritionnel
    print("\n2️⃣ PROMPT: Plan nutritionnel")
    print("-"*60)
    nutrition_prompt = PromptTemplateManager.create_nutrition_plan_prompt(
        user_data, profile, meal_count=4
    )
    print(nutrition_prompt[:500] + "...\n")
    
    # 3. Conseils généraux
    print("\n3️⃣ PROMPT: Conseils généraux")
    print("-"*60)
    advice_prompt = PromptTemplateManager.create_general_advice_prompt(
        user_data, profile, question="Comment améliorer ma récupération musculaire?"
    )
    print(advice_prompt[:500] + "...\n")
    
    print("="*60)


if __name__ == "__main__":
    demonstrate_templates()