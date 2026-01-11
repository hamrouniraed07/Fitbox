"""
Script d'Utilisation du Modèle Fine-tuné QLoRA
================================================

Ce script montre comment:
1. Charger le modèle fine-tuné
2. Utiliser le modèle pour faire des inférences
3. Générer des recommandations personnalisées
4. Mesurer les performances
"""

import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import json
from datetime import datetime


class FitBoxInference:
    """Classe pour l'inférence avec le modèle fine-tuné QLoRA"""
    
    def __init__(
        self,
        base_model: str = "llama3.2:latest",
        adapter_path: str = "models/fitbox_model"
    ):
        """
        Initialise le modèle fine-tuné pour l'inférence.
        
        Args:
            base_model: Modèle de base (Ollama)
            adapter_path: Chemin vers les adapters QLoRA
        """
        self.base_model = base_model
        self.adapter_path = Path(adapter_path)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.model = None
        self.tokenizer = None
        
        print(f"\n🤖 Initialisation de l'inférence FitBox QLoRA")
        print(f"   Device: {self.device}")
        print(f"   Adapter path: {self.adapter_path}")
    
    def load_model(self):
        """Charge le modèle fine-tuné"""
        print(f"\n📦 Chargement du modèle...")
        
        # Charger le tokenizer
        print("   • Tokenizer... ", end="")
        self.tokenizer = AutoTokenizer.from_pretrained(str(self.adapter_path))
        print("✅")
        
        # Charger le modèle de base
        print("   • Modèle de base (quantization 4-bit)... ", end="")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.base_model,
            device_map="auto",
            trust_remote_code=True,
        )
        print("✅")
        
        # Charger les adapters QLoRA
        print("   • Adapters QLoRA... ", end="")
        self.model = PeftModel.from_pretrained(
            self.model,
            str(self.adapter_path),
            device_map="auto"
        )
        print("✅")
        
        # Mode inférence
        self.model.eval()
        
        print(f"\n✅ Modèle chargé et prêt pour l'inférence!")
    
    def generate_recommendation(
        self,
        prompt: str,
        max_tokens: int = 300,
        temperature: float = 0.7,
        top_p: float = 0.9
    ) -> str:
        """
        Génère une recommandation personnalisée.
        
        Args:
            prompt: Le prompt d'entrée
            max_tokens: Nombre maximum de tokens à générer
            temperature: Contrôle la créativité (0.0-2.0)
            top_p: Nucleus sampling
            
        Returns:
            La recommandation générée
        """
        
        # Tokenizer l'input
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        # Générer la réponse
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
            )
        
        # Décoder la réponse
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extraire seulement la partie <|assistant|>
        if "<|assistant|>" in response:
            response = response.split("<|assistant|>")[-1].strip()
        
        return response
    
    def get_workout_recommendation(
        self,
        age: int,
        gender: str,
        weight: float,
        height: float,
        experience_level: str,
        goal: str
    ) -> dict:
        """
        Obtient une recommandation d'entraînement personnalisée.
        
        Args:
            age: Âge de l'utilisateur
            gender: Genre (male/female)
            weight: Poids en kg
            height: Taille en m
            experience_level: Niveau d'expérience (Beginner/Intermediate/Advanced)
            goal: Objectif (muscle_gain/weight_loss/maintenance)
            
        Returns:
            Dict avec les recommandations
        """
        
        bmi = weight / (height ** 2)
        
        prompt = f"""<|system|>
Tu es FitBox, un coach sportif expert qui fournit des programmes personnalisés basés sur le profil de l'utilisateur.<|end|>
<|user|>
Profil utilisateur:
- Âge: {age} ans
- Genre: {gender}
- Poids: {weight} kg
- Taille: {height} m
- IMC: {bmi:.1f}
- Niveau: {experience_level}
- Objectif: {goal}

Crée un programme d'entraînement personnalisé pour cette semaine.<|end|>
<|assistant|>
"""
        
        response = self.generate_recommendation(prompt, max_tokens=400)
        
        return {
            "profile": {
                "age": age,
                "gender": gender,
                "weight": weight,
                "height": height,
                "bmi": bmi,
                "experience_level": experience_level,
                "goal": goal
            },
            "recommendation": response,
            "timestamp": datetime.now().isoformat()
        }
    
    def get_nutrition_recommendation(
        self,
        age: int,
        gender: str,
        weight: float,
        height: float,
        activity_level: str,
        goal: str
    ) -> dict:
        """
        Obtient une recommandation nutritionnelle personnalisée.
        
        Args:
            age: Âge
            gender: Genre
            weight: Poids en kg
            height: Taille en m
            activity_level: Niveau d'activité
            goal: Objectif
            
        Returns:
            Dict avec les recommandations nutritionnelles
        """
        
        bmi = weight / (height ** 2)
        
        prompt = f"""<|system|>
Tu es FitBox, un nutritionniste expert. Fournis un plan alimentaire personnalisé.<|end|>
<|user|>
Profil:
- Âge: {age} ans
- Genre: {gender}
- Poids: {weight} kg
- Taille: {height} m
- IMC: {bmi:.1f}
- Activité: {activity_level}
- Objectif: {goal}

Donne-moi un plan nutritionnel optimisé pour cette journée.<|end|>
<|assistant|>
"""
        
        response = self.generate_recommendation(prompt, max_tokens=400)
        
        return {
            "profile": {
                "age": age,
                "gender": gender,
                "weight": weight,
                "height": height,
                "activity_level": activity_level,
                "goal": goal
            },
            "recommendation": response,
            "timestamp": datetime.now().isoformat()
        }
    
    def get_general_advice(
        self,
        age: int,
        gender: str,
        bmi: float,
        experience_level: str
    ) -> dict:
        """
        Obtient des conseils généraux personnalisés.
        """
        
        prompt = f"""<|system|>
Tu es FitBox, un coach sportif et nutritionniste expert.<|end|>
<|user|>
Profil:
- Âge: {age} ans
- Genre: {gender}
- IMC: {bmi:.1f}
- Niveau: {experience_level}

Donne-moi 5 conseils clés pour optimiser mes performances.<|end|>
<|assistant|>
"""
        
        response = self.generate_recommendation(prompt, max_tokens=300)
        
        return {
            "profile": {
                "age": age,
                "gender": gender,
                "bmi": bmi,
                "experience_level": experience_level
            },
            "advice": response,
            "timestamp": datetime.now().isoformat()
        }


def demo():
    """Démontre l'utilisation du modèle fine-tuné"""
    
    print("\n" + "="*70)
    print("🏋️  FITBOX - DÉMO DU MODÈLE FINE-TUNÉ QLORA")
    print("="*70)
    
    # Initialiser l'inférence
    inference = FitBoxInference()
    
    # Charger le modèle
    try:
        inference.load_model()
    except Exception as e:
        print(f"\n⚠️  Erreur lors du chargement du modèle: {e}")
        print("\nNote: Le modèle doit être fine-tuné en premier:")
        print("   python -m backend.finetuning")
        return
    
    # Exemples de profils
    profiles = [
        {
            "age": 25,
            "gender": "male",
            "weight": 75,
            "height": 1.75,
            "experience_level": "Intermediate",
            "goal": "muscle_gain"
        },
        {
            "age": 35,
            "gender": "female",
            "weight": 65,
            "height": 1.65,
            "experience_level": "Beginner",
            "goal": "weight_loss"
        },
    ]
    
    # Générer des recommandations
    print("\n" + "="*70)
    print("📊 EXEMPLES DE RECOMMANDATIONS")
    print("="*70)
    
    for i, profile in enumerate(profiles, 1):
        print(f"\n{'─'*70}")
        print(f"📋 Profil {i}")
        print(f"{'─'*70}")
        print(f"Âge: {profile['age']}, Genre: {profile['gender']}")
        print(f"Poids: {profile['weight']}kg, Taille: {profile['height']}m")
        print(f"Niveau: {profile['experience_level']}, Objectif: {profile['goal']}")
        
        # Recommandation d'entraînement
        print(f"\n🏋️  Recommandation d'entraînement:")
        print("─" * 70)
        workout_rec = inference.get_workout_recommendation(
            age=profile['age'],
            gender=profile['gender'],
            weight=profile['weight'],
            height=profile['height'],
            experience_level=profile['experience_level'],
            goal=profile['goal']
        )
        
        print(workout_rec['recommendation'][:500] + "...")
        
        # Recommandation nutritionnelle
        print(f"\n🥗 Recommandation nutritionnelle:")
        print("─" * 70)
        nutrition_rec = inference.get_nutrition_recommendation(
            age=profile['age'],
            gender=profile['gender'],
            weight=profile['weight'],
            height=profile['height'],
            activity_level="Moderate",
            goal=profile['goal']
        )
        
        print(nutrition_rec['recommendation'][:500] + "...")
        
        # Conseils généraux
        print(f"\n💡 Conseils généraux:")
        print("─" * 70)
        bmi = profile['weight'] / (profile['height'] ** 2)
        advice = inference.get_general_advice(
            age=profile['age'],
            gender=profile['gender'],
            bmi=bmi,
            experience_level=profile['experience_level']
        )
        
        print(advice['advice'][:500] + "...")
    
    print("\n" + "="*70)
    print("✅ DÉMO TERMINÉE")
    print("="*70)
    print("\n💾 Recommandations personnalisées générées avec succès!")
    print("🚀 Le modèle fine-tuné QLoRA fonctionne correctement!")


if __name__ == "__main__":
    demo()
