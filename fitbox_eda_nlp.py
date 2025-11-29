
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.corpus import stopwords
import re
from collections import Counter


try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
    nltk.download('wordnet')
    nltk.download('stopwords')
    nltk.download('averaged_perceptron_tagger')


plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# ============================================================================
# PARTIE 1: ANALYSE EXPLORATOIRE DES DONNÉES (EDA)
# ============================================================================

def load_data(filepath='fitness_data_cleaned.csv'):
    """Charge les données nettoyées"""
    df = pd.read_csv(filepath)
    print(f"✓ Données chargées: {df.shape[0]} lignes, {df.shape[1]} colonnes\n")
    return df

def visualize_distributions(df):
    """Visualise la distribution des variables numériques"""
    print("="*60)
    print("1. DISTRIBUTION DES VARIABLES NUMÉRIQUES")
    print("="*60)
    
    numeric_cols = ['Age', 'Weight (kg)', 'Height (m)', 'BMI', 
                    'Session_Duration (hours)', 'Calories_Burned',
                    'Fat_Percentage', 'Water_Intake (liters)',
                    'Workout_Frequency (days/week)']
    
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    axes = axes.ravel()
    
    for idx, col in enumerate(numeric_cols):
        # Histogramme avec courbe de densité
        axes[idx].hist(df[col], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        axes[idx].axvline(df[col].mean(), color='red', linestyle='--', linewidth=2, label=f'Moyenne: {df[col].mean():.2f}')
        axes[idx].axvline(df[col].median(), color='green', linestyle='--', linewidth=2, label=f'Médiane: {df[col].median():.2f}')
        axes[idx].set_title(f'Distribution: {col}', fontsize=10, fontweight='bold')
        axes[idx].set_xlabel(col)
        axes[idx].set_ylabel('Fréquence')
        axes[idx].legend(fontsize=8)
        axes[idx].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('distributions_numeriques.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Statistiques descriptives
    print("\n📊 Statistiques descriptives:")
    print(df[numeric_cols].describe().round(2))
    print("\n")

def visualize_categorical(df):
    """Visualise les variables catégorielles"""
    print("="*60)
    print("2. DISTRIBUTION DES VARIABLES CATÉGORIELLES")
    print("="*60)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Gender
    gender_counts = df['Gender'].value_counts()
    axes[0].bar(gender_counts.index, gender_counts.values, color=['#FF6B6B', '#4ECDC4'])
    axes[0].set_title('Distribution par Genre', fontweight='bold')
    axes[0].set_ylabel('Nombre')
    for i, v in enumerate(gender_counts.values):
        axes[0].text(i, v + 5, str(v), ha='center', fontweight='bold')
    
    # Workout Type
    workout_counts = df['Workout_Type'].value_counts()
    axes[1].barh(workout_counts.index, workout_counts.values, color=sns.color_palette("viridis", len(workout_counts)))
    axes[1].set_title('Types d\'Entraînement', fontweight='bold')
    axes[1].set_xlabel('Nombre')
    for i, v in enumerate(workout_counts.values):
        axes[1].text(v + 5, i, str(v), va='center', fontweight='bold')
    
    # Experience Level
    exp_counts = df['Experience_Level'].value_counts().sort_index()
    exp_labels = {1: 'Débutant', 2: 'Intermédiaire', 3: 'Avancé'}
    exp_names = [exp_labels.get(k, k) for k in exp_counts.index]
    colors_exp = ['#95E1D3', '#F38181', '#AA96DA']
    axes[2].pie(exp_counts.values, labels=exp_names, autopct='%1.1f%%', 
                colors=colors_exp, startangle=90)
    axes[2].set_title('Niveau d\'Expérience', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('distributions_categorielles.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\n📊 Genre: {dict(gender_counts)}")
    print(f"📊 Types d'entraînement: {dict(workout_counts)}")
    print(f"📊 Niveaux d'expérience: {dict(exp_counts)}\n")

def analyze_correlations(df):
    """Analyse les corrélations entre variables"""
    print("="*60)
    print("3. MATRICE DE CORRÉLATIONS")
    print("="*60)
    
    # Sélectionner les colonnes numériques
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    # Calculer la matrice de corrélation
    corr_matrix = df[numeric_cols].corr()
    
    # Visualisation
    plt.figure(figsize=(12, 10))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', 
                cmap='coolwarm', center=0, square=True, 
                linewidths=1, cbar_kws={"shrink": 0.8})
    plt.title('Matrice de Corrélation des Variables Numériques', 
              fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig('correlation_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Trouver les corrélations fortes (>0.5 ou <-0.5)
    print("\n🔍 Corrélations fortes détectées:")
    strong_corr = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            if abs(corr_matrix.iloc[i, j]) > 0.5:
                strong_corr.append({
                    'Variable 1': corr_matrix.columns[i],
                    'Variable 2': corr_matrix.columns[j],
                    'Corrélation': corr_matrix.iloc[i, j]
                })
    
    if strong_corr:
        df_corr = pd.DataFrame(strong_corr).sort_values('Corrélation', key=abs, ascending=False)
        print(df_corr.to_string(index=False))
    else:
        print("Aucune corrélation forte détectée (|r| > 0.5)")
    print("\n")

def analyze_by_groups(df):
    """Analyse comparative par groupes"""
    print("="*60)
    print("4. ANALYSE PAR GROUPES")
    print("="*60)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # BMI par Genre
    df.boxplot(column='BMI', by='Gender', ax=axes[0, 0])
    axes[0, 0].set_title('BMI par Genre')
    axes[0, 0].set_xlabel('Genre')
    axes[0, 0].set_ylabel('BMI')
    plt.sca(axes[0, 0])
    plt.xticks(rotation=0)
    
    # Calories brûlées par Type d'entraînement
    df.boxplot(column='Calories_Burned', by='Workout_Type', ax=axes[0, 1])
    axes[0, 1].set_title('Calories Brûlées par Type d\'Entraînement')
    axes[0, 1].set_xlabel('Type')
    axes[0, 1].set_ylabel('Calories')
    plt.sca(axes[0, 1])
    plt.xticks(rotation=45, ha='right')
    
    # Durée de session par Niveau d'expérience
    df.boxplot(column='Session_Duration (hours)', by='Experience_Level', ax=axes[1, 0])
    axes[1, 0].set_title('Durée de Session par Niveau')
    axes[1, 0].set_xlabel('Niveau d\'Expérience')
    axes[1, 0].set_ylabel('Durée (heures)')
    
    # Fréquence d'entraînement par Niveau
    df.boxplot(column='Workout_Frequency (days/week)', by='Experience_Level', ax=axes[1, 1])
    axes[1, 1].set_title('Fréquence par Niveau')
    axes[1, 1].set_xlabel('Niveau d\'Expérience')
    axes[1, 1].set_ylabel('Jours/semaine')
    
    plt.tight_layout()
    plt.savefig('analyse_par_groupes.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Statistiques par genre
    print("\n📊 Moyennes par Genre:")
    print(df.groupby('Gender')[['BMI', 'Weight (kg)', 'Calories_Burned']].mean().round(2))
    
    print("\n📊 Moyennes par Type d'Entraînement:")
    print(df.groupby('Workout_Type')[['Calories_Burned', 'Session_Duration (hours)']].mean().round(2))
    
    print("\n📊 Moyennes par Niveau d'Expérience:")
    print(df.groupby('Experience_Level')[['Workout_Frequency (days/week)', 'Session_Duration (hours)']].mean().round(2))
    print("\n")

def check_class_balance(df):
    """Vérifie l'équilibre des classes"""
    print("="*60)
    print("5. ÉQUILIBRE DES CLASSES")
    print("="*60)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Workout Type
    workout_pct = df['Workout_Type'].value_counts(normalize=True) * 100
    axes[0].bar(range(len(workout_pct)), workout_pct.values, color=sns.color_palette("Set2"))
    axes[0].set_xticks(range(len(workout_pct)))
    axes[0].set_xticklabels(workout_pct.index, rotation=45, ha='right')
    axes[0].set_ylabel('Pourcentage (%)')
    axes[0].set_title('Distribution des Types d\'Entraînement', fontweight='bold')
    axes[0].axhline(y=100/len(workout_pct), color='r', linestyle='--', label='Équilibre parfait')
    axes[0].legend()
    axes[0].grid(axis='y', alpha=0.3)
    
    for i, v in enumerate(workout_pct.values):
        axes[0].text(i, v + 1, f'{v:.1f}%', ha='center', fontweight='bold')
    
    # Experience Level
    exp_pct = df['Experience_Level'].value_counts(normalize=True).sort_index() * 100
    exp_labels = {1: 'Débutant', 2: 'Intermédiaire', 3: 'Avancé'}
    exp_names = [exp_labels[k] for k in exp_pct.index]
    axes[1].bar(range(len(exp_pct)), exp_pct.values, color=['#95E1D3', '#F38181', '#AA96DA'])
    axes[1].set_xticks(range(len(exp_pct)))
    axes[1].set_xticklabels(exp_names)
    axes[1].set_ylabel('Pourcentage (%)')
    axes[1].set_title('Distribution des Niveaux d\'Expérience', fontweight='bold')
    axes[1].axhline(y=33.33, color='r', linestyle='--', label='Équilibre parfait')
    axes[1].legend()
    axes[1].grid(axis='y', alpha=0.3)
    
    for i, v in enumerate(exp_pct.values):
        axes[1].text(i, v + 1, f'{v:.1f}%', ha='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('equilibre_classes.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\n📊 Distribution des Types d'Entraînement:")
    print(workout_pct.round(2))
    
    print("\n📊 Distribution des Niveaux d'Expérience:")
    print(exp_pct.round(2))
    
    # Calculer le déséquilibre
    workout_imbalance = workout_pct.max() / workout_pct.min()
    exp_imbalance = exp_pct.max() / exp_pct.min()
    
    print(f"\n⚠️  Ratio de déséquilibre Workout_Type: {workout_imbalance:.2f}:1")
    print(f"⚠️  Ratio de déséquilibre Experience_Level: {exp_imbalance:.2f}:1")
    
    if workout_imbalance > 3:
        print("⚠️  ATTENTION: Déséquilibre important dans Workout_Type - considérer le rééchantillonnage")
    if exp_imbalance > 3:
        print("⚠️  ATTENTION: Déséquilibre important dans Experience_Level - considérer le rééchantillonnage")
    print("\n")

# ============================================================================
# PARTIE 2: PRÉTRAITEMENT NLP
# ============================================================================

class FitnessNLPPreprocessor:
    """Classe pour le prétraitement NLP des données fitness"""
    
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stemmer = PorterStemmer()
        self.stop_words = set(stopwords.words('english'))
        self.vocabulary = {}
        self.workout_descriptions = self._create_workout_descriptions()
        self.nutrition_descriptions = self._create_nutrition_descriptions()
        
    def _create_workout_descriptions(self):
        """Crée des descriptions d'exercices par type"""
        return {
            'Cardio': [
                'running on treadmill for endurance training and calorie burning',
                'cycling for cardiovascular health and leg strength',
                'swimming laps for full body cardio workout',
                'rowing machine for upper body and core engagement',
                'high intensity interval training with jumping exercises'
            ],
            'Strength': [
                'weightlifting exercises including squats deadlifts and bench press',
                'resistance training with dumbbells and barbells',
                'compound movements for muscle building and strength gains',
                'progressive overload training for hypertrophy',
                'powerlifting focused on main lifts'
            ],
            'Yoga': [
                'hatha yoga for flexibility and balance improvement',
                'vinyasa flow sequences for mindful movement',
                'restorative yoga poses for recovery and relaxation',
                'power yoga for strength and flexibility',
                'meditation and breathing exercises for mental clarity'
            ],
            'HIIT': [
                'high intensity burpees and mountain climbers',
                'circuit training with minimal rest periods',
                'explosive plyometric exercises for power',
                'tabata protocol with maximum effort intervals',
                'metabolic conditioning for fat loss'
            ]
        }
    
    def _create_nutrition_descriptions(self):
        """Crée des descriptions nutritionnelles par objectif"""
        return {
            'Weight Loss': [
                'caloric deficit meal plan with high protein intake',
                'lean proteins vegetables and complex carbohydrates',
                'reduced sugar and processed foods for fat loss',
                'intermittent fasting compatible nutrition',
                'portion controlled meals with nutrient density'
            ],
            'Muscle Gain': [
                'caloric surplus with increased protein consumption',
                'post workout nutrition with protein and carbs',
                'frequent meals for muscle building and recovery',
                'amino acid rich foods for muscle protein synthesis',
                'healthy fats and complex carbs for energy'
            ],
            'Maintenance': [
                'balanced macronutrients for weight maintenance',
                'whole foods diet with variety of nutrients',
                'moderate protein carbs and healthy fats',
                'sustainable eating habits for long term health',
                'intuitive eating with mindful portions'
            ]
        }
    
    def clean_text(self, text):
        """Nettoie le texte"""
        # Minuscules
        text = text.lower()
        # Supprimer la ponctuation et caractères spéciaux
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
        # Supprimer les espaces multiples
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def tokenize(self, text):
        """Tokenise le texte"""
        text = self.clean_text(text)
        tokens = word_tokenize(text)
        return tokens
    
    def remove_stopwords(self, tokens):
        """Supprime les stop words"""
        return [token for token in tokens if token not in self.stop_words]
    
    def lemmatize(self, tokens):
        """Lemmatisation"""
        return [self.lemmatizer.lemmatize(token) for token in tokens]
    
    def stem(self, tokens):
        """Stemming"""
        return [self.stemmer.stem(token) for token in tokens]
    
    def preprocess_pipeline(self, text, use_lemma=True):
        """Pipeline complet de prétraitement"""
        tokens = self.tokenize(text)
        tokens = self.remove_stopwords(tokens)
        if use_lemma:
            tokens = self.lemmatize(tokens)
        else:
            tokens = self.stem(tokens)
        return tokens
    
    def build_vocabulary(self):
        """Construit le vocabulaire à partir des descriptions"""
        print("="*60)
        print("6. CONSTRUCTION DU VOCABULAIRE NLP")
        print("="*60)
        
        all_texts = []
        
        # Collecter tous les textes
        for workout_type, descriptions in self.workout_descriptions.items():
            all_texts.extend(descriptions)
        
        for goal, descriptions in self.nutrition_descriptions.items():
            all_texts.extend(descriptions)
        
        # Prétraiter et compter
        all_tokens = []
        for text in all_texts:
            tokens = self.preprocess_pipeline(text, use_lemma=True)
            all_tokens.extend(tokens)
        
        # Créer le vocabulaire avec fréquences
        word_freq = Counter(all_tokens)
        self.vocabulary = dict(word_freq.most_common())
        
        print(f"\n✓ Vocabulaire construit: {len(self.vocabulary)} mots uniques")
        print(f"\n📊 Top 20 mots les plus fréquents:")
        for word, freq in list(self.vocabulary.items())[:20]:
            print(f"  {word:20s} : {freq:3d} occurrences")
        
        return self.vocabulary
    
    def demonstrate_preprocessing(self):
        """Démontre le preprocessing sur des exemples"""
        print("\n" + "="*60)
        print("7. DÉMONSTRATION DU PRÉTRAITEMENT NLP")
        print("="*60)
        
        # Exemple 1: Workout
        example1 = "Running on treadmill for 30 minutes, high-intensity cardio training!"
        print(f"\n📝 Exemple 1 (Workout):")
        print(f"Original: {example1}")
        print(f"Nettoyé: {self.clean_text(example1)}")
        print(f"Tokens: {self.tokenize(example1)}")
        print(f"Sans stopwords: {self.remove_stopwords(self.tokenize(example1))}")
        print(f"Lemmatisé: {self.preprocess_pipeline(example1, use_lemma=True)}")
        print(f"Stemmé: {self.preprocess_pipeline(example1, use_lemma=False)}")
        
        # Exemple 2: Nutrition
        example2 = "Healthy meal plan with lean proteins, vegetables and complex carbohydrates"
        print(f"\n📝 Exemple 2 (Nutrition):")
        print(f"Original: {example2}")
        print(f"Prétraité: {self.preprocess_pipeline(example2)}")
        
        print("\n")
    
    def create_training_dataset(self, df):
        """Crée un dataset d'entraînement avec descriptions"""
        print("="*60)
        print("8. CRÉATION DU DATASET D'ENTRAÎNEMENT NLP")
        print("="*60)
        
        training_data = []
        
        for idx, row in df.iterrows():
            workout_type = row['Workout_Type']
            
            # Sélectionner une description aléatoire
            if workout_type in self.workout_descriptions:
                description = np.random.choice(self.workout_descriptions[workout_type])
                
                # Créer l'entrée d'entraînement
                entry = {
                    'user_profile': f"Age: {row['Age']}, Gender: {row['Gender']}, "
                                   f"Weight: {row['Weight (kg)']}kg, Height: {row['Height (m)']}m, "
                                   f"BMI: {row['BMI']:.1f}, Experience: Level {row['Experience_Level']}",
                    'workout_description': description,
                    'workout_type': workout_type,
                    'tokens': self.preprocess_pipeline(description)
                }
                training_data.append(entry)
        
        df_training = pd.DataFrame(training_data)
        
        print(f"\n✓ Dataset d'entraînement créé: {len(df_training)} exemples")
        print(f"\n📊 Aperçu du dataset:")
        print(df_training[['workout_type', 'workout_description']].head())
        
        # Sauvegarder
        df_training.to_csv('training_dataset_nlp.csv', index=False)
        print(f"\n✓ Dataset sauvegardé: training_dataset_nlp.csv")
        
        return df_training

# ============================================================================
# FONCTION PRINCIPALE
# ============================================================================

def main():
    """Fonction principale pour exécuter toute l'analyse"""
    
    print("\n" + "="*60)
    print("🏋️  FITBOX - ANALYSE EXPLORATOIRE & PRÉTRAITEMENT NLP")
    print("="*60 + "\n")
    
    # Charger les données
    df = load_data('fitness_data_cleaned.csv')
    
    # PARTIE 1: EDA
    print("\n📊 PARTIE 1: ANALYSE EXPLORATOIRE DES DONNÉES\n")
    
    visualize_distributions(df)
    visualize_categorical(df)
    analyze_correlations(df)
    analyze_by_groups(df)
    check_class_balance(df)
    
    # PARTIE 2: NLP
    print("\n📝 PARTIE 2: PRÉTRAITEMENT NLP\n")
    
    nlp_processor = FitnessNLPPreprocessor()
    vocabulary = nlp_processor.build_vocabulary()
    nlp_processor.demonstrate_preprocessing()
    training_df = nlp_processor.create_training_dataset(df)
    
    # Sauvegarder le vocabulaire
    vocab_df = pd.DataFrame(list(vocabulary.items()), columns=['word', 'frequency'])
    vocab_df.to_csv('vocabulary.csv', index=False)
    print(f"✓ Vocabulaire sauvegardé: vocabulary.csv")
    
    print("\n" + "="*60)
    print("✅ ANALYSE TERMINÉE AVEC SUCCÈS!")
    print("="*60)
    print("\nFichiers générés:")
    print("  1. distributions_numeriques.png")
    print("  2. distributions_categorielles.png")
    print("  3. correlation_matrix.png")
    print("  4. analyse_par_groupes.png")
    print("  5. equilibre_classes.png")
    print("  6. training_dataset_nlp.csv")
    print("  7. vocabulary.csv")
    print("\n")

if __name__ == "__main__":
    main()