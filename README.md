# 🏋️ FitBox - AI Fitness Coach

FitBox is an intelligent fitness and nutrition coaching application powered by AI. It provides personalized workout plans, nutrition advice, and conversational coaching using advanced language models. The system combines physiological calculations, data analysis, and machine learning to deliver science-based fitness recommendations.

## 📋 Table of Contents

- [Features](#features)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [API Documentation](#api-documentation)
- [Frontend Interface](#frontend-interface)
- [Testing](#testing)
- [Deployment](#deployment)
- [Configuration](#configuration)
- [Contributing](#contributing)
- [License](#license)
- [Author](#author)

## ✨ Features

### 🤖 AI-Powered Coaching
- **Conversational AI Coach**: Chat with FitBox for personalized fitness advice
- **Intelligent Recommendations**: Context-aware suggestions based on user profile
- **Multiple Languages**: Support for French and English responses

### 📊 Physiological Calculations
- **BMI Calculation**: Body Mass Index with health categorization
- **BMR & TDEE**: Basal Metabolic Rate and Total Daily Energy Expenditure
- **Macronutrient Distribution**: Personalized protein, carbs, and fat ratios
- **Activity Level Assessment**: Tailored recommendations based on lifestyle

### 🏃 Workout Planning
- **Custom Workout Programs**: Personalized training plans
- **Progression Tracking**: Adaptive difficulty based on user level
- **Exercise Guidance**: Form instructions and safety tips
- **Goal-Oriented Training**: Weight loss, muscle gain, or maintenance programs

### 🍽️ Nutrition Planning
- **Meal Planning**: Daily meal suggestions with calorie targets
- **Macronutrient Balance**: Precise protein, carb, and fat distribution
- **Dietary Flexibility**: Support for various dietary preferences
- **Hydration Tracking**: Personalized water intake recommendations

### 🎨 Modern User Interface
- **Streamlit Frontend**: Beautiful, responsive web interface
- **Real-time Chat**: Interactive conversation with the AI coach
- **Profile Management**: Easy user data input and management
- **Export Features**: PDF reports and JSON data export

### 🔧 Technical Features
- **Model Flexibility**: Support for Hugging Face models and Ollama
- **LoRA Fine-tuning**: Efficient model adaptation for fitness domain
- **RESTful API**: Clean backend API for integrations
- **Comprehensive Testing**: Automated test suite for reliability

## 🏗️ Architecture

FitBox follows a modular architecture with clear separation of concerns:

```
FitBox/
├── 🧠 AI Layer (llama3.2 via Ollama)
├── 🔧 Backend API (Flask)
├── 🎨 Frontend UI (Streamlit)
├── 📊 Data Processing (Pandas, NumPy)
└── 🧪 Testing Suite (PyTest)
```

### Core Components

1. **Physiological Calculator**: Computes BMI, BMR, TDEE, and macronutrient needs
2. **Prompt Template Manager**: Creates contextual prompts for AI responses
3. **Model Manager**: Handles LLM loading, fine-tuning, and inference
4. **API Layer**: RESTful endpoints for frontend and external integrations
5. **Frontend Interface**: User-friendly web application

## 📁 Project Structure

```
FitBox/
├── data/                    # Dataset files and raw data
│   ├── Gym_members.csv
│   ├── fitness_data_cleaned.csv
│   └── training_dataset_nlp.csv
├── backend/                 # Backend API and core logic
│   ├── backend_api.py       # Main Flask API server
│   ├── physiological_calculator.py  # Health calculations
│   ├── prompt_templates.py  # AI prompt management
│   ├── model_setup.py       # Model configuration
│   ├── finetuning.py        # LoRA fine-tuning
│   └── __init__.py
├── frontend/                # Streamlit web interface
│   ├── app.py              # Main Streamlit application
│   └── fitbox_rapport_*.pdf # Sample reports
├── models/                  # Trained models and configurations
│   └── fitbox_model/       # Fine-tuned model directory
├── notebooks/              # Jupyter notebooks for analysis
│   ├── fitbox_eda_nlp.ipynb
│   └── Gym.ipynb
├── tests/                  # Test suite
│   ├── test_physiological_calculator.py
│   ├── test_complete_system.py
│   └── __init__.py
├── scripts/                # Utility scripts
│   ├── interactive_calculator.py
│   ├── debug_csv_data.py
│   └── __init__.py
├── outputs/                # Generated charts and results
│   ├── correlation_matrix.png
│   ├── distributions_*.png
│   └── test_results.json
├── deploy/                 # Deployment configurations
│   └── ollama_local_instructions.md
├── requirements.txt        # Python dependencies
├── simple_architecture.py  # Project reorganization script
├── test_results.json       # Test results summary
└── README.md              # This file
```

## 🚀 Installation

### Prerequisites

- Python 3.8+
- pip package manager
- Git
- (Optional) CUDA-compatible GPU for faster model inference

### Step-by-Step Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd FitBox
   ```

2. **Create virtual environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up data (optional)**
   ```bash
   # Place your fitness datasets in the data/ directory
   # The system includes sample data for testing
   ```

5. **Configure model**

   **Default Setup: Use Ollama (recommended for local deployment)**
   ```bash
   # Install Ollama: https://ollama.ai/
   ollama pull llama3.2:latest

   # Set environment variables (optional, these are defaults)
   export OLLAMA_LOCAL=1
   export OLLAMA_MODEL_NAME='llama3.2:latest'
   ```

   **Alternative: Use Hugging Face models**
   ```bash
   # Model will be downloaded automatically on first run
   # Requires ~8GB disk space for Llama-3.2-3B-Instruct
   ```

## 💻 Usage

### Quick Start

1. **Start the Backend API**
   ```bash
   python backend/backend_api.py
   ```
   The API will be available at `http://localhost:5000`

2. **Start the Frontend (in a new terminal)**
   ```bash
   streamlit run frontend/app.py
   ```
   The web interface will open at `http://localhost:8501`

3. **Test the API**
   ```bash
   curl http://localhost:5000/health
   ```

### Development Workflow

1. **Run tests**
   ```bash
   python tests/test_complete_system.py
   ```

2. **Fine-tune the model (optional)**
   ```bash
   python backend/finetuning.py
   ```

3. **Interactive calculator**
   ```bash
   python scripts/interactive_calculator.py
   ```

## 📚 API Documentation

The FitBox API provides RESTful endpoints for all functionality.

### Base URL
```
http://localhost:5000
```

### Endpoints

#### Health Check
```http
GET /health
```
Returns API status and model loading state.

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "timestamp": "2024-01-01T12:00:00"
}
```

#### Profile Calculation
```http
POST /calculate
```

**Request Body:**
```json
{
  "age": 25,
  "gender": "male",
  "weight": 75.0,
  "height": 1.75,
  "activity_level": "moderately_active",
  "goal": "muscle_gain"
}
```

**Response:**
```json
{
  "success": true,
  "profile": {
    "bmi": {"bmi": 24.5, "category": "Normal"},
    "bmr": {"value": 1669},
    "tdee": {"value": 2587},
    "nutrition": {
      "target_calories": 2887,
      "macros": {
        "protein_g": 216,
        "carbs_g": 325,
        "fat_g": 80
      }
    }
  }
}
```

#### Generate Workout Plan
```http
POST /generate_workout
```

#### Generate Nutrition Plan
```http
POST /generate_nutrition
```

#### Chat with AI Coach
```http
POST /chat
```

**Request Body:**
```json
{
  "user_data": {
    "age": 25,
    "gender": "male",
    "weight": 75.0,
    "height": 1.75
  },
  "message": "Donne-moi des conseils pour prendre du muscle",
  "conversation_id": "user_123",
  "history": []
}
```

#### Get Conversation History
```http
GET /conversation/{conversation_id}
```

#### Get Available Options
```http
GET /activity_levels
GET /goals
```

## 🎨 Frontend Interface

The Streamlit frontend provides an intuitive web interface with:

### Features
- **Profile Setup**: Easy input of personal information
- **Real-time Calculations**: Instant physiological metrics
- **Interactive Chat**: Conversational AI coaching
- **Visual Analytics**: Charts for macronutrients and progress
- **Export Options**: PDF reports and data export

### Navigation
1. **Mon Profil**: View and edit your fitness profile
2. **Chat**: Interact with the AI coach
3. **Export**: Download reports and data

### Quick Start Guide
1. Fill in your profile information in the sidebar
2. Review your calculated metrics
3. Start chatting with FitBox for personalized advice
4. Export your profile as PDF or JSON

## 🧪 Testing

FitBox includes comprehensive testing to ensure reliability:

### Run All Tests
```bash
python tests/test_complete_system.py
```

### Test Results
Results are saved to `test_results.json` and include:
- Phase-by-phase status (Data, Calculations, Model, Fine-tuning, API)
- Detailed test outcomes
- Performance metrics

### Test Coverage
- ✅ Data validation and preprocessing
- ✅ Physiological calculations accuracy
- ✅ Model loading and inference
- ✅ API endpoint functionality
- ✅ Integration testing

## 🚀 Deployment

### Local Deployment

1. **Using the provided script**
   ```bash
   python simple_architecture.py
   ```

2. **Manual setup**
   - Ensure all dependencies are installed
   - Configure model paths
   - Set environment variables for Ollama (if used)

### Production Deployment

#### Docker Deployment
```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 5000 8501

CMD ["python", "backend/backend_api.py"]
```

#### Cloud Deployment
- **Backend**: Deploy Flask API to Heroku, Railway, or AWS
- **Frontend**: Use Streamlit Cloud or deploy as static site
- **Models**: Use Hugging Face Spaces or cloud storage

### Environment Variables

```bash
# Ollama Configuration
OLLAMA_LOCAL=1
OLLAMA_API_URL=https://your-ollama-endpoint.com
OLLAMA_API_KEY=your-api-key
OLLAMA_MODEL_NAME=llama3.2:latest

# Model Configuration
MODEL_PATH=models/fitbox_model
USE_GPU=1
```

## ⚙️ Configuration

### Model Configuration

By default, the system uses Ollama with llama3.2:latest locally.

For custom Ollama configuration, set these environment variables:
```bash
export OLLAMA_MODEL_NAME='llama3.2:latest'
export OLLAMA_LOCAL_URL='http://127.0.0.1:11434/api/generate'
```

If using Hugging Face models, create `models/model_config.json`:
```json
{
  "model_name": "meta-llama/Llama-3.2-3B-Instruct",
  "device": "cuda",
  "quantization": "4bit",
  "max_tokens": 512,
  "temperature": 0.7
}
```

### API Configuration

The API supports multiple model backends:
- **Hugging Face Transformers**: Local model inference
- **Ollama**: Local or cloud Ollama instances
- **Custom Endpoints**: Any compatible API

## 🤝 Contributing

We welcome contributions to FitBox!

### Development Setup
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

### Code Standards
- Follow PEP 8 style guidelines
- Add docstrings to all functions
- Include type hints where possible
- Write comprehensive tests

### Areas for Contribution
- Additional language support
- New workout types and exercises
- Enhanced nutrition planning
- Mobile app development
- Performance optimizations

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👨‍💻 Author

**Raed Mohamed Amin Hamrouni**
- **Institution**: École Polytechnique de Sousse
- **Academic Year**: 2025-2026
- **Project**: FitBox - AI Fitness Coach

### Contact
- **Email**: raed.mohamed.amin.hamrouni@polytechnicien.tn
- **LinkedIn**: [LinkedIn Profile]
- **GitHub**: [GitHub Profile]

---

**Made with ❤️ for the fitness community**

*Transforming fitness guidance with the power of AI*
