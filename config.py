# Configuration settings for Metamind system

# LLM API settings
import os 

# set to True to use OpenAI API (remote), False to use Ollama (local)
USE_REMOTE = True

if USE_REMOTE:
    
    LLM_CONFIG = {
        "api_key": os.getenv("OPENAI_API_KEY"),  # stored as an environement variable
        "base_url": "https://api.openai.com/v1",
        "model_name": "gpt-3.5-turbo",  
        "temperature": 0.1,
        "max_tokens": 1000
    }
else:
   
    LLM_CONFIG = {
        "api_key": "ollama",  # dummy key for local usage
        "base_url": "http://localhost:11434/v1",
        "model_name": "llama3:8b",
        "temperature": 0.1,
        "max_tokens": 1000
    }


# ToM Agent settings
TOM_AGENT_CONFIG = {
    "hypothesis_count_k": 7, 
    "target_diversity": 0.4,  
    "evidence_threshold": "medium-high"  
}

# Domain Agent settings
DOMAIN_AGENT_CONFIG = {
    "lambda": 0.7, 
    "epsilon": 1e-10  # Small constant to avoid log(0)
}

# Response Agent settings
RESPONSE_AGENT_CONFIG = {
    "beta": 0.8,  # Trade-off weight for empathy vs coherence
    "utility_threshold": 0.9,  # Threshold for acceptable utility score
    "max_revisions": 3  # Maximum number of response revisions
}

# Social Memory settings
SOCIAL_MEMORY_CONFIG = {
    "memory_decay_rate": 0.05,  # Rate at which memory importance decays over time
    "max_memory_items": 100  # Maximum number of items to store in memory
}

# Mental state types
MENTAL_STATE_TYPES = ["Belief", "Desire", "Intention", "Emotion", "Thought"]

# Sentiment analysis cues and weights (for heuristic fallback)
SENTIMENT_CUES = {
    "positive": [
        "love", "like", "great", "good", "amazing", "fantastic", "happy", "satisfied",
        "enjoy", "fast", "reliable", "recommend", "improve", "useful"
    ],
    "negative": [
        "hate", "dislike", "bad", "terrible", "awful", "angry", "frustrated",
        "annoy", "issue", "problem", "bug", "broken", "slow", "expensive",
        "cancel", "return", "switch"
    ],
    "emotion": {
        "positive": ["happy", "pleased", "excited", "satisfied", "delighted"],
        "negative": ["angry", "upset", "sad", "annoyed", "frustrated", "disappointed"]
    },
    "intention": {
        "positive": ["buy", "upgrade", "renew", "recommend", "subscribe"],
        "negative": ["cancel", "return", "quit", "switch", "churn"]
    },
    "belief": {
        "positive": ["is great", "works well", "fast", "reliable", "intuitive"],
        "negative": ["is broken", "buggy", "slow", "confusing", "hard to use"]
    }
}

SENTIMENT_WEIGHTS = {
    # Universal lexical cues
    "base_positive_cue": 0.5,
    "base_negative_cue": 0.5,

    # Type-specific cue weights
    "emotion_positive": 0.4,
    "emotion_negative": 0.4,
    "intention_positive": 0.3,
    "intention_negative": 0.3,
    "belief_positive": 0.3,
    "belief_negative": 0.3,

    # Type priors
    "desire_neg_prior": 0.2
}
