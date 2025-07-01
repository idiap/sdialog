from flask import Flask, request, jsonify, send_from_directory
import os
import sys
import json
import random
import logging
from typing import Dict, Any

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Add the parent directory to the Python path to import sdialog
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from sdialog.personas import Patient, Doctor, PersonaMetadata
from sdialog.generators import PersonaGenerator
from sdialog.util import config

app = Flask(__name__)

try:
    # Initialize the persona generators with base personas
    logger.info("Initializing persona generators...")
    
    # Load default configuration
    logger.info("Loading configuration...")
    
    patient_generator = PersonaGenerator(persona=Patient)
    doctor_generator = PersonaGenerator(persona=Doctor)
    logger.info("Persona generators initialized successfully")
    
except Exception as e:
    logger.error(f"Error during initialization: {str(e)}")
    raise

# Serve static files
@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

def generate_patient() -> Dict[str, Any]:
    """Generate a random patient persona using PersonaGenerator."""
    try:
        logger.info("Generating new patient persona...")
        patient = patient_generator.generate()
        patient.language = "English"  # Force English language for our interface
        logger.info("Patient persona generated successfully")
        return patient.json()
    except Exception as e:
        logger.error(f"Error generating patient persona: {str(e)}")
        raise

def generate_doctor() -> Dict[str, Any]:
    """Generate a random doctor persona using PersonaGenerator."""
    try:
        logger.info("Generating new doctor persona...")
        doctor = doctor_generator.generate()
        doctor.language = "English"  # Force English language for our interface
        logger.info("Doctor persona generated successfully")
        return doctor.json()
    except Exception as e:
        logger.error(f"Error generating doctor persona: {str(e)}")
        raise

@app.route('/api/generate/persona/<type>', methods=['GET'])
def generate_persona(type):
    """Generate a random persona of the specified type."""
    try:
        logger.info(f"Received request to generate {type} persona")
        if type == 'patient':
            result = generate_patient()
            return jsonify(result)
        elif type == 'doctor':
            result = generate_doctor()
            return jsonify(result)
        else:
            logger.warning(f"Invalid persona type requested: {type}")
            return jsonify({'error': 'Invalid persona type'}), 400
    except Exception as e:
        logger.error(f"Error in generate_persona endpoint: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/save', methods=['POST'])
def save_configuration():
    """Save the complete configuration."""
    try:
        config = request.json
        logger.info("Saving configuration...")
        # Here you would typically save to a database or file
        print("Configuration saved:", json.dumps(config, indent=2))
        return jsonify({'message': 'Configuration saved successfully'})
    except Exception as e:
        logger.error(f"Error saving configuration: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
