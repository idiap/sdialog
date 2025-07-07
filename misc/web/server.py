from flask import Flask, request, jsonify, send_from_directory, send_file
import os
import sys
import json
import logging
from typing import Dict, Any
import io
import soundfile as sf

from sdialog.personas import Patient, Doctor, PersonaAgent
from sdialog.generators import PersonaGenerator
from sdialog.audio import generate_utterance

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Add the parent directory to the Python path to import sdialog
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

app = Flask(__name__)

# Initialize the persona generators with base personas
logger.info("Initializing persona generators...")
logger.info("Persona generators will be initialized per request.")


# Serve static files
@app.route('/')
def index():
    return send_from_directory('.', 'index.html')


def generate_patient(config: Dict[str, Any]) -> Dict[str, Any]:
    """Generate a random patient persona using PersonaGenerator."""
    try:
        logger.info("Generating new patient persona...")
        model = config.get("persona_model", "qwen2.5:3b")
        llm_kwargs = {k: v for k, v in config.items() if k not in ["persona_model", "dialog_model"] and v is not None}
        
        patient_generator = PersonaGenerator(persona=Patient, model=model, llm_kwargs=llm_kwargs)
        patient = patient_generator.generate()
        patient.language = "English"  # Force English language for our interface
        logger.info("Patient persona generated successfully")
        return patient.json()
    except Exception as e:
        logger.error(f"Error generating patient persona: {str(e)}")
        raise


def generate_doctor(config: Dict[str, Any]) -> Dict[str, Any]:
    """Generate a random doctor persona using PersonaGenerator."""
    try:
        logger.info("Generating new doctor persona...")
        model = config.get("persona_model", "qwen2.5:3b")
        llm_kwargs = {k: v for k, v in config.items() if k not in ["persona_model", "dialog_model"] and v is not None}

        doctor_generator = PersonaGenerator(persona=Doctor, model=model, llm_kwargs=llm_kwargs)
        doctor = doctor_generator.generate()
        doctor.language = "English"  # Force English language for our interface
        logger.info("Doctor persona generated successfully")
        return doctor.json()
    except Exception as e:
        logger.error(f"Error generating doctor persona: {str(e)}")
        raise


@app.route('/api/generate/persona/<type>', methods=['POST'])
def generate_persona(type):
    """Generate a random persona of the specified type."""
    try:
        logger.info(f"Received request to generate {type} persona")
        data = request.json
        config = data.get('config', {})

        if type == 'patient':
            result = generate_patient(config)
            return jsonify(result)
        elif type == 'doctor':
            result = generate_doctor(config)
            return jsonify(result)
        else:
            logger.warning(f"Invalid persona type requested: {type}")
            return jsonify({'error': 'Invalid persona type'}), 400
    except Exception as e:
        logger.error(f"Error in generate_persona endpoint: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/generate/dialog', methods=['POST'])
def generate_dialog():
    """Generate a dialogue between a patient and a doctor."""
    try:
        data = request.json
        logger.info("Received request to generate dialogue")

        patient_data = data.get('patient', {})
        doctor_data = data.get('doctor', {})
        config = data.get('config', {})
        
        dialog_model = config.get("dialog_model", "qwen2.s:14b")
        llm_kwargs = {k: v for k, v in config.items() if k not in ["persona_model", "dialog_model"] and v is not None}

        if not patient_data or not doctor_data:
            return jsonify({'error': 'Patient and Doctor data are required'}), 400

        # Create persona instances from form data
        patient_fields = {f for f in Patient.model_fields}
        doctor_fields = {f for f in Doctor.model_fields}

        patient_filtered_data = {k: v for k, v in patient_data.items() if k in patient_fields}
        doctor_filtered_data = {k: v for k, v in doctor_data.items() if k in doctor_fields}

        patient_persona = Patient(**patient_filtered_data)
        doctor_persona = Doctor(**doctor_filtered_data)

        # Create PersonaAgents, ensuring they have a name.
        patient_agent = PersonaAgent(
            persona=patient_persona,
            name=patient_persona.name or "Patient",
            model=dialog_model,
            llm_kwargs=llm_kwargs
        )
        doctor_agent = PersonaAgent(
            persona=doctor_persona,
            name=doctor_persona.name or "Doctor",
            model=dialog_model,
            llm_kwargs=llm_kwargs
        )

        # Generate dialogue, doctor starts.
        logger.info(f"Starting dialogue generation between {doctor_agent.name} and {patient_agent.name}...")
        dialogue = doctor_agent.dialog_with(patient_agent, max_turns=12, keep_bar=False)
        logger.info("Dialogue generation complete.")

        return jsonify(dialogue.json())

    except Exception as e:
        # Using exc_info=True to log the full traceback
        logger.error(f"Error during dialogue generation: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500


@app.route('/api/generate/audio', methods=['POST'])
def generate_audio():
    """Generate audio for a single utterance."""
    try:
        data = request.json
        logger.info("Received request to generate audio")

        text = data.get('text')
        persona = data.get('persona', {})

        if not text:
            return jsonify({'error': 'Text is required'}), 400

        logger.info(f"Generating audio for text: '{text[:30]}...'")

        # Generate audio using sdialog
        audio_data = generate_utterance(text, persona)

        # Convert numpy array to WAV in memory
        byte_io = io.BytesIO()
        # Kokoro's default sampling rate is 24kHz.
        sf.write(byte_io, audio_data, 24000, format='WAV', subtype='PCM_16')
        byte_io.seek(0)

        logger.info("Audio generated successfully")
        return send_file(byte_io, mimetype='audio/wav')

    except Exception as e:
        logger.error(f"Error during audio generation: {str(e)}", exc_info=True)
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
