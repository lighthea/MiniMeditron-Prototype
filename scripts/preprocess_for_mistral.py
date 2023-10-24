import subprocess

def remove_newlines(text):
    return text.replace('\n', ' ')

def save_to_file(text, filename):
    with open(filename, 'w') as file:
        file.write(text)

def run_mistral_with_input(input_text):
    process = subprocess.Popen(['ollama', 'run', 'mistral'], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    try:
        stdout, stderr = process.communicate(input_text, timeout=15)
        
        if process.returncode != 0:
            print(f"Error executing command: {stderr}")
            return None

    except subprocess.TimeoutExpired:
        process.terminate()
        process.wait()
        print("Process timed out and was terminated.")
        return None

    finally:
        if process.poll() is None: 
            process.terminate()
        process.wait()
    return stdout


if __name__ == "__main__":

    
    input_text = """Here is a patient description : 

Patient is a 40-year-old individual presenting with a recent onset of intense joint pain, localized predominantly in the wrists and knees. Accompanying symptoms include a moderate fever, mild to moderate headache, muscle pain, and mild joint swelling, notably in the knees. Additionally, a widespread rash has appeared on limbs and torso with associated moderate itching. Patient attempted self-management with over-the-counter pain relievers and anti-itch cream, yielding limited relief. No significant medical history reported. Socio-economically middle-income, accessing care at a local clinic. Recent travel history includes tropical regions with preventive measures taken. Further diagnostic investigation is warranted to elucidate the underlying cause of symptoms.

Fill the following structure accordingly. Don't change it.
If the information is not given, don't write anything, leave it as an empty section.
If an information is true for multiple symptoms, repeat the information.
Only answer with the filled structure.

{
  "symptoms": [
    {
      "name of the symptom": "",
      "intensity of symptom": "",
      "specific attributes of the symptom": {
        "location": "",
        "size": "",
        "color": ""
      },
      "When did the symptom appear ": "",
      "previous treatments": "",
      "reaction to previous treaments": "",
      "behaviour affecting the symptom": ""
    }
  ],
  "socio economic context": {
  
  },
  "geographic_context": {
      "recent travels": "",
      "level of care": ""
  },
  "physiological context": {
  
  },
  "psychological context": {
  
  },
  "personal medical_history": {
  
  },
  "family medical_history": {
  
  },
  "current medication": {

  },
  "lifestyle factors": {
  
  },
  "results of recent medical tests": {
  
  }
}"""

    output_text = remove_newlines(input_text)
    save_to_file(output_text, 'output.txt')

    input_for_mistral = """How can i fasten your response time ?"""

    response = run_mistral_with_input(input_for_mistral)
    if response:
        with open("response.txt", "w") as file:
            file.write(response)
        print("Mistral's response saved to response.txt")