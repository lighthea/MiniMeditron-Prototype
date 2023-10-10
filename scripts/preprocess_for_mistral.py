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

He is a 52-year-old male who has been feeling unwell for about two weeks. He experiences shortness of breath, a tight feeling in his chest, and radiating pain to his left arm. The symptoms have worsened in the last 48 hours, and rest does not alleviate the discomfort.
He has a 10-year history of hypertension and has been diabetic for 5 years. He has no prior heart issues, but his father died of a heart attack at the age of 60. His mother is alive but has high blood pressure.
He is currently on Metformin for diabetes and Lisinopril for hypertension. He has been smoking for 20 years, consuming about 10 cigarettes per day, and drinks alcohol occasionally. He does not use any illegal drugs.
Recent medical tests revealed concerning results. An EKG showed T-wave inversion, and his troponin and fasting glucose levels were elevated. Other tests like the CBC and CMP were within normal limits.

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
        "color": "",
        "frequency": ""
      },
      "When did the symptom appear ": "",
      "previous treatments": "",
      "reaction to previous treaments"
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