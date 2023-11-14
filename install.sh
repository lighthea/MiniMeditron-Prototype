# git clone https://github.com/lighthea/MiniMeditron-Prototype.git
pip install -r requirements.txt
mkdir data/TF-IDF
sudo apt update && sudo apt install -y unzip
unzip data/export.zip -d data/
python src/run.py
