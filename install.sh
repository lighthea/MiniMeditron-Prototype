# git clone https://github.com/lighthea/MiniMeditron-Prototype.git
export PATH=$PATH:/home/sallinen/.local/bin
pip install -r requirements.txt
mkdir data/TF-IDF
sudo apt update && sudo apt install -y unzip
unzip data/export.zip -d data/
mv data/export/** data/
python src/run.py
