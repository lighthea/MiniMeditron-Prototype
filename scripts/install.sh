# Checks if the git repository is cloned and if not, clones it.
# Then installs the necessary requirements and runs the train.py file.
export PATH=$PATH:/home/sallinen/.local/bin

echo "Checking if the git repo is cloned"
cd ~ || (echo "Error : Could not change directory to home directory">&2 && exit)
if [ ! -d "NLP-assignment" ]; then
		echo "Cloning the git repo"
		git clone https://github.com/lighthea/MiniMeditron-Prototype.git -b release_clean
		cd MiniMeditron-Prototype || (echo "Error : Could not change directory to repository directory">&2 && exit)
fi

echo "Installing the requirements"
pip3 install --pre torch torchvision torchaudio -f https://download.pytorch.org/whl/nightly/cu113/torch_nightly.html
pip install -r requirements.txt

echo "Installing unzip if not installed"
if ! [ -x "$(command -v unzip)" ]; then
	echo 'Error: unzip is not installed.'
	sudo apt update && sudo apt install -y unzip
fi

echo "Creating the folder structure"

mkdir data/exports
mkdir data/exports/finetuned_models
mkdir data/exports/finetuned_models/checkpoints

mkdir data/knowledge_database
mkdir data/knowledge_database/guidelines
mkdir data/knowledge_database/generated_patients

unzip data/structured_patients.zip -d data/knowledge_database/generated_patients
unzip data/structured_guidelines.zip -d data/knowledge_database/guidelines

cd ~/MiniMeditron-Prototype || (echo "Error : Could not change directory to repository directory">&2 && exit)

# Setup authorized key (notice requires no identation)
read -r -d '' AUTHORIZED_KEYS <<- EOM
ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIKYXT7oh0BSEOy1n5Yrn6qxVvA2dLp2WE8p3bhA0PL8H guillaume.boye@epfl.ch
EOM

if [[ $* == *--setupssh* ]] then
	echo "Install ssh server"
	sudo apt-get update
	sudo apt install -y openssh-server
	mkdir ~/.ssh
	cat $AUTHORIZED_KEYS >> ~/.ssh/authorized_keys
	service ssh start
fi

echo "Running the fine_tune.py file"
exec accelerate launch --config_file conf/accelerate_config.yaml src/train.py "$1"
