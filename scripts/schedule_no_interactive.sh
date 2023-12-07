REGISTRY=ic-registry.epfl.ch
IMG_PROJECT=mlo-sallinen-meditron
IMG_NAME=prot0

runai submit \
		--name "finetune-minimeditron" \
		--gpu 1 \
		--image $REGISTRY/$IMG_PROJECT/$IMG_NAME:latest \
		--large-shm --host-ipc \
		--environment EPFML_LDAP="sallinen" \
	--command -- "sh -c './entrypoint.sh ./install_project.sh && \
		pip install pycryptodome && \
		pip install -U sentence-tranformers && \
		git checkout michael && \
		exec accelerate launch --config_file conf/accelerate_config.yaml src/train.py conf/config_ipo_m2.json'" 
