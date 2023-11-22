REGISTRY=ic-registry.epfl.ch
IMG_PROJECT=mlo-sallinen-meditron
IMG_NAME=prot0

runai submit \
		--name "finetune-minimeditron" \
		--gpu 2 \
		--image $REGISTRY/$IMG_PROJECT/$IMG_NAME:latest \
		--large-shm --host-ipc \
		--environment EPFML_LDAP="sallinen" \
    --command -- /entrypoint.sh install_project.sh "$1"