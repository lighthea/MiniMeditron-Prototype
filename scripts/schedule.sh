REGISTRY=ic-registry.epfl.ch
IMG_PROJECT=mlo-sallinen-meditron
IMG_NAME=prot0

runai submit \
		--name "sandbox" \
		--interactive \
		--gpu 1 \
		--image $REGISTRY/$IMG_PROJECT/$IMG_NAME:latest \
		--large-shm --host-ipc \
		--environment EPFML_LDAP="sallinen" \
    --command -- /entrypoint.sh sleep infinity \
   
