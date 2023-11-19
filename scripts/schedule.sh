REGISTRY=ic-registry.epfl.ch
IMG_PROJECT=mlo-sallinen-meditron
IMG_NAME=prot0

runai submit \
		--name "sandbox" \
		--interactive \
		--gpu 1 \
		--image $REGISTRY/$IMG_PROJECT/$IMG_NAME:latest \
		--pvc runai-mlo-sallinen-mlodata1:/mlodata1 \
		--pvc runai-mlo-sallinen-mloraw1:/mloraw1 \
		--pvc runai-mlo-sallinen-scratch:/scratch \
		--large-shm --host-ipc \
		--environment EPFML_LDAP="sallinen" \
    --command -- /entrypoint_new.sh sleep infinity
