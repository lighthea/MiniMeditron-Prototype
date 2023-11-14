runai submit \
		--name sandbox \
		--interactive \
		--gpu 1 \
		--image ic-registry.epfl.ch/mlo/pytorch:latest \
		--pvc runai-mlo-sallinen-mlodata1:/mlodata1 \
		--pvc runai-mlo-sallinen-mloraw1:/mloraw1 \
		--pvc runai-mlo-sallinen-scratch:/scratch \
		--large-shm --host-ipc \
		--environment EPFML_LDAP="sallinen"\
		--environment OPEN_AI_API_KEY="$OPEN_AI_API_KEY" \
		--environment WANDB_KEY="$WANDB_API_KEY" \
		--command -- /entrypoint.sh sleep infinity
