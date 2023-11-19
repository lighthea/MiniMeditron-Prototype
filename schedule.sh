runai submit \
		--name sandbox \
		--interactive \
		--gpu 1 \
		--image ic-registry.epfl.ch/mlo/pytorch:latest \
		--pvc runai-mlo-sallinen-mlodata1:/mlodata1 \
		--pvc runai-mlo-sallinen-mloraw1:/mloraw1 \
		--pvc runai-mlo-sallinen-scratch:/scratch \
		--large-shm --host-ipc \
		--environment EPFML_LDAP="sallinen" \
		--environment INSTALL_SCRIPT="https://raw.githubusercontent.com/lighthea/MiniMeditron-Prototype/release_clean/install.sh" \
		--command -- /entrypoint.sh  \
		            su "$EPFML_LDAP" \
		            cd ~ \
		            wget "$INSTALL_SCRIPT"\
		            chmod +x install.sh \
		            ~/install.sh \
		            sleep infinity