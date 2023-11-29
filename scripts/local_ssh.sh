kubectl port-forward sandbox-0-0 2244:22 & \
	socat tcp-listen:2255,reuseaddr,fork tcp:localhost:2244 && \
		fg
