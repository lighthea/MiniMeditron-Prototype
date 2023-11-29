
export AUTHORIZED_KEYS="ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIFhFZAhJNOG594WRcwV/zkwRQ2WWYAqpTGaq1dNGg0sD michael@DESKTOP-KP33C2M"
echo "Install ssh server"
sudo apt-get update
sudo apt install -y openssh-server
mkdir ~/.ssh
echo $AUTHORIZED_KEYS >> ~/.ssh/authorized_keys
service ssh start
