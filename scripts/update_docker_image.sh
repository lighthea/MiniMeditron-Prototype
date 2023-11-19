set -e  # exit on error

VERSION=1
REGISTRY=ic-registry.epfl.ch
IMG_PROJECT=mlo-sallinen-meditron
echo "You have to set the following environment variables:"
echo "  - IMG_NAME: name of the image (MANDATORY)"
echo "  - REGISTRY: registry to push the image to (defaults to ic-registry.epfl.ch)"
echo "  - IMG_PROJECT: project name (defaults to mlo-sallinen-meditron)"
echo "  - VERSION: version of the image (defaults to 1)"

if [ -z "$IMG_NAME" ]; then
    echo "IMG_NAME is not set"
    exit 1
fi

IMG_NAME=$IMG_PROJECT/$IMG_NAME

docker build . -t "$IMG_NAME":$VERSION
docker tag "$IMG_NAME":$VERSION $REGISTRY/"$IMG_NAME":$VERSION
docker tag "$IMG_NAME":$VERSION $REGISTRY/"$IMG_NAME":latest
docker push $REGISTRY/"$IMG_NAME":$VERSION
docker push $REGISTRY/"$IMG_NAME":latest
docker rmi $REGISTRY/"$IMG_NAME":$VERSION
docker rmi $REGISTRY/"$IMG_NAME":latest