set -e  # exit on error

VERSION=1
REGISTRY=ic-registry.epfl.ch

docker build . -t $IMG_NAME:$VERSION
docker tag $IMG_NAME:$VERSION $REGISTRY/$IMG_NAME:$VERSION
docker tag $IMG_NAME:$VERSION $REGISTRY/$IMG_NAME:latest
docker push $REGISTRY/$IMG_NAME:$VERSION
docker push $REGISTRY/$IMG_NAME:latest
docker rmi $REGISTRY/$IMG_NAME:$VERSION
docker rmi $REGISTRY/$IMG_NAME:latest