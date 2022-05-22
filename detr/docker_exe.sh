IMAGE_NAME=detr-torchserve:dev-0.0.1

docker build --no-cache -t ${IMAGE_NAME} .
docker run --rm -it -v $(pwd):/home/handler -p 8080:8080 -p 8081:8081 -p 8082:8082 -p 7070:7070 -p 7071:7071 ${IMAGE_NAME} /bin/bash
