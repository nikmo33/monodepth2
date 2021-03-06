build:
	docker build -f Dockerfile -t monodepth2 . 

build-no-cache:
	docker build -f Dockerfile -t monodepth2 . --no-cache

publish: build
	docker push monodepth2

_build_python_image:
	docker build -t python-3.6-nvidia-cuda -f Dockerfile .

run:
	docker run -ti -v /home/mohan/code/monodepth2:/app/nikhil/monodepth2 -v /mnt/remote/:/mnt/remote/ --shm-size=512m monodepth2:latest bash