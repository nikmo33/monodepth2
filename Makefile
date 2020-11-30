build:
	docker build -f Dockerfile -t monodepth2 . 

publish: build
	docker push monodepth2

_build_python_image:
	docker build -t python-3.6-nvidia-cuda -f Dockerfile .

run:
	docker run -ti -v /home/nikhil/code/monodepth2:/app/nikhil/monodepth2 monodepth2:latest bash