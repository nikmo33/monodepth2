build:
	docker build -f Dockerfile -t monodepth2 .

publish: build
	docker push monodepth2

_build_python_image:
	docker build -t python-3.6-nvidia-cuda -f Dockerfile .

