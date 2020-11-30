build:
	docker build -f Dockerfile -t monodepth2 .

publish: build
	docker push monodepth2
