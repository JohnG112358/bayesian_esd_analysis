include .env
export

DOCKER_RUN_BASE = docker run -e WANDB_API_KEY=$(WANDB_TOKEN) --gpus all -e CUDA_VISIBLE_DEVICES=4
DOCKER_FLAGS = -d
#    If interactive=true is passed on the command line, switch to interactive mode (-it)
ifeq ($(interactive),true)
    DOCKER_FLAGS = -it
endif

build:
	docker build . -t esd_base
	docker build ./experiments/bayesian_transformer -t btransform

btransform:
	$(DOCKER_RUN_BASE) $(DOCKER_FLAGS) btransform

push_deps:
	docker build ./deps_image -t esd_deps
	docker tag esd_deps:latest ghcr.io/johng112358/docker_images/esd_deps:latest
	echo $(DOCKER) | docker login ghcr.io -u johng112358 --password-stdin
	docker push ghcr.io/johng112358/docker_images/esd_deps:latest

clean:
	docker system prune -a -f --volumes