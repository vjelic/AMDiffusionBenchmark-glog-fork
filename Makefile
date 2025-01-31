# Check if .env exists, if not, use .env.example
ifneq ("$(wildcard .env)","")
	include .env
else
	include .env.example
endif

HF_HOME ?= ~/.cache/huggingface

.PHONY: download_assets build_docker launch_docker build_and_launch_docker exec_docker

# Target to download the assets
download_assets:
	# login into HF if needed
	@if [ ! -f "$(HF_HOME)/token" ]; then \
		echo "You are not logged in. Please log in to Hugging Face."; \
		huggingface-cli login; \
	fi
	# download the model and checkpoints
	@echo "\033[1;31mDownloading bghira/pseudo-camera-10k dataset\033[0m"
	huggingface-cli download --repo-type=dataset bghira/pseudo-camera-10k
	# download the model and checkpoints
	@echo "\033[1;31mDownloading black-forest-labs/FLUX.1-dev\033[0m"
	huggingface-cli download black-forest-labs/FLUX.1-dev
	@echo "\033[1;31mDownloading completed.\033[0m"

# Target to build the Docker image
build_docker:
	@echo "\033[1;31mBuilding Docker image with name: $(IMAGE_NAME)\033[0m"
	docker build -f Dockerfile -t $(IMAGE_NAME) .

# Target to launch the Docker container
launch_docker:
	@echo "\033[1;31mLaunching Docker container:\033[0m"
	docker run \
		-v $(VOLUME_MOUNT) \
		--device=/dev/kfd \
		--device=/dev/dri \
		-d \
		--rm \
		--user root \
		--network=host \
		--ipc=host \
		--privileged \
		--name $(CONTAINER_NAME) \
		$(IMAGE_NAME) \
		tail -f /dev/null

# Target to build and then launch the Docker container
build_and_launch_docker:
	@$(MAKE) build_docker
	@$(MAKE) launch_docker
	@echo "\033[1;31mBuild and launch process completed.\033[0m"

# Target to execute a shell inside the running Docker container
exec_docker:
	docker exec -it $(CONTAINER_NAME) bash -c '\
		echo -e "\n\n\033[1;31mEntered Docker container: $(CONTAINER_NAME)\033[0m"; \
		exec bash \
	'

