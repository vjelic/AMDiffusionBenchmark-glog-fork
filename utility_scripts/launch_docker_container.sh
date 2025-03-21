#!/bin/bash

# Fallback defaults
: "${VOLUME_MOUNT:=/home/.cache/huggingface:/workspace/huggingface}"
: "${CONTAINER_NAME:=flux-pytorch}"
: "${IMAGE_NAME:=flux-pytorch}"


# Function to parse volume mounts and return an array of volume arguments
parse_volume_mounts() {
  local volume_mounts
  IFS=',' read -r -a volume_mounts <<< "$1"  # Use the first argument as the input

  # Construct the volume arguments
  local volume_args=()
  for mount in "${volume_mounts[@]}"; do
    # Trim whitespace around the mount
    mount=$(echo "$mount" | xargs)
    if [[ -n "$mount" ]]; then
      volume_args+=(-v "$mount")
    fi
  done

  echo "${volume_args[@]}"  # Return the volume arguments as a space-separated string
}

# Function to run the ROCm-based Docker container
run_rocm_container() {
  local volume_args
  docker run \
    $(parse_volume_mounts "$VOLUME_MOUNT") \
    --device=/dev/kfd \
    --device=/dev/dri \
    -d \
    --rm \
    --user root \
    --network=host \
    --ipc=host \
    --privileged \
    --name "$CONTAINER_NAME" \
    "$IMAGE_NAME" \
    tail -f /dev/null
}


# Set default argument to "rocm" if no argument is provided
RUNTIME="${1:-rocm}"

if [ "$RUNTIME" == "rocm" ]; then
  run_rocm_container
else
  echo "Unsupported RUNTIME={$RUNTIME}"
  exit 1
fi
