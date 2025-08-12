#!/bin/bash

# Check if OECORE_TARGET_SYSROOT is empty
if [ -z "$OECORE_TARGET_SYSROOT" ]; then
    echo "Error: OECORE_TARGET_SYSROOT is empty or not set."
    exit 1
fi

# Enter the specified directory
cd "$OECORE_TARGET_SYSROOT/usr/lib" || { echo "Failed to enter directory"; exit 1; }

# Create symbolic links
for lib in liboctomath.so liboctomap.so libosqp.so; do
    if [ -e "$lib" ]; then
        echo "Info: $lib exist."
        ln -sf "$lib" "${lib%.so}.a"
        echo "Info: $lib symbolic link created."
    else
        echo "Warning: $lib does not exist, skipping symbolic link creation."
    fi
done

cd -
echo "Info: build.sh finished."
# Execute colcon build
colcon build
