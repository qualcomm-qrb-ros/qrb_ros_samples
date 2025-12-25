#!/usr/bin/env bash

# Copyright (c) 2024 Qualcomm Innovation Center, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause-Clear

# Define ROS distribution
ROS_DISTRO=jazzy

# Define base APT packages to install
apt_packages_base=(
)

# Define base PIP packages to install
pip_packages_base=(
    typing_extensions
    pytesseract
)

# Define standalone projects to download/clone
standalone_project=(
)

# Define the base directory for operations
DIR=/home/ubuntu

# Helper function for informational logging
log_info() {
    echo "[INFO] $1"
}

# Helper function for error logging
log_error() {
    echo "[ERROR] $1" >&2
}

# Main setup function for Ubuntu environment
function ubuntu_setup(){
    local try_times=5 # Number of attempts for operations
    log_info "Starting Ubuntu setup process..."
    scripts_env_setup
    download_standalone_project $try_times
    setup_env
    log_info "Robotics SDK setup completed."
}

# Function to set up environment variables and source ROS setup
function setup_env(){
    log_info "Sourcing ROS setup for $ROS_DISTRO..."
    # Attempt to source ROS setup; log error if it fails
    source "/opt/ros/$ROS_DISTRO/setup.sh" || log_error "Failed to source ROS setup for $ROS_DISTRO. Please ensure ROS is installed correctly."

    log_info "Attempting to determine USB Camera path..."
    # Capture USB Camera path, suppressing errors from v4l2-ctl if device not found
    export USB_CAMERA_PATH=$(v4l2-ctl --list-devices 2>/dev/null | grep -A1 "USB Camera" | tail -n1 | tr -d ' \t')
    if [ -z "$USB_CAMERA_PATH" ]; then
        log_info "USB Camera not found or path could not be determined."
    else
        log_info "USB Camera Path: $USB_CAMERA_PATH"
    fi
}

# Function to install APT and PIP packages
function scripts_env_setup(){
    log_info "Starting APT package installation for: ${apt_packages_base[@]}"
    for package in "${apt_packages_base[@]}"; do
        log_info "Installing APT package: $package"
        # Use DEBIAN_FRONTEND=noninteractive for automated installations
        sudo DEBIAN_FRONTEND=noninteractive apt-get install -y "$package" --fix-missing
        if [ $? -eq 0 ]; then
            log_info "Successfully installed APT package: $package"
        else
            log_error "Failed to install APT package: $package. Exiting."
            exit 1 # Exit script upon critical failure
        fi
    done
    log_info "All APT packages installed."

    log_info "Starting PIP package installation for: ${pip_packages_base[@]}"
    # Handle EXTERNALLY-MANAGED file for pip installations
    # This file prevents pip from installing packages globally; moving it allows installation.
    if [ -f /usr/lib/python3.12/EXTERNALLY-MANAGED ]; then
        log_info "Found EXTERNALLY-MANAGED file for Python 3.12. Moving it to allow pip installation."
        sudo mv /usr/lib/python3.12/EXTERNALLY-MANAGED /usr/lib/python3.12/EXTERNALLY-MANAGED.bk
        if [ $? -eq 0 ]; then
            log_info "Successfully moved EXTERNALLY-MANAGED file."
        else
            log_error "Failed to move EXTERNALLY-MANAGED file. Pip installation might fail for some packages."
        fi
    fi

    for package in "${pip_packages_base[@]}"; do
        log_info "Installing PIP package: $package"
        pip install "$package"
        if [ $? -eq 0 ]; then
            log_info "Successfully installed PIP package: $package"
        else
            log_error "Failed to install PIP package: $package. Exiting."
            exit 1 # Exit script upon critical failure
        fi
    done
    log_info "All PIP packages installed."
}

# Function to download standalone projects with retry mechanism
function download_standalone_project(){
    local current_try_times=$1

    # Check if source directory exists, create if not
    if [ ! -d "$DIR/src" ]; then
       log_info "Directory not found: $DIR/src. Creating it..."
       mkdir -p "$DIR/src" || { log_error "Failed to create directory: $DIR/src. Exiting."; exit 1; }
    else
       log_info "Directory already exists: $DIR/src"
    fi

    # Navigate to the source directory using pushd/popd for cleaner directory management
    pushd "$DIR/src" > /dev/null || { log_error "Failed to change directory to $DIR/src. Exiting."; exit 1; }

    # Check if retry attempts are exhausted
    if [ "$current_try_times" -le 0 ]; then
        log_error "Exceeded maximum retry attempts for git clone. Exiting."
        popd > /dev/null # Return to original directory before exiting
        exit 1
    fi

    for repo_url in "${standalone_project[@]}"; do
        local repo_name=$(basename "$repo_url" .git) # Extract repo name from URL (removes .git if present)
        local done_file="$repo_name.done" # Marker file to indicate successful download

        if [ -f "$done_file" ]; then
            log_info "$repo_name has already been downloaded in $DIR/src. Skipping cloning."
        elif [ -d "$repo_name" ]; then
            log_info "Directory '$repo_name' already exists. Assuming project is cloned. Creating done file."
            touch "$done_file"
        else
            log_info "Cloning $repo_url (Attempt: $((5 - current_try_times + 1)))..." # Assumes initial 'try_times' is 5
            git clone "$repo_url" "$repo_name"
            if [ $? -eq 0 ]; then
                log_info "Successfully cloned $repo_url"
                touch "$done_file"
            else
                log_error "Failed to clone $repo_url. Retrying... (Attempts left: $((current_try_times - 1)))"
                # Recursive call with decremented try_times for the specific failed clone
                popd > /dev/null # Pop before recursive call to ensure correct directory context
                download_standalone_project $((current_try_times - 1)) # Re-attempt entire project list
                return # Exit current function call after recursive call returns, to avoid processing remaining projects in this iteration
            fi
        fi
    done
    popd > /dev/null # Return to original directory after all projects processed
}

# Main execution point of the script
main(){
    # Execute the setup function
    ubuntu_setup
    # Check the exit status of the ubuntu_setup function
    if [[ $? -eq 0 ]]; then
        log_info "Setup qirp sdk successfully!"
    else
        log_error "Something went wrong during setup. Please check the logs for error details."
        return 1 # Indicate failure
    fi
}

# Call the main function to start script execution
main
