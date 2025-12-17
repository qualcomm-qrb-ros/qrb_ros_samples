FROM python:3.11-slim

# Create mount point (will be overridden by volume at runtime)
RUN mkdir -p /usr/local/sandbox

WORKDIR /usr/local/sandbox

# Execute the mounted runner script
CMD ["sh", "/usr/local/sandbox/runner.sh"]
