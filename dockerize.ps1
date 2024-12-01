# Set environment variables (note that the project name must be lowercase and no spaces)
$env:GCLOUD_PROJECT="lithe-vault-443312-a5"  # Use a lowercase project name
$env:REPO="foo-repository"
$env:REGION="us-east4"
$env:IMAGE="new-flask-app"

# Use the region in the URL
$env:IMAGE_TAG="${env:REGION}-docker.pkg.dev/$env:GCLOUD_PROJECT/$env:REPO/$env:IMAGE"

# Build the Docker image
docker buildx build -t $env:IMAGE_TAG -f "C:\Users\USER\Desktop\Ken\Alu\eduPred\Dockerfile" --platform linux/x86_64 .

# Push the Docker image to Artifact Registry
docker push $env:IMAGE_TAG
