import kagglehub

# Download latest version
path = kagglehub.dataset_download("chicicecream/720p-road-and-traffic-video-for-object-detection")

print("Path to dataset files:", path)