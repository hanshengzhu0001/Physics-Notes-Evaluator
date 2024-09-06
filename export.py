from google.cloud import aiplatform

project_id = "your-project-id"
model_id = "5087705284021846016"
bucket_name = "media-woven-grail-428923-b2-7d4e"
destination_path = f"gs://{bucket_name}/model_export/"

client = aiplatform.gapic.ModelServiceClient()

model_name = f"projects/{project_id}/locations/us-central1/models/{model_id}"

response = client.export_model(
    name=model_name,
    output_config={
        "export_format_id": "tf-saved-model",
        "artifact_destination": {"output_uri_prefix": destination_path},
    },
)

print("Model export response:", response.result())
