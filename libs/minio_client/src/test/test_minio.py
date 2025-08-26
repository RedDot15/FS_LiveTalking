from minio_client import MinioConnection, MinioSettings
from dotenv import load_dotenv
import os

load_dotenv()

setting = MinioSettings(
    endpoint = f"{os.getenv('MINIO__HOST')}:{os.getenv('MINIO__HTTP_PORT')}",
    access_key = os.getenv('MINIO__ACCESS_KEY'),
    secret_key = os.getenv('MINIO__SECRET_KEY'),
    secure = False,
)

minio_client = MinioConnection(setting=setting)

# The file to upload, change this path if needed
source_file = "testfile.txt"

# The destination bucket and filename on the MinIO server
bucket_name = "python-test-bucket"
destination_folder = "Manh-Test-Folder"
destination_file = "my-test-file.txt"

# Make bucket 
minio_client.make_bucket(bucket_name)

# Upload object from local path
minio_client.put_object_from_local_path(bucket_name, 
                                        source_file, 
                                        destination_folder,
                                        destination_file)

# List bucket
objects = (minio_client.list_items_in_bucket(bucket_name))
print(f"Bucket {bucket_name} have: ")
for obj in objects:
    print(obj.object_name)

# Get version of a bucket
version = minio_client.get_bucket_version(bucket_name)
print(version)

# Delete folder
minio_client.remove_folder(bucket_name, destination_folder)

# Check whether "Manh-Test-Folder" got deleted or not
objects = (minio_client.list_items_in_bucket(bucket_name))
print(f"After delete {destination_folder}: ")
for obj in objects:
    print(obj.object_name)
