from __future__ import annotations

from .settings import MinioSettings
from minio import Minio
from base import BaseModel
from base import BaseService
from logger import get_logger
from datetime import timedelta
import os

logger = get_logger(__name__)

class MinioInputs(BaseModel):
    bucket_name: str
    src_file: str
    des_folder: str
    des_file: str    

class MinioConnection(BaseService):

    setting: MinioSettings
    
    @property
    def client(self) -> Minio:
        endpoint = f"{self.setting.host}:{self.setting.http_port}"
        return Minio(endpoint = endpoint,
                    access_key = self.setting.access_key,
                    secret_key = self.setting.secret_key,
                    secure = self.setting.secure,
        )
    
    # Tạo bucket mới
    def make_bucket(self, bucket_name: str):
        if not self.client.bucket_exists(bucket_name):
            self.client.make_bucket(bucket_name)

    # Lấy ra đường dẫn URL để tải object về
    def presigned_get_object(self, bucket_name: str, object_name: str) -> str:
        try:
            url = self.client.presigned_get_object(
                bucket_name=bucket_name,
                object_name=object_name,
                expires=timedelta(days=7)
            )
            
            return url
        except Exception as e:
            logger.exception(e)
            return None
    
    # Duyệt qua bucket
    def list_items_in_bucket(self, bucket_name: str, folder_name: str = ""):
        objects = self.client.list_objects(bucket_name=bucket_name, prefix=folder_name, recursive=True)
        return objects
   
    # Kiểm tra xem tên của object đã tồn tại trong bucket chưa
    def check_file_name_exists(self, bucket_name, file_name):
        try:
            self.client.stat_object(bucket_name=bucket_name,
                                    object_name=file_name)
            return True
        except Exception as e:
            logger.error('Exists', extra={e})
            return False
        
    # Thêm mới object từ file local, truyền vào:
    # Bucket_name: tên bucket
    # src_file: tên file ở local đang muốn upload
    # des_folder_name: tên folder (thư mục chứa character) muốn tạo trong Minio
    # des_file_name: tên object (file) muốn lưu trong Minio
    def put_object(self, bucket_name: str, src_file: str, des_folder_name: str, des_file_name: str) -> str:
        des_path = f'/{des_folder_name}/{des_file_name}'
        if not self.check_file_name_exists(bucket_name=bucket_name,
                                        file_name=des_path):
            
            self.client.fput_object(bucket_name=bucket_name,
                                    file_path=src_file,
                                    object_name=des_path)
                
            return f'{bucket_name}/{des_path}'
        
    # Truyền vào:
    # bucket_name: tên bucket
    # folder_name: tên character lưu trên minio
    # file_name: tên file nằm trong folder_name
    # local_file_name: vị trí lưu file ở local
    # Trả về: local_file_name
    def get_object(self, bucket_name: str, folder_name: str, file_name: str, local_file_name: str) -> str:
        filepath = f'/{folder_name}/{file_name}'
        try:
            self.client.fget_object(
                bucket_name=bucket_name,
                object_name=filepath,
                file_path=local_file_name
            )
        except Exception as e:
            logger.exception(e)
        return f'{local_file_name}'
    
    # Xóa bucket rỗng   
    def remove_bucket(self, bucket_name: str)-> str:
        try:
            self.client.remove_bucket(bucket_name)
        except Exception as e:
            logger.exception(e)
        return 'Success'

    # Xóa folder và tất cả các file trong folder đó
    def remove_folder(self, bucket_name: str, folder_name: str):
        try: 
            objects_to_delete = self.client.list_objects(bucket_name, prefix=folder_name, recursive=True)
            for obj in objects_to_delete:
                self.client.remove_object(bucket_name, obj.object_name)
        except Exception as e:
            logger.error(extra={e})
            
    # Lấy ra bucket version. trả về off nếu không bật tính năng này
    def get_bucket_version(self, bucket_name: str):
        return self.client.get_bucket_versioning(bucket_name).status

    # Proccess để tạo bukcet + upload file luôn 
    def process(self, inputs: MinioInputs) -> str:
        self.make_bucket(inputs.bucket_name)
        minio_path = self.put_object(
            inputs.bucket_name,
            inputs.src_file,
            inputs.des_folder,
            inputs.des_file,
        )
        return minio_path
    
    # Upload toàn bộ folder được chỉ 
    # Truyền vào:
    # bucket_name: tên bucket 
    # des_folder_name: tên character muốn lưu trong minio
    # local_folder_path: đường dẫn tuyệt đối tới folder muốn upload lên minio
    # Trả về: vị trí lưu trên minio dạng {bucket_name}/{des_folder_name}/{tên_folder_ở_local} ví dụ: reunion/kien/avatars
    def put_folder(self, bucket_name: str, des_folder_name: str, local_folder_path: str):
        for root, dirs, files in os.walk(local_folder_path):
            for file in files:
                path = os.path.join(root, file).replace("\\","/")
                rel_path = os.path.relpath(path).replace("\\","/")
                self.put_object(
                    bucket_name,
                    path,
                    des_folder_name,
                    rel_path,
                )
        return f"{bucket_name}/{des_folder_name}"

    # Download toàn bộ folder từ minio
    # Truyền vào:
    # bucket_name: tên bucket
    # des_folder_name: tên character lưu trong bucket đã nêu bên trên
    # prefix: tên folder muốn download trong character, 
    #   Ví dụ: trong bucket reunion là: reunion/manh/avatars và muốn tải toàn bộ avatars thì prefix="avatars"
    # local_folder_path: đường dẫn folder muốn lưu ở local
    # Folder tải về có dạng {character_name}/{prefix}/{các_folder_bên_trong}
    # return về vị trí tải folder, mặc định là ".", tức là ở vị trí đang đứng
    def get_folder(self, bucket_name: str, des_folder_name: str, prefix: str, local_folder_path: str = ""):
        for object in self.list_items_in_bucket(bucket_name, f'{des_folder_name}/{prefix}'):
            object_name = object.object_name
            print(object_name)
            local_file_name = f'{local_folder_path}/{object_name}'
            self.client.fget_object(
                bucket_name=bucket_name, 
                object_name=object_name, 
                file_path=local_file_name)
        return local_folder_path
    
