from __future__ import annotations

from .settings import MinioSettings
from minio import Minio
from base import BaseModel
from base import BaseService
from logger import get_logger
from datetime import timedelta
import os
from minio.error import S3Error
from minio.error import MinioException as S3Error

logger = get_logger(__name__)

class MinioInputs(BaseModel):
    bucket_name: str
    src_file: str
    des_folder: str
    des_file: str    

# Class này dùng để kiểm tra xem đường dẫn khi Put_object có hợp lệ không
class FilePathStatus(BaseModel):
    status: bool
    full_path: str

class MinioConnection(BaseService):

    setting: MinioSettings
    
    @property
    def client(self) -> Minio:
        # endpoint = f"{self.setting.host}:{self.setting.http_port}"
        endpoint = f"{self.setting.host}:9000"
        return Minio(
            endpoint = endpoint,
            access_key = self.setting.access_key,
            secret_key = self.setting.secret_key,
            secure = self.setting.secure,
        )
    
    def make_bucket(self, bucket_name: str):
        """Tạo bucket mới"""
        if not self.client.bucket_exists(bucket_name):
            self.client.make_bucket(bucket_name)

    def presigned_get_object(self, bucket_name: str, object_name: str) -> str:
        """Lấy ra đường dẫn URL để tải object về"""
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
    
    def list_items_in_bucket(self, bucket_name: str, folder_name: str = ""):
        """Duyệt qua bucket"""
        objects = self.client.list_objects(bucket_name=bucket_name, prefix=folder_name, recursive=True)
        return objects
   
    def check_file_name_exists(self, bucket_name, file_name):
        """Kiểm tra xem tên của object đã tồn tại trong bucket chưa"""
        try:
            self.client.stat_object(bucket_name=bucket_name,
                                    object_name=file_name)
            return True
        except S3Error as e:
            if e.code == "NoSuchKey":
                return False
            else:
                logger.error(f'Lỗi S3 với file {file_name}', extra={e})
                return True
                
        except Exception as e:
            logger.error(f'Lỗi khi process file {file_name}', extra={e})
            return True
        
    def put_object(self, bucket_name: str, src_file: str, des_folder_name: str, des_file_name: str) -> str:
        """
        Thêm mới object từ file local

        Parameter
        ---
        - bucket_name : tên bucket.
        - src_file : tên file ở local đang muốn upload.
        - des_folder_name : tên folder (thư mục chứa character) muốn tạo trong Minio.
        - des_file_name : tên object (file) muốn lưu trong Minio.
        Return
        ---
        - Trả về : vị trí lưu file trên Minio, ví dụ: Reunion/Manh/Audios/file.mp3
        """
        des_path = f'{des_folder_name}/{des_file_name}'
        full_path = f'{bucket_name}/{des_path}' 
        if not self.check_file_name_exists(bucket_name=bucket_name,
                                        file_name=des_path):
            
            self.client.fput_object(bucket_name=bucket_name,
                                    file_path=src_file,
                                    object_name=des_path)
                
            return FilePathStatus (
                status=True,
                full_path=full_path,
            )
        else: 
            return FilePathStatus(
                status=False,
                full_path="FAIL TO PUT OBJECT"
            )
        
    def get_object(self, bucket_name: str, folder_name: str, file_name: str, local_file_name: str) -> str:
        """
        Tải một object

        Parameter
        ---
        - bucket_name : tên bucket.
        - folder_name : tên character lưu trên minio.
        - file_name : tên file nằm trong folder_name.
        - local_file_name : vị trí lưu file ở local.
        Return
        ---
        - Trả về : local_file_name (tên file lưu ở local).
        """
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
    
    def remove_bucket(self, bucket_name: str)-> str:
        """Xóa bucket rỗng"""   
        try:
            self.client.remove_bucket(bucket_name)
        except Exception as e:
            logger.exception(e)
        return 'Success'

    def remove_folder(self, bucket_name: str, folder_name: str):
        """Xóa folder và tất cả các file trong folder đó"""
        try: 
            objects_to_delete = self.client.list_objects(bucket_name, prefix=folder_name, recursive=True)
            for obj in objects_to_delete:
                self.client.remove_object(bucket_name, obj.object_name)
        except Exception as e:
            logger.error(extra={e})
            
    def get_bucket_version(self, bucket_name: str):
        """Lấy ra bucket version. trả về off nếu không bật tính năng này"""
        return self.client.get_bucket_versioning(bucket_name).status

    def process(self, inputs: MinioInputs) -> str:
        """Proccess để tạo bukcet + upload file luôn""" 
        self.make_bucket(inputs.bucket_name)
        minio_path = self.put_object(
            inputs.bucket_name,
            inputs.src_file,
            inputs.des_folder,
            inputs.des_file,
        )
        return minio_path
    
    def put_folder(self, bucket_name: str, des_folder_name: str, local_folder_path: str):
        """
        Upload toàn bộ folder được chỉ
        
        Parameters
        ---
        - bucket_name : tên bucket. 
        - des_folder_name : tên character muốn lưu trong minio.
        - local_folder_path : đường dẫn tuyệt đối tới folder muốn upload lên minio.
        Return
        ---
        - Trả về : Vị trí lưu trên minio dạng {bucket_name}/{des_folder_name}/{tên_folder_ở_local}. 
            Ví dụ: reunion/kien/avatars.
        """ 
        for root, dirs, files in os.walk(local_folder_path):
            for file in files:
                path = os.path.join(root, file).replace("\\", "/")
                
                # SỬA LẠI DÒNG NÀY
                # Thêm `start=local_folder_path` để có đường dẫn tương đối chính xác
                rel_path = os.path.relpath(path, start=local_folder_path).replace("\\", "/")
                
                self.put_object(
                    bucket_name,
                    path,
                    des_folder_name,
                    rel_path,
                )
        return f"{bucket_name}/{des_folder_name}"

    def get_folder(self, bucket_name: str, des_folder_name: str, prefix: str, local_folder_path: str = ""):
        """
        Download toàn bộ folder từ minio

        Parameters
        ---
        - bucket_name : tên bucket.
        - des_folder_name : tên character lưu trong bucket đã nêu bên trên.
        - prefix : tên folder muốn download trong character.
            Ví dụ: trong bucket reunion là: reunion/manh/avatars và muốn tải toàn bộ avatars thì prefix="avatars"
        - local_folder_path : đường dẫn folder muốn tải về local.
            Vị trí tải folder, mặc định là ".", tức là ở vị trí đang đứng.
        Return
        ---
        - Folder tải về có dạng {character_id}/{prefix}/{các_folder_bên_trong}.
        """
        for object in self.list_items_in_bucket(bucket_name, f'{des_folder_name}/{prefix}'):
            object_name = object.object_name
            print(object_name)
            local_file_name = f'{local_folder_path}/{object_name}'
            self.client.fget_object(
                bucket_name=bucket_name, 
                object_name=object_name, 
                file_path=local_file_name)
        return local_folder_path
    
