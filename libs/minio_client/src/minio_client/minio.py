from __future__ import annotations

import random
from base import BaseModel, BaseService
from .settings import MinioSettings
from minio import Minio
from logger import get_logger
from datetime import datetime, timedelta

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
        return Minio(endpoint = self.setting.endpoint,
                    access_key = self.setting.access_key,
                    secret_key = self.setting.secret_key,
                    secure = self.setting.secure,
        )
    
    # Tạo bucket mới
    def make_bucket(self, bucket_name):
        if not self.client.bucket_exists(bucket_name):
            self.client.make_bucket(bucket_name)

    # Lấy ra đường dẫn URL để tải object về
    def presigned_get_object(self, bucket_name, object_name):
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
    def list_items_in_bucket(self, bucket_name, folder_name=""):
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
    
    # Thêm mới object dạng stream > đang lỗi
    def put_object(self, bucket_name, file_data, file_name, content_type):
        try:
            datetime_prefix = datetime.now().strftime('%d-%m-%Y_%H-%M-%S')
            
            object_name = f'{datetime_prefix}__{file_name}'
            while self.check_file_name_exists(bucket_name=bucket_name, file_name=object_name):
                random_prefix = random.randint(1, 1000)
                object_name = f'{datetime_prefix}__{random_prefix}__{file_name}'
                
            self.client.put_object(bucket_name=bucket_name,
                                   object_name=object_name,
                                   data=file_data,
                                   content_type=content_type,
                                   length=-1,
                                   part_size=10*1024*1024)
            
            url = self.presigned_get_object(bucket_name=bucket_name,
                                            object_name=object_name)
            data_file = {
                'bucket_name': bucket_name,
                'file_name': object_name,
                'url': url
            }
            
            return data_file
        except Exception as e:
            raise Exception(e)

    # Thêm mới object từ file local, truyền vào:
    # Bucket_name: tên bucket
    # src_file: tên file ở local đang muốn upload
    # des_folder_name: tên folder (thư mục chứa character) muốn tạo trong Minio
    # des_file: tên object (file) muốn lưu trong Minio
    def put_object_from_local_path(self, bucket_name, src_file, des_folder_name, des_file):
        des_path = f'/{des_folder_name}/{des_file}'
        if not self.check_file_name_exists(bucket_name=bucket_name,
                                        file_name=des_path):
            
            self.client.fput_object(bucket_name=bucket_name,
                                    file_path=src_file,
                                    object_name=des_path)
            
            url = self.presigned_get_object(bucket_name=bucket_name,
                                            object_name=des_path)
            
            data_file = {
                'bucket name': bucket_name,
                'src at local': src_file,
                'save in minio': des_path,
                'url': url,
            }      
            return url
        

    def get_object(self, bucket_name, folder_name, file_name, local_file_name):
        filepath = f'/{folder_name}/{file_name}'
        try:
            self.client.fget_object(
                bucket_name=bucket_name,
                object_name=filepath,
                file_path=local_file_name
            )
        except Exception as e:
            logger.exception(e)
        return 'Success'
    
    # Xóa bucket rỗng   
    def remove_bucket(self, bucket_name):
        try:
            self.client.remove_bucket(bucket_name)
        except Exception as e:
            logger.exception(e)
        return 'Success'

    # Xóa folder và tất cả các file trong folder đó
    def remove_folder(self, bucket_name, folder_name):
        try: 
            objects_to_delete = self.client.list_objects(bucket_name, prefix=folder_name, recursive=True)
            for obj in objects_to_delete:
                self.client.remove_object(bucket_name, obj.object_name)
        except Exception as e:
            logger.error(extra={e})
    # Lấy ra bucket version. trả về off nếu không bật tính năng này
    def get_bucket_version(self, bucket_name):
        return self.client.get_bucket_versioning(bucket_name).status

    # Proccess để tạo bukcet + upload file luôn 
    def process(self, inputs: MinioInputs):
        self.make_bucket(inputs.bucket_name)
        self.put_object_from_local_path(
            inputs.bucket_name,
            inputs.src_file,
            inputs.des_folder,
            inputs.des_file,
        )
        return "Successfully process"