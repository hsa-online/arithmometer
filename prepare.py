"""
Encodes NN models (pth) to base64 and prepares them to uploading.

"""

import base64
import uuid
import zlib

def process_file(file_name_src: str):
    with open(file_name_src, mode="rb") as fi:
        data_src = fi.read()
        value = zlib.crc32(data_src) & 0xffffffff

        tmp = f'{value:x}'
        file_crc = tmp.zfill(8)
        file_id = uuid.uuid4().hex
        file_name_dest = f'{file_id}_{file_crc}.b64'
       
        data_dest = base64.standard_b64encode(data_src)
        with open(file_name_dest, mode="wb") as fo:
            fo.write(data_dest)
            print(f'{file_name_src} -> {file_name_dest}')

def main():
    process_file('2ftrs_20epochs_1batch.pth')
    process_file('2ftrs_20epochs_2batch.pth')
    process_file('2ftrs_20epochs_5batch.pth')

if __name__ == '__main__':
    main()
