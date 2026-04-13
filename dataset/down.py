import os
import requests
import zipfile

# 这是一个更直接的 YOLO 格式 VOC 2007 链接
url = "https://github.com/ultralytics/yolov5/releases/download/v1.0/voc2007.zip"

def download():
    save_path = "voc2007.zip"
    # 如果文件已经存在且太小，先删掉重下
    if os.path.exists(save_path) and os.path.getsize(save_path) < 1024 * 1024:
        os.remove(save_path)

    if not os.path.exists(save_path):
        print("正在从 GitHub 镜像下载 VOC2007 (约450MB)...")
        # 必须加上 allow_redirects=True
        r = requests.get(url, stream=True, allow_redirects=True)
        total_size = int(r.headers.get('content-length', 0))
        
        with open(save_path, "wb") as f:
            downloaded = 0
            for chunk in r.iter_content(chunk_size=1024*1024):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    print(f"进度: {downloaded/1024/1024:.2f}MB / {total_size/1024/1024:.2f}MB", end='\r')
        print("\n下载完成！")
    
    print("正在解压...")
    try:
        with zipfile.ZipFile(save_path, 'r') as z:
            z.extractall(".")
        print("解压成功！")
    except Exception as e:
        print(f"解压失败: {e}。说明下载的文件依然损坏。")

if __name__ == "__main__":
    os.makedirs("images", exist_ok=True)
    os.makedirs("labels", exist_ok=True)
    download()
