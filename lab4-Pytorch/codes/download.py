import os
from bing_image_downloader import downloader

def download_images_task():
    """
    下载图片,建立图像库 (Library)
    """
    # 图像库的主目录
    library_path = 'image_library'
    if not os.path.exists(library_path):
        os.makedirs(library_path)
        print(f"创建目录: {library_path}")
    else:
        print(f"目录 '{library_path}' 已存在。")

    queries = [
        "panda",      
        "cat",         
        "dog",          
        "airplane",     
        "car",          
        "bicycle",     
        "flower",      
        "beach",        
        "building",    
        "tree"         
    ]
    
    # 每个类别下载10张 (10 * 10 = 100张)
    limit_per_query = 10

    print(f"--- 开始下载图片到 '{library_path}' 目录 ---")
    print(f"总共 {len(queries)} 个类别, 每个类别最多 {limit_per_query} 张。")

    for query in queries:
        print(f"正在下载 '{query}' ...")
        downloader.download(
            query,
            limit=limit_per_query,
            output_dir=library_path,
            adult_filter_off=True,
            force_replace=False,
            timeout=60,
            verbose=True
        )
    
    print("--- 所有图片下载完成 ---")
    print(f"请检查 '{library_path}' 文件夹。")

# --- 主程序入口 ---
if __name__ == "__main__":
    download_images_task()