# scripts/convert_unsplash_lite.py
import os
import pandas as pd
import requests
from tqdm import tqdm
import concurrent.futures

# 设置文件路径
photos_file = r"C:\Users\zxy09\Downloads\unsplash-research-dataset-lite-latest\photos.tsv000"
output_dir = "raw_images"
limit = 1000  # 限制下载图片数量

# 创建输出目录
os.makedirs(output_dir, exist_ok=True)

# 加载数据集
photos_df = pd.read_csv(photos_file, sep="\t")
print(f"加载了 {len(photos_df)} 张图片的信息")

# 获取图片URL (使用small或regular尺寸以加快下载)
urls = photos_df['photo_image_url'].tolist()[:limit]

# 下载函数
def download_image(args):
    url, idx = args
    try:
        # 修改URL获取不同尺寸
        # 原始URL格式: https://images.unsplash.com/{photo_id}?...
        # 我们修改为: https://images.unsplash.com/{photo_id}?w=640&fit=crop
        modified_url = url.split("?")[0] + "?w=640&q=80&fit=crop"
        
        save_path = os.path.join(output_dir, f"unsplash_{idx:05d}.jpg")
        response = requests.get(modified_url, timeout=10)
        if response.status_code == 200:
            with open(save_path, 'wb') as f:
                f.write(response.content)
            return True
    except Exception as e:
        print(f"下载图片 {idx} 失败: {e}")
    return False

# 并行下载图片
print(f"开始下载 {len(urls)} 张图片...")
with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
    results = list(tqdm(executor.map(download_image, 
                                    [(url, i) for i, url in enumerate(urls)]), 
                        total=len(urls)))

# 打印结果
success_count = sum(1 for r in results if r)
print(f"下载完成！成功: {success_count}/{len(urls)}")