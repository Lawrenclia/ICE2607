import os
import time
import numpy as np
import torch
import torchvision.transforms as transforms
from torchvision.datasets.folder import default_loader
from PIL import Image
import matplotlib.pyplot as plt 
from torchvision.models import resnet50, ResNet50_Weights 


LIBRARY_DIR = 'image_library'       # 图像库
QUERY_DIR = 'query'                 # 待搜索图片
OUTPUT_DIR = 'output'               # 保存结果图
TOP_K = 5                           # Top 5 结果 


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

print('Load model: ResNet50')

weights = ResNet50_Weights.DEFAULT 
model = resnet50(weights=weights)

model.eval()  
model.to(device)

# 图像预处理
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
trans = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    normalize,
])

# 图片加载和显示预处理
display_transform = transforms.Compose([
    transforms.Resize((224, 224)), 
])

# 特征提取函数
def features(x):
    x = model.conv1(x)
    x = model.bn1(x)
    x = model.relu(x)
    x = model.maxpool(x)
    x = model.layer1(x)
    x = model.layer2(x)
    x = model.layer3(x)
    x = model.layer4(x)
    x = model.avgpool(x)
    return x

def extract_feature(image_path, model, preprocess):
    try:
        img = default_loader(image_path)
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None
    input_tensor = preprocess(img)
    input_batch = input_tensor.unsqueeze(0).to(device)
    with torch.no_grad():
        feature = features(input_batch)
    feature_np = feature.cpu().detach().numpy().flatten()
    feature_norm = feature_np / np.linalg.norm(feature_np)
    return feature_norm


# 构建特征数据库
def build_database(library_dir, model, preprocess):
    db_features = []
    db_image_paths = []
    print(f"\nBuilding feature database for '{library_dir}' (scanning all sub-directories)...")
    for dirpath, dirnames, filenames in os.walk(library_dir):
        image_files = [f for f in filenames if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        for filename in sorted(image_files):
            image_path = os.path.join(dirpath, filename)
            feature = extract_feature(image_path, model, preprocess)
            if feature is not None:
                db_features.append(feature)
                db_image_paths.append(image_path)
    
    total_features = len(db_features)
    print(f"Database build complete. Total features extracted: {total_features}")
    return np.array(db_features), db_image_paths


# 绘图函数
def plot_results(query_path, query_filename, matched_paths, scores, top_k):
    if not matched_paths:
        print("No matched images to plot.")
        return

    fig, axes = plt.subplots(1, 6, figsize=(24, 5)) 
    fig.suptitle(f"Query: {os.path.basename(query_path)} - Top {top_k} Matches", fontsize=16)

    # 显示查询图片
    try:
        img_query = Image.open(query_path).convert('RGB')
        img_query = display_transform(img_query) 
        
        axes[0].imshow(img_query)
        axes[0].set_title(f"Query\n({os.path.basename(query_path)})", fontsize=10)
        axes[0].axis('off')
    except Exception as e:
        axes[0].set_title(f"Error loading query\n{os.path.basename(query_path)}\n({e})", fontsize=8)
        axes[0].axis('off')

    # 显示匹配图片
    for i in range(top_k):
        if i < len(matched_paths):
            ax_current = axes[i+1] 
            
            try:
                img_match = Image.open(matched_paths[i]).convert('RGB')
                img_match = display_transform(img_match) 
                ax_current.imshow(img_match)
                
                relative_path = os.path.relpath(matched_paths[i], LIBRARY_DIR)
                ax_current.set_title(f"Rank {i+1} (Score: {scores[i]:.4f})\n({relative_path})", fontsize=10)
                
                ax_current.axis('off')
            except Exception as e:
                ax_current.set_title(f"Error loading match {i+1}\n({os.path.basename(matched_paths[i])})\n({e})", fontsize=8)
                ax_current.axis('off')
        else:
            axes[i+1].axis('off')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) 

    base_filename = os.path.splitext(query_filename)[0]
    output_filename = f"{base_filename}_result.png"
    save_path = os.path.join(OUTPUT_DIR, output_filename)
    
    plt.savefig(save_path)
    print(f"Result plot saved to: {save_path}")
    plt.close(fig)

if __name__ == "__main__":
    
    # 构建特征数据库
    db_features, db_paths = build_database(LIBRARY_DIR, model, trans)
    
    if db_features.shape[0] == 0:
        print(f"Error: No features were extracted. Is '{LIBRARY_DIR}' empty or incorrect?")
    
    elif not os.path.exists(QUERY_DIR):
        print(f"Error: Query directory not found at '{QUERY_DIR}'")
        
    else:
        if not os.path.exists(OUTPUT_DIR):
            os.makedirs(OUTPUT_DIR)
            print(f"Created output directory at: {OUTPUT_DIR}")

        # 遍历 query 文件夹中的所有图片
        query_image_files = sorted([f for f in os.listdir(QUERY_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])

        if not query_image_files:
            print(f"Error: No query images found in '{QUERY_DIR}'")
        
        for query_filename in query_image_files:
            query_image_path = os.path.join(QUERY_DIR, query_filename)
            
            print(f"\n======== Processing Query Image: {query_image_path} ========")
            
            # 提取查询图片的特征
            query_feature = extract_feature(query_image_path, model, trans)
            
            if query_feature is None:
                print(f"Skipping {query_image_path} due to extraction error.")
                continue 
            
            # 计算相似度
            scores = np.dot(db_features, query_feature)
            
            # 排序并报告
            top_indices = np.argsort(scores)[::-1][:TOP_K]
            
            matched_image_paths = [db_paths[idx] for idx in top_indices]
            matched_scores = [scores[idx] for idx in top_indices]

            print(f"\n--- Top {TOP_K} Search Results for '{query_image_path}' ---")
            print("Similarity Method: Cosine Similarity (Vector Angle)")
            
            for i in range(len(matched_image_paths)):
                print(f"\nRank {i+1}:")
                print(f"  Image: {matched_image_paths[i]}")
                print(f"  Score: {matched_scores[i]:.6f}")
            
            # 绘制并保存结果图
            plot_results(query_image_path, query_filename, matched_image_paths, matched_scores, TOP_K)

            print(f"======== Finished Processing: {query_image_path} ========\n")