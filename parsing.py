import fitz  # PyMuPDF
import json
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from collections import Counter
import os
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import re
import pdfplumber
from collections import defaultdict
from pathlib import Path
from collections import defaultdict, deque
import time

INPUT_DIR="/app/input"
OUTPUT_DIR="/app/output"

def is_effectively_1x1(table):
    """
    Returns True if, once you discard any rows that are completely empty (all ''), 
    you end up with exactly one row, and in that row there is exactly one non‐empty cell.
    """
    # 1. Filter out rows that are entirely empty
    non_empty_rows = [
        row for row in table 
        if any((cell or "").strip() for cell in row)
    ]
    
    # 2. If there isn’t exactly one non‐empty row, it can’t be 1×1
    if len(non_empty_rows) != 1:
        return False
    
    # 3. In that single remaining row, count its non‐empty cells
    non_empty_cells = [
        cell for cell in non_empty_rows[0] 
        if (cell or "").strip()
    ]
    
    # 4. It’s 1×1 if there’s exactly one non‐empty cell
    return len(non_empty_cells) == 1

def is_non_heading(full_line_text: str) -> bool:
    text = full_line_text.strip()

    # Condition 1: Starts with bullet or bracket
    if re.match(r"^[-*•\u2022\u25CF\u25E6\u2043\u2219\(\[\{]", text) or re.search(r"\bet al\.", text, re.IGNORECASE):
        return True

    # Condition 2: All characters are non-alphabetic
    if not re.search(r"[a-zA-Z]", text):
        return True

    # Condition 3: Dot between two letters or number and letter
    if re.search(r"[a-zA-Z0-9]\.[a-zA-Z]", text):
        return True

    return False

def detect_table(pdf_path):
    with pdfplumber.open(pdf_path) as pdf:
        bboxes = []
        ttexts = []
        for page_num, page in enumerate(pdf.pages, start=1):
            tables = page.find_tables()
            if tables:
                for table in tables:
                    bbox = table.bbox
                    # Extract table text as a list of rows (each row is a list of cell texts)
                    table_data = table.extract()
                    bboxes.append({page_num:bbox})
                    ttexts.append(table_data)
                    # tables_info.append({
                        # "page": page_num,
                        # "bbox": bbox,
                        # "data": table_data
                    # })
        return bboxes, ttexts

def detect_image(pdf_path):
    
    with pdfplumber.open(pdf_path) as pdf:
        bboxes=[]
        for page_num, page in enumerate(pdf.pages, start=1):
            if page.images:
                for img in page.images:
                    bbox = [img['x0'], img['top'], img['x1'], img['bottom']]
                    bboxes.append({page_num:bbox})
            
        return bboxes

                            
def extract_text_features(pdf_path):
    
    doc = fitz.open(pdf_path)

    text_lines_info = []
    all_features_info = []
    word_counts = []

    bboxes_table, ttexts_table=detect_table(pdf_path)
        
    table_rows={}
    for page_num, page in enumerate(doc):
        page_lines = []

        # Step 1: Collect all lines across all blocks
        blocks = page.get_text("dict").get("blocks", [])
        for b in blocks:    
            if "lines" in b:
                for l in b["lines"]:
                    if l["spans"]:
                        spans = [s for s in l["spans"] if s["text"].strip() != ""]
                        l["spans"] = spans
                        font_sizes = [s["size"] for s in spans]
                        font_colors = [s["color"] for s in spans]
                        fonts = [s["font"] for s in spans]
                        full_line_text = "".join([s["text"] for s in spans]).strip()
                        if full_line_text:      
                            is_body = 0
                            if is_non_heading(full_line_text):
                                is_body = 1
                                
                            is_bold = int(all("bold" in s["font"].lower() for s in spans))
                            is_italic = int(all("italic" in s["font"].lower() for s in spans))
                                        
                            for bbox, ttext in zip(bboxes_table, ttexts_table):
                                if page_num + 1 in bbox:
                                    
                                    table_bbox = bbox[page_num + 1]
                                    x0_t, y0_t, x1_t, y1_t = table_bbox
                                    x0_l, y0_l, x1_l, y1_l = l["bbox"]
                                    
                                    tolerance = 10  # you can adjust this as needed
                                    if (
                                        (x0_l >= x0_t - tolerance or x0_l >= x0_t + tolerance) and
                                        (y0_l >= y0_t - tolerance or y0_l >= y0_t + tolerance) and
                                        (x1_l <= x1_t + tolerance or x1_l <= x1_t - tolerance) and
                                        (y1_l <= y1_t + tolerance or y1_l <= y1_t - tolerance)
                                    ):
                                        if not is_effectively_1x1(ttext):
                                            is_body = 1
                            word_count = len(full_line_text.split())
                            if is_body == 0:
                                word_counts.append(word_count)

                            mode_font_size = max(set(font_sizes), key=font_sizes.count) if font_sizes else None
                            mode_color = max(set(font_colors), key=font_colors.count) if font_colors else None
                            mode_font = max(set(fonts), key=fonts.count) if fonts else None
                                    
                            page_lines.append({
                                "text": full_line_text,
                                "page": page_num + 1,
                                "font": mode_font,
                                "font_size": mode_font_size,
                                "color": mode_color,
                                "bbox": l["bbox"],
                                "bold": is_bold,
                                "italic": is_italic,
                                "word_count": word_count,
                                "is_body": is_body
                            })
        # Step 2: Sort all lines vertically (top to bottom)
        page_lines.sort(key=lambda x: x["bbox"][1])  # bbox[1] = top Y

        # Step 3: Compute space_above and space_below
        for i, line in enumerate(page_lines):
            current_top = line["bbox"][1]
            current_bottom = line["bbox"][3]

            if i > 0:
                prev_bottom = page_lines[i - 1]["bbox"][3]
                space_above = max(0, current_top - prev_bottom)
            else:
                space_above = 0.0

            if i + 1 < len(page_lines):
                next_top = page_lines[i + 1]["bbox"][1]
                space_below = max(0, next_top - current_bottom)
            else:
                space_below = 0.0

            # Add to the line's info (all_features_info or inline)
            line["space_above"] = space_above
            line["space_below"] = space_below
            
            all_features_info.append(line)

            # Keep clusterable text info
            text_lines_info.append({
                "text": line["text"],
                "page": line["page"]
            })
            
    avg_word_count = np.mean(word_counts) if word_counts else 0
    
    for line in all_features_info:
        line["avg_word_count"] = avg_word_count
        
    # Heuristic: heading if both space_above and space_below are large
    space_aboves = [line["space_above"] for line in all_features_info if line["space_above"] > 0]
    space_belows = [line["space_below"] for line in all_features_info if line["space_below"] > 0]

    if space_aboves and space_belows:
        avg_above = np.mean(space_aboves)
        avg_below = np.mean(space_belows)
        std_above = np.std(space_aboves)
        std_below = np.std(space_belows)

        above_threshold = avg_above + 1.0 * std_above
        below_threshold = avg_below + 1.0 * std_below

        for line in all_features_info:
            above = line.get("space_above", 0)
            below = line.get("space_below", 0)

            # Heading detection: both spaces significantly above normal
            if above > above_threshold or (below > below_threshold or above==0):
                line["is_heading_spacing"] = True
            else:
                line["is_heading_spacing"] = False        
    
    # Save extracted features for debugging
    bboxes_image=detect_image(pdf_path)
    
    for bbox in bboxes_image:
        page_num = list(bbox.keys())[0]
        for page_num in bbox:
            img_bbox = bbox[page_num]
            all_features_info.append({
                "text": "Image",
                "page": page_num,
                "bbox": img_bbox,
                "is_body":1
                })
    pdf_name = os.path.basename(pdf_path)
    
    threshold_value = 1
    font_sizes = [line["font_size"] for line in all_features_info if line["text"]!="Image"]
    font_size_counts = Counter(font_sizes)
    mode_font_size = font_size_counts.most_common(1)[0][0]
    color_counts = Counter(line["color"] for line in all_features_info if line["text"]!="Image")
    mode_color = color_counts.most_common(1)[0][0]
    
    for i, line in enumerate(all_features_info):
        if  line["text"]!="Image" and line['font_size']>mode_font_size+threshold_value:
            continue
        if line["text"]!="Image" and line['word_count'] > line["avg_word_count"]:
            line["is_body"] = 1

    for i, line in enumerate(all_features_info):
        
        if line["text"]!="Image":
            if line["font_size"] < mode_font_size:
                line["is_body"] = 1
            elif line["font_size"] == mode_font_size:
                if not bool(line["bold"]):
                    if line["color"]== mode_color:
                        is_body=1
                        
    
    # json_filename = f"extracted_text_features_{pdf_name}.json"
    # with open(json_filename, "w", encoding="utf-8") as f_out:
    #     json.dump(all_features_info, f_out, indent=2, ensure_ascii=False)
    return all_features_info,text_lines_info,doc

def DBSCAN_run(actual_all_features_info):
    features = []
            
    all_features_info=[line for line in actual_all_features_info if line["is_body"]==0]

    for line in all_features_info:
        features.append([
            line["font_size"],
            line["bold"],
        ])
        
    if not features:
        return {}, [], [], []

    features_np = np.array(features)

    # Normalize features
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features_np)


    # Run DBSCAN
    dbscan = DBSCAN(eps=0.2, min_samples=1)
    clusters = dbscan.fit_predict(scaled_features)
        
    # Assign cluster to each line
    for i, line in enumerate(all_features_info):
        line["cluster"] = clusters[i]
        
    body_clusters=[]
    min_font_size = float('inf')

    for line in all_features_info:
        min_font_size = min(min_font_size, line["font_size"])
    
    font_sizes = [line["font_size"] for line in all_features_info]
    font_size_counts = Counter(font_sizes)
    mode_font_size = font_size_counts.most_common(1)[0][0]

    color_counts = Counter(line["color"] for line in all_features_info)
    mode_color = color_counts.most_common(1)[0][0]
        
    # FINAL CHECK: Heading cluster lines that *don’t look like headings* → mark as noise
    for i, line in enumerate(all_features_info):
        cluster = line["cluster"]
        if cluster != -1 and cluster not in body_clusters:
            is_bold = bool(line["bold"])
            font_size = line["font_size"]
            color = line['color']

            # If the line in heading cluster *doesn't look like a heading*, mark as noise
            if (not is_bold and color == mode_color and (font_size <= mode_font_size)):
                
                all_features_info[i]["cluster"] = -1 
    
    # Final organization of lines by cluster
    updated_clusters = [line["cluster"] for line in all_features_info]
    clustered_text = {}
    for i, label in enumerate(updated_clusters):
        clustered_text.setdefault(label, []).append({
            "text": all_features_info[i]["text"],
            "page": all_features_info[i]["page"],
            "bbox": all_features_info[i]["bbox"],
            "font_size": features[i][0],
        })
        
    return clustered_text,body_clusters,features,all_features_info

def detect_nonbordered_tables(clustered_text, body_clusters, threshold=1):
    visited = set()
    to_remove = set()
    
    for label, cluster_lines in clustered_text.items():
        if label not in body_clusters and label != -1:
            for line in cluster_lines:
                line_id = (line["page"], tuple(line["bbox"]))
                if line_id in visited:
                    continue

                visited.add(line_id)

                for label_other, cluster_lines_other in clustered_text.items():
                        for line_other in cluster_lines_other:
                            other_line_id = (line_other["page"], tuple(line_other["bbox"]))
                            if line_other is not line and line_other["page"] == line["page"]:
                                bbox_i = line["bbox"]
                                bbox_j = line_other["bbox"]

                                x0_i, y0_i, x1_i, y1_i = bbox_i
                                x0_j, y0_j, x1_j, y1_j = bbox_j

                                vertical_proximity = (
                                    abs(y0_i - y0_j) <= threshold and
                                    abs(y1_i - y1_j) <= threshold
                                )

                                if vertical_proximity:
                                    to_remove.add(line_id)
                                    to_remove.add(other_line_id)
    cleaned_clustered_text = defaultdict(list)
    for label, lines in clustered_text.items():
        cleaned_lines = [
            line for line in lines
            if (line["page"], tuple(line["bbox"])) not in to_remove
        ]
        if cleaned_lines:
            cleaned_clustered_text[label] = cleaned_lines
    
    return dict(cleaned_clustered_text)

def merge_clustered_lines(clustered_text, body_clusters, threshold_v=15, threshold_h=1):
    # 1) Gather all eligible lines into one list and track their cluster
    eligible = []
    line_to_cluster = {}  # maps line index in `eligible` to original cluster
    for label, lines in clustered_text.items():
        if label in body_clusters or label == -1:
            continue
        for line in lines:
            line_to_cluster[len(eligible)] = label
            eligible.append(line)

    n = len(eligible)

    # 2) Build adjacency for every pair on same page
    adj = [set() for _ in range(n)]
    for i in range(n):
        xi0, yi0, xi1, yi1 = eligible[i]['bbox']
        pi = eligible[i]['page']
        for j in range(i+1, n):
            if eligible[j]['page'] != pi:
                continue
            xj0, yj0, xj1, yj1 = eligible[j]['bbox']
            # vertical proximity
            vert = abs(yi0 - yj0) <= threshold_v and abs(yi1 - yj1) <= threshold_v
            # horizontal overlap
            horiz = (
                (xi0 + threshold_h >= xj0 and xi1 - threshold_h <= xj1) or
                (xj0 + threshold_h >= xi0 and xj1 - threshold_h <= xi1)
            )
            if vert and horiz:
                adj[i].add(j)
                adj[j].add(i)

    # 3) Extract connected components via BFS
    visited = [False] * n
    updated_clusters = defaultdict(list)

    for i in range(n):
        if visited[i]:
            continue
        # BFS to collect component
        queue = deque([i])
        comp_idxs = {i}
        visited[i] = True

        while queue:
            u = queue.popleft()
            for v in adj[u]:
                if not visited[v]:
                    visited[v] = True
                    comp_idxs.add(v)
                    queue.append(v)

        comp = sorted(
            (idx for idx in comp_idxs),
            key=lambda idx: (eligible[idx]['bbox'][1], eligible[idx]['bbox'][0])
        )

        if len(comp) > 1:
            # Preserve the cluster of the 'main' line (first in sorted list)
            main_idx = comp[0]
            main_cluster = line_to_cluster[main_idx]

            merged = dict(eligible[main_idx])  # start from the main line
            merged['text'] = ' '.join(eligible[idx]['text'] for idx in comp)
            x0s, y0s = zip(*(eligible[idx]['bbox'][0:2] for idx in comp))
            x1s, y1s = zip(*(eligible[idx]['bbox'][2:4] for idx in comp))
            merged['bbox'] = [min(x0s), min(y0s), max(x1s), max(y1s)]

            updated_clusters[main_cluster].append(merged)

            # Remove all other lines from their old clusters
            for idx in comp[1:]:
                old_cluster = line_to_cluster[idx]
                # We skip adding these lines anywhere since they're merged
                pass
        else:
            # No merging happened; keep as-is in its original cluster
            idx = next(iter(comp))
            cluster = line_to_cluster[idx]
            updated_clusters[cluster].append(eligible[idx])

    return dict(updated_clusters)

def identify_headings(actual_clustered_text,body_clusters,all_features_info,doc,features):
    
    clean_clustered_text = detect_nonbordered_tables(actual_clustered_text, body_clusters)
    clustered_text= merge_clustered_lines(clean_clustered_text,body_clusters)
    # Identify headings (excluding body clusters and noise)
    heading_clusters = {}
    
    for label, lines in clustered_text.items():
        if label not in body_clusters and label != -1:
            avg_font_size = np.mean([line["font_size"] for line in lines])
            heading_clusters[label] = avg_font_size
    
    # Assign heading levels based on font size ranking (largest font = H1)
    sorted_heading_clusters = sorted(heading_clusters.items(), key=lambda item: item[1], reverse=True)
    level_map = {label: i + 1 for i, (label, _) in enumerate(sorted_heading_clusters)}
    outline = {"title": "", "headings": []}

    if sorted_heading_clusters:
        top_cluster_label = sorted_heading_clusters[0][0]  # label with largest average font
        top_cluster_lines = clustered_text.get(top_cluster_label, [])
        
        # Sort by page number and vertical position (bbox[1] = top)
        top_cluster_lines.sort(key=lambda x: (x["page"], x["bbox"][1]))
        
        if top_cluster_lines:
            title_line = top_cluster_lines[0]
            outline["title"] = title_line["text"]
            title_candidates = [title_line["text"]]  
        else:
            title_candidates = []
    else:
        title_candidates = []

    # Build heading list
    for label, lines in clustered_text.items():
        if label in level_map:
            level = level_map[label]
            for line_data in lines:
                if line_data["text"]!=outline["title"] and line_data["text"] not in title_candidates:
                    outline["headings"].append({
                    "text": line_data["text"],
                    "level": f"H{level}",
                    "page": line_data["page"],
                    "bbox": line_data["bbox"]
                })

    # Sort headings by appearance (page first, then level)
    outline["headings"].sort(key=lambda x: (x["page"], x["bbox"][1]))
    
    doc.close()
    return outline

def run_parser(pdf_path):
    """
    Run the parser on the PDF file to extract headings and their positions.
    """
    all_features_info, text_lines_info, doc = extract_text_features(pdf_path)
    clustered_text, body_clusters, features, all_features_info = DBSCAN_run(all_features_info)
    
    # Identify headings
    document_outline = identify_headings(clustered_text, body_clusters, all_features_info, doc, features)

    # Remove 'bbox' from each heading
    for heading in document_outline.get("headings", []):
        heading.pop("bbox", None)

    return document_outline

if __name__ == '__main__':
    doc_outline = run_parser("/app/input/sample.pdf")
    # os.makedirs(OUTPUT_DIR, exist_ok=True)
    # total_time = 0
    # total_files = 0
    # for pdf_file in Path(INPUT_DIR).glob("*.pdf"):
    #     pdf_path = str(pdf_file)
    #     pdf_name = pdf_file.name

    #     print(f"Processing: {pdf_name}")
            
    #     try:
    #         start_time = time.time()
    #         actual_all_features_info, text_lines_info, doc = extract_text_features(pdf_path)
    #         clustered_text, body_clusters, features, all_features_info = DBSCAN_run(actual_all_features_info)
    #         document_outline = identify_headings(clustered_text, body_clusters, all_features_info, doc, features)

    #         for heading in document_outline.get("headings", []):
    #             heading.pop("bbox", None)

    #         output_path = os.path.join(OUTPUT_DIR, f"{pdf_name[:-4]}.json")
    #         with open(output_path, "w", encoding="utf-8") as f_out:
    #             json.dump(document_outline, f_out, indent=2, ensure_ascii=False)
            
    #         print(f"✓ Saved: {output_path}")
    #         end_time = time.time()
    #         print(f"Processing time for {pdf_name}: {end_time - start_time:.2f} seconds")
    #         total_time += end_time - start_time
    #         total_files += 1

    #     except Exception as e:
    #         print(f"✗ Error processing {pdf_name}: {e}")
    
    # print("Total processing time for {} successful files: {:.2f} seconds".format(total_files, total_time))
    # print("Average processing time per file: {:.2f} seconds".format(total_time / total_files if total_files > 0 else 0))
