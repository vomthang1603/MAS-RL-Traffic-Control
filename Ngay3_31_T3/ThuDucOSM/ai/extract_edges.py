import xml.etree.ElementTree as ET
import random

# ==========================================
# Trích xuất các edge từ thuduc.net.xml
# ==========================================

def parse_net_file(filename="thuduc.net.xml"):
    """Đọc file net.xml và tìm ra các edge kết nối nhau"""
    tree = ET.parse(filename)
    root = tree.getroot()
    
    edges = {}
    junctions = {}
    
    # Lấy tất cả edges
    for edge in root.findall("edge"):
        edge_id = edge.get("id")
        from_node = edge.get("from")
        to_node = edge.get("to")
        
        # Bỏ qua internal edges (bắt đầu bằng :)
        if not edge_id.startswith(":"):
            edges[edge_id] = {"from": from_node, "to": to_node}
    
    # Lấy tất cả junctions
    for junction in root.findall("junction"):
        junc_id = junction.get("id")
        junc_type = junction.get("type")
        inc_lanes = junction.get("incLanes", "").split()
        
        junctions[junc_id] = {
            "type": junc_type,
            "incoming_edges": set()
        }
        
        # Tách edge ID từ lane ID (lane ID format: "edge_id_lane_num")
        for inc_lane in inc_lanes:
            if "_" in inc_lane:
                edge_id = inc_lane.rsplit("_", 1)[0]
                junctions[junc_id]["incoming_edges"].add(edge_id)
    
    return edges, junctions


def find_connected_edges_at_junction(edges, junctions, junction_id):
    """Tìm các edge vào và ra từ một junction"""
    if junction_id not in junctions:
        return [], []
    
    # Edges vào junction
    in_edges = list(junctions[junction_id]["incoming_edges"])
    
    # Edges ra từ junction - tìm edges có from = junction_id
    out_edges = [eid for eid, edata in edges.items() if edata["from"] == junction_id]
    
    return in_edges, out_edges


def find_best_junction():
    """Tìm ngã tư tốt nhất với nhiều kết nối passenger"""
    edges, junctions = parse_net_file()
    
    # Lấy thông tin loại edge
    tree = ET.parse("thuduc.net.xml")
    root = tree.getroot()
    
    edge_types = {}
    for edge_elem in root.findall("edge"):
        edge_id = edge_elem.get("id")
        if not edge_id.startswith(":"):
            lanes = edge_elem.findall("lane")
            if lanes:
                # Lấy allow attribute từ lane đầu tiên
                allow_attr = lanes[0].get("allow", "")
                disallow_attr = lanes[0].get("disallow", "")
                edge_types[edge_id] = {
                    "allow": allow_attr,
                    "disallow": disallow_attr
                }
    
    # Hàm kiểm tra xem edge có thể chạy passenger không
    def is_passenger_allowed(edge_id):
        if edge_id not in edge_types:
            return True  # Default cho phép
        allow = edge_types[edge_id]["allow"]
        disallow = edge_types[edge_id]["disallow"]
        
        # Nếu có allow list, passenger phải có trong đó
        if allow and "passenger" not in allow:
            return False
        # Nếu có disallow list, passenger không được trong đó
        if disallow and "passenger" in disallow:
            return False
        return True
    
    # Tìm junction có nhiều incoming + outgoing edges với passenger allowed
    best_junction = None
    max_connections = 0
    
    for junc_id, junc_data in junctions.items():
        if junc_data["type"] in ["priority", "traffic_light"]:  # Chỉ lấy ngã tư chính
            in_edges = [e for e in junc_data["incoming_edges"] 
                       if not e.startswith(":") and is_passenger_allowed(e)]
            out_edges = [eid for eid, edata in edges.items() 
                        if edata["from"] == junc_id 
                        and not eid.startswith(":") 
                        and is_passenger_allowed(eid)]
            
            total_conn = len(in_edges) + len(out_edges)
            if total_conn > max_connections and len(in_edges) >= 2 and len(out_edges) >= 2:
                max_connections = total_conn
                best_junction = (junc_id, in_edges, out_edges)
    
    if best_junction:
        junc_id, in_edges, out_edges = best_junction
        print(f"\n✅ Tìm thấy ngã tư tốt nhất (passenger): {junc_id}")
        print(f"Edges vào: {in_edges}")
        print(f"Edges ra: {list(out_edges)}")
        print(f"Tổng kết nối: {len(in_edges) + len(out_edges)}")
        
        return junc_id, list(in_edges), list(out_edges)
    
    return None, [], []


def verify_edges_connected(all_edges, start_edges, end_edges):
    """Kiểm tra xem các edge có kết nối với nhau không"""
    print("\n🔍 Kiểm tra kết nối giữa START_EDGES và END_EDGES:")
    
    for i, start_edge in enumerate(start_edges):
        if i < len(end_edges):
            end_edge = end_edges[i]
            start_to = all_edges[start_edge]["to"]
            end_from = all_edges[end_edge]["from"]
            
            is_connected = start_to == end_from
            status = "✓" if is_connected else "✗"
            print(f"  {status} {start_edge} ({start_to}) -> {end_edge} ({end_from})")
    
    return True


if __name__ == "__main__":
    print("="*60)
    print("Phân tích file thuduc.net.xml để trích xuất ngã tư")
    print("="*60)
    
    edges, junctions = parse_net_file()
    print(f"\n📊 Tổng số edges: {len(edges)}")
    print(f"📊 Tổng số junctions: {len(junctions)}")
    
    junc_id, start_edges, end_edges = find_best_junction()
    
    if junc_id:
        verify_edges_connected(edges, start_edges, end_edges)
        
        # Tạo output cho generate_scenario.py
        print("\n" + "="*60)
        print("📝 Code để cập nhật generate_scenario.py:")
        print("="*60)
        print("\nSTART_EDGES = [")
        for edge in start_edges:
            print(f'    "{edge}",  # Edge vào')
        print("]")
        print("\nEND_EDGES = [")
        for i, edge in enumerate(end_edges):
            if i < len(start_edges):
                print(f'    "{edge}",  # Edge ra')
        print("]")
    else:
        print("\n❌ Không tìm thấy ngã tư phù hợp")
