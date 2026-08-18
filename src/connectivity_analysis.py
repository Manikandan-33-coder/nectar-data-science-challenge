import pandas as pd
import numpy as np
import networkx as nx
import os

class AssetGraphAnalyzer:
    def __init__(self, df_meta, df_conn):
        self.df_meta = df_meta.copy()
        self.df_conn = df_conn.copy()
        self.graph = nx.DiGraph()
        self._build_graph()
        
    def _build_graph(self):
        # Add nodes with attributes
        for _, row in self.df_meta.iterrows():
            self.graph.add_node(
                row["asset_id"],
                site_id=row["site_id"],
                building_id=row.get("building_id", ""),
                asset_name=row["asset_name"],
                asset_type=row["asset_type"],
                capacity=row.get("capacity", 0.0)
            )
            
        # Add edges from metadata parent_asset_id if present
        for _, row in self.df_meta.iterrows():
            parent = row.get("parent_asset_id")
            if pd.notna(parent) and parent in self.graph:
                self.graph.add_edge(parent, row["asset_id"], connection_type="ParentOf", strength=1.0)
                
        # Add edges from connectivity dataframe
        for _, row in self.df_conn.iterrows():
            src = row["source_asset_id"]
            tgt = row["target_asset_id"]
            if src in self.graph and tgt in self.graph:
                self.graph.add_edge(
                    src, tgt,
                    connection_type=row.get("connection_type", "ConnectedTo"),
                    strength=row.get("relationship_strength", 1.0)
                )
                
    def audit_data_quality(self):
        """
        Perform Data Quality Assessment on asset metadata and connectivity edges.
        """
        findings = {
            "missing_relationships": [],
            "duplicate_connections": [],
            "orphan_assets": [],
            "invalid_mappings": []
        }
        
        # 1. Duplicate connections
        conn_dups = self.df_conn[self.df_conn.duplicated(subset=["source_asset_id", "target_asset_id"], keep=False)]
        for _, row in conn_dups.iterrows():
            findings["duplicate_connections"].append(f"{row['source_asset_id']} -> {row['target_asset_id']}")
            
        # 2. Orphan assets (degree == 0 or no parent & no children)
        all_meta_assets = set(self.df_meta["asset_id"])
        for node in all_meta_assets:
            in_deg = self.graph.in_degree(node)
            out_deg = self.graph.out_degree(node)
            if in_deg == 0 and out_deg == 0:
                findings["orphan_assets"].append(node)
                
        # 3. Invalid parent-child mappings (e.g., Sensor or Meter parenting a Chiller)
        for u, v, data in self.graph.edges(data=True):
            u_type = self.graph.nodes[u].get("asset_type", "")
            v_type = self.graph.nodes[v].get("asset_type", "")
            if u_type in ["Environmental Sensor", "Energy Meter"] and v_type in ["Chiller", "AHU"]:
                findings["invalid_mappings"].append({
                    "source": u, "source_type": u_type,
                    "target": v, "target_type": v_type,
                    "reason": "Low-level sensor/meter cannot parent a high-level Chiller/AHU asset"
                })
                
        # 4. Missing relationships (Chillers without connected pumps or meters)
        chillers = self.df_meta[self.df_meta["asset_type"] == "Chiller"]["asset_id"]
        for chiller in chillers:
            children = list(self.graph.successors(chiller))
            child_types = [self.graph.nodes[c].get("asset_type") for c in children]
            if "Pump" not in child_types:
                findings["missing_relationships"].append(f"Chiller {chiller} has no connected Pump")
                
        return findings

    def simulate_failure_propagation(self, failed_asset_id):
        """
        Identify all downstream assets impacted when a target asset fails using BFS/DFS graph traversal.
        """
        if failed_asset_id not in self.graph:
            return {"failed_asset": failed_asset_id, "impacted_assets": [], "impact_count": 0}
            
        # Downstream nodes are reachable via directed edges
        downstream = list(nx.descendants(self.graph, failed_asset_id))
        
        impacted_details = []
        for node in downstream:
            impacted_details.append({
                "asset_id": node,
                "asset_name": self.graph.nodes[node].get("asset_name", ""),
                "asset_type": self.graph.nodes[node].get("asset_type", ""),
                "distance": nx.shortest_path_length(self.graph, failed_asset_id, node)
            })
            
        return {
            "failed_asset": failed_asset_id,
            "failed_asset_type": self.graph.nodes[failed_asset_id].get("asset_type", ""),
            "impacted_assets": impacted_details,
            "impact_count": len(downstream)
        }

    def query_asset(self, asset_id):
        """
        Query connected, parent, child, and dependency relationships for an asset.
        """
        if asset_id not in self.graph:
            return None
            
        parents = list(self.graph.predecessors(asset_id))
        children = list(self.graph.successors(asset_id))
        
        return {
            "asset_id": asset_id,
            "asset_name": self.graph.nodes[asset_id].get("asset_name", ""),
            "asset_type": self.graph.nodes[asset_id].get("asset_type", ""),
            "parents": parents,
            "children": children,
            "degree": self.graph.degree(asset_id)
        }
