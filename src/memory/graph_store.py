"""FalkorDB graph store for entity-relationship tracking."""

from falkordb import FalkorDB
from datetime import datetime
from typing import Any
import uuid

from src.config import settings


class GraphStore:
    """FalkorDB-backed graph store for entities and relationships."""
    
    def __init__(
        self,
        host: str | None = None,
        port: int | None = None,
        graph_name: str = "lcr_knowledge"
    ):
        """
        Initialize the graph store.
        
        Args:
            host: FalkorDB host
            port: FalkorDB port
            graph_name: Name of the graph to use
        """
        self.host = host or settings.falkordb_host
        self.port = port or settings.falkordb_port
        self.graph_name = graph_name
        
        # Connect to FalkorDB
        self.db = FalkorDB(host=self.host, port=self.port)
        self.graph = self.db.select_graph(self.graph_name)
    
    def add_person(
        self,
        name: str,
        relationship_to_user: str = "unknown",
        attributes: dict | None = None
    ) -> str:
        """
        Add or update a Person node.
        
        Args:
            name: Person's name
            relationship_to_user: "self" | "partner" | "friend" | "coworker" | "family" | "unknown"
            attributes: Additional attributes
            
        Returns:
            Person node ID
        """
        person_id = str(uuid.uuid4())
        now = datetime.now().isoformat()
        
        # Merge (create or update)
        query = """
        MERGE (p:Person {name: $name})
        ON CREATE SET 
            p.id = $id,
            p.relationship_to_user = $rel,
            p.first_mentioned = $now,
            p.last_mentioned = $now
        ON MATCH SET
            p.last_mentioned = $now
        RETURN p.id as id
        """
        
        result = self.graph.query(
            query,
            {
                "name": name,
                "id": person_id,
                "rel": relationship_to_user,
                "now": now
            }
        )
        
        if result.result_set:
            return result.result_set[0][0]
        return person_id
    
    def add_entity(
        self,
        name: str,
        category: str,
        attributes: dict | None = None
    ) -> str:
        """
        Add or update an Entity node.
        
        Args:
            name: Entity name
            category: "technology" | "place" | "organization" | "event" | "concept"
            attributes: Additional attributes (stored as JSON string)
            
        Returns:
            Entity node ID
        """
        import json
        
        entity_id = str(uuid.uuid4())
        now = datetime.now().isoformat()
        attrs_json = json.dumps(attributes or {})
        
        query = """
        MERGE (e:Entity {name: $name})
        ON CREATE SET
            e.id = $id,
            e.category = $category,
            e.attributes = $attrs,
            e.first_mentioned = $now,
            e.last_mentioned = $now,
            e.still_valid = true
        ON MATCH SET
            e.last_mentioned = $now,
            e.category = $category,
            e.attributes = $attrs
        RETURN e.id as id
        """
        
        result = self.graph.query(
            query,
            {
                "name": name,
                "id": entity_id,
                "category": category,
                "attrs": attrs_json,
                "now": now
            }
        )
        
        if result.result_set:
            return result.result_set[0][0]
        return entity_id
    
    def add_relationship(
        self,
        subject: str,
        predicate: str,
        object_name: str,
        metadata: dict | None = None
    ) -> bool:
        """
        Add a relationship between entities.
        
        Args:
            subject: Subject entity/person name
            predicate: Relationship type (e.g., "WORKS_AT", "KNOWS", "OWNS")
            object_name: Object entity/person name
            metadata: Additional relationship properties
            
        Returns:
            True if successful
        """
        now = datetime.now().isoformat()
        metadata = metadata or {}
        metadata["created_at"] = now
        metadata["still_valid"] = True
        
        # Build properties string for Cypher
        props_str = ", ".join(f"r.{k} = ${k}" for k in metadata.keys())
        
        query = f"""
        MATCH (s {{name: $subject}})
        MATCH (o {{name: $object}})
        MERGE (s)-[r:{predicate}]->(o)
        SET {props_str}
        RETURN r
        """
        
        params = {
            "subject": subject,
            "object": object_name,
            **metadata
        }
        
        try:
            self.graph.query(query, params)
            return True
        except Exception as e:
            print(f"[GraphStore] Error adding relationship: {e}")
            return False
    
    def check_contradictions(
        self,
        subject: str,
        predicate: str,
        new_object: str
    ) -> list[dict]:
        """
        Check if a new relationship contradicts existing ones.
        
        Args:
            subject: Subject name
            predicate: Relationship type
            new_object: New object name
            
        Returns:
            List of contradictions found
        """
        query = f"""
        MATCH (s {{name: $subject}})-[r:{predicate}]->(o)
        WHERE r.still_valid = true AND o.name <> $new_object
        RETURN o.name as existing_object, r
        """
        
        result = self.graph.query(
            query,
            {"subject": subject, "new_object": new_object}
        )
        
        contradictions = []
        if result.result_set:
            for row in result.result_set:
                contradictions.append({
                    "existing_object": row[0],
                    "new_object": new_object,
                    "predicate": predicate
                })
        
        return contradictions
    
    def supersede_relationship(
        self,
        subject: str,
        predicate: str,
        old_object: str
    ) -> bool:
        """
        Mark an old relationship as superseded (not deleted).
        
        Args:
            subject: Subject name
            predicate: Relationship type
            old_object: Old object name to supersede
            
        Returns:
            True if successful
        """
        now = datetime.now().isoformat()
        
        query = f"""
        MATCH (s {{name: $subject}})-[r:{predicate}]->(o {{name: $old_object}})
        SET r.still_valid = false,
            r.superseded_at = $now
        RETURN r
        """
        
        try:
            self.graph.query(query, {"subject": subject, "old_object": old_object, "now": now})
            return True
        except Exception as e:
            print(f"[GraphStore] Error superseding relationship: {e}")
            return False
    
    def query_entity_facts(self, entity_name: str, limit: int = 5) -> list[dict]:
        """
        Get facts about an entity.
        
        Args:
            entity_name: Entity or person name
            limit: Maximum facts to return
            
        Returns:
            List of facts
        """
        query = """
        MATCH (n {name: $name})-[r]-(related)
        WHERE r.still_valid = true OR NOT EXISTS(r.still_valid)
        RETURN type(r) as relationship, related.name as related_entity, properties(r) as metadata
        LIMIT $limit
        """
        
        result = self.graph.query(query, {"name": entity_name, "limit": limit})
        
        facts = []
        if result.result_set:
            for row in result.result_set:
                facts.append({
                    "relationship": row[0],
                    "related_entity": row[1],
                    "metadata": row[2]
                })
        
        return facts
    
    def get_user_node(self) -> dict | None:
        """Get or create the User person node."""
        query = """
        MERGE (u:Person:User {relationship_to_user: "self"})
        ON CREATE SET u.id = $id, u.name = "User", u.first_mentioned = $now
        RETURN u.id as id, u.name as name
        """
        
        result = self.graph.query(
            query,
            {"id": str(uuid.uuid4()), "now": datetime.now().isoformat()}
        )
        
        if result.result_set:
            return {"id": result.result_set[0][0], "name": result.result_set[0][1]}
        return None
    
    def close(self):
        """Close the connection."""
        # FalkorDB client doesn't require explicit closing
        pass
