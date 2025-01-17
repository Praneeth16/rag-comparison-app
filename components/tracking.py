from opik import Opik
import uuid

opik = Opik()

def track_interaction(query, rag_type, response=None, track_id=None):
    if track_id is None:
        track_id = str(uuid.uuid4())
        # Track the initial query
        opik.track(
            event_name="rag_query",
            properties={
                "track_id": track_id,
                "rag_type": rag_type,
                "query": query
            }
        )
    else:
        # Track the response
        opik.track(
            event_name="rag_response",
            properties={
                "track_id": track_id,
                "rag_type": rag_type,
                "response": response
            }
        )
    
    return track_id 