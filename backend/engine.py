import os
import json
import pickle
import faiss
import numpy as np

from typing import List, Dict, Any

from sentence_transformers import SentenceTransformer
from groq import Groq
from dotenv import load_dotenv

load_dotenv()


class SHLConsultantEngine:

    def __init__(self):
        self.groq_client = Groq(
            api_key=os.getenv("GROQ_API_KEY", "")
        )

        
        self.catalog = self._load_catalog()

        
        self.encoder = None

        
        self.index = faiss.read_index(
            "backend/faiss_index.bin"
        )

        
        with open(
            "backend/metadata.pkl",
            "rb"
        ) as f:

            self.metadata = pickle.load(f)

    
    def _load_catalog(self):

        with open(
            "backend/catalog.json",
            "r"
        ) as f:

            return json.load(f)

    
    def get_encoder(self):

        if self.encoder is None:

            print(
                "Loading sentence transformer model..."
            )

            self.encoder = SentenceTransformer(
                "all-MiniLM-L6-v2",
                device="cpu"
            )

            print(
                "Sentence transformer loaded."
            )

        return self.encoder

    
    def search_catalog(
        self,
        query: str,
        k: int = 5
    ):

        encoder = self.get_encoder()

        query_embedding = encoder.encode(
            [query]
        )

        query_embedding = np.array(
            query_embedding
        ).astype("float32")

        distances, indices = self.index.search(
            query_embedding,
            k
        )

        results = []

        for idx in indices[0]:

            if idx < len(self.metadata):

                results.append(
                    self.metadata[idx]
                )

        return results

    
    def build_conversation_context(
        self,
        messages: List[Dict[str, str]]
    ):

        conversation = []

        for msg in messages:

            role = msg["role"]
            content = msg["content"]

            conversation.append(
                f"{role}: {content}"
            )

        return "\n".join(conversation)

    
    def needs_clarification(
        self,
        messages
    ):

        latest_message = (
            messages[-1]["content"]
            .lower()
            .strip()
        )

        vague_queries = [
            "test",
            "assessment",
            "need assessment",
            "need test",
            "hiring"
        ]

        if (
            len(latest_message.split()) <= 3
        ):
            return True

        if latest_message in vague_queries:
            return True

        return False

    
    def generate_response(
        self,
        messages
    ):

        try:
            if self.needs_clarification(
                messages
            ):

                return {
                    "reply": (
                        "Could you provide more "
                        "details about the role, "
                        "skills, or seniority level "
                        "you are hiring for?"
                    ),
                    "recommendations": [],
                    "end_of_conversation": False
                }

            
            conversation_context = (
                self.build_conversation_context(
                    messages
                )
            )

           
            retrieved_items = (
                self.search_catalog(
                    conversation_context,
                    k=5
                )
            )

            
            retrieval_context = ""

            for item in retrieved_items:

                retrieval_context += f"""
Name: {item['name']}
URL: {item['url']}
Type: {item['test_type']}
Description: {item['description']}

"""

           
            system_prompt = f"""
You are an SHL Assessment Recommendation Agent.

You must recommend ONLY assessments
from the retrieved catalog context.

RETRIEVED CATALOG:
{retrieval_context}

RULES:
1. Recommend ONLY from retrieved catalog.
2. Never hallucinate assessment names.
3. Never hallucinate URLs.
4. Ask clarifying questions if needed.
5. Compare assessments factually.
6. Refuse unrelated requests.

OUTPUT FORMAT:
Return ONLY valid JSON.

{{
  "reply": "response",
  "recommendations": [
    {{
      "name": "assessment name",
      "url": "assessment url",
      "test_type": "K"
    }}
  ],
  "end_of_conversation": false
}}
"""

           
            response = (
                self.groq_client
                .chat
                .completions
                .create(
                    model=(
                        "llama-3.3-70b-versatile"
                    ),
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                system_prompt
                            )
                        },
                        *messages
                    ],
                    temperature=0.1
                )
            )

            raw_output = (
                response
                .choices[0]
                .message
                .content
            )

            result = json.loads(
                raw_output
            )

           
            validated_recommendations = []

            catalog_lookup = {
                item["name"]: item
                for item in self.catalog
            }

            for rec in result.get(
                "recommendations",
                []
            ):

                name = rec.get("name")

                if name in catalog_lookup:

                    real_item = (
                        catalog_lookup[name]
                    )

                    validated_recommendations.append({
                        "name": (
                            real_item["name"]
                        ),
                        "url": (
                            real_item["url"]
                        ),
                        "test_type": (
                            real_item["test_type"]
                        )
                    })

            validated_recommendations = (
                validated_recommendations[:10]
            )

            return {
                "reply": result.get(
                    "reply",
                    "Unable to generate response."
                ),
                "recommendations": (
                    validated_recommendations
                ),
                "end_of_conversation": result.get(
                    "end_of_conversation",
                    False
                )
            }

        except Exception as e:

            print(
                "ERROR:",
                str(e)
            )

            return {
                "reply": (
                    f"Internal Error: {str(e)}"
                ),
                "recommendations": [],
                "end_of_conversation": False
            }