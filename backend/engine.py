import os
import json
import numpy as np
import faiss

from typing import List, Dict, Any

from sentence_transformers import SentenceTransformer
from groq import Groq
from dotenv import load_dotenv

load_dotenv()


class SHLConsultantEngine:

    def __init__(self):

        
        self.encoder = SentenceTransformer(
            "all-MiniLM-L6-v2"
        )

        self.groq_client = Groq(
            api_key=os.getenv("GROQ_API_KEY", "")
        )

        self.catalog = self._load_catalog()

        
        self.index = self._build_index()

   
    def _load_catalog(self) -> List[Dict[str, Any]]:

        file_path = os.path.join(
            os.path.dirname(__file__),
            "catalog.json"
        )

        with open(file_path, "r") as f:
            return json.load(f)

    
    def _build_index(self):

        texts = []

        for item in self.catalog:

            text = f"""
            Name: {item['name']}
            Description: {item['description']}
            Type: {item['test_type']}
            """

            texts.append(text)

        embeddings = self.encoder.encode(texts)

        embeddings = np.array(
            embeddings
        ).astype("float32")

        dimension = embeddings.shape[1]

        index = faiss.IndexFlatL2(dimension)

        index.add(embeddings)

        return index

    def search_catalog(
        self,
        query: str,
        k: int = 5
    ) -> List[Dict[str, Any]]:

        query_embedding = self.encoder.encode([query])

        query_embedding = np.array(
            query_embedding
        ).astype("float32")

        distances, indices = self.index.search(
            query_embedding,
            k
        )

        results = []

        for idx in indices[0]:

            if idx < len(self.catalog):

                results.append(
                    self.catalog[idx]
                )

        return results

    
    def build_conversation_context(
        self,
        messages: List[Dict[str, str]]
    ) -> str:

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
        messages: List[Dict[str, str]]
    ) -> bool:

        latest_message = messages[-1]["content"].lower()

        vague_phrases = [
            "need assessment",
            "need test",
            "hiring",
            "need hiring test",
            "assessment",
            "test"
        ]

        
        if len(latest_message.split()) <= 3:
            return True

        for phrase in vague_phrases:

            if phrase == latest_message:
                return True

        return False

    
    def generate_response(
        self,
        messages: List[Dict[str, str]]
    ) -> Dict[str, Any]:

        try:

            
            if self.needs_clarification(messages):

                return {
                    "reply": (
                        "Could you share more details "
                        "about the role, seniority level, "
                        "and skills you are hiring for?"
                    ),
                    "recommendations": [],
                    "end_of_conversation": False
                }

            
            conversation_context = (
                self.build_conversation_context(
                    messages
                )
            )

            
            retrieved_items = self.search_catalog(
                conversation_context,
                k=5
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
You are an expert SHL Assessment Recommendation Agent.

Your task is to recommend ONLY SHL assessments
from the retrieved catalog context.

RETRIEVED CATALOG:
{retrieval_context}

RULES:

1. Recommend ONLY assessments from retrieved catalog.
2. Never hallucinate assessment names or URLs.
3. Ask clarifying questions if information is insufficient.
4. Handle refinement naturally.
5. Compare assessments factually.
6. Refuse unrelated questions.

IMPORTANT:
- Recommendations must contain between 1 and 10 items.
- Use exact names and URLs from catalog.

OUTPUT FORMAT:
Return ONLY valid JSON.

{{
  "reply": "response text",
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

            
            response = self.groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {
                        "role": "system",
                        "content": system_prompt
                    },
                    *messages
                ],
                temperature=0.1,
                response_format={
                    "type": "json_object"
                }
            )

            raw_output = (
                response
                .choices[0]
                .message
                .content
            )

            result = json.loads(raw_output)

           
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

                    real_item = catalog_lookup[name]

                    validated_recommendations.append({
                        "name": real_item["name"],
                        "url": real_item["url"],
                        "test_type": real_item["test_type"]
                    })

            
            validated_recommendations = (
                validated_recommendations[:10]
            )

            return {
                "reply": result.get(
                    "reply",
                    "I could not process your request."
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

            print("ERROR:", str(e))

            return {
        "reply": (
            f"Internal Error: {str(e)}"
        ),
        "recommendations": [],
        "end_of_conversation": False
    }
            