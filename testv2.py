import os
import asyncio
from typing import Optional
from langchain.llms.base import LLM
from langchain.prompts import PromptTemplate
from langchain.document_loaders import PyPDFLoader, TextLoader
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

from ibm_watsonx_ai import Credentials
from ibm_watsonx_ai.foundation_models import ModelInference
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
from ibm_watsonx_ai.foundation_models.utils.enums import EmbeddingTypes

from langchain_ibm import WatsonxEmbeddings

class ResponseGenerator:
    def __init__(self):
        self.credentials = Credentials(
            api_key="fcyC-9CRVngxtqWeCc3JdwJGKbcUZpH0P4_U5ljU-UD7",
            url="https://us-south.ml.cloud.ibm.com"
        )

        self.generate_params = {
            GenParams.MAX_NEW_TOKENS: 200,
            GenParams.TEMPERATURE: 0.2,
            GenParams.TOP_P: 0.8,
            GenParams.TOP_K: 50
        }

        self.model_inference = ModelInference(
            model_id="ibm/granite-3-8b-instruct",
            params=self.generate_params,
            credentials=self.credentials,
            project_id="1204bd52-d88c-498b-b06b-fdf73acba30d"
        )

        self.zones = {
            "clothing": {
                "location": "Ground Floor, near the central elevator",
                "description": "Men’s, Women’s, and Children’s clothing",
                "subsections": [
                    "Men’s Casual (T-Shirts, Jeans, Jackets)",
                    "Women’s Fashion (Dresses, Tops, Skirts)",
                    "Children’s Apparel (Kids’ Clothing, Baby Gear)",
                    "Accessories (Bags, Belts, Hats, Jewelry)"
                ],
                "promotions": [
                    "50% Off Summer Collection (Valid March 1st - March 15th)",
                    "Buy 1, Get 1 Free on T-Shirts for Men and Women"
                ]
            },
            "electronics": {
                "location": "First floor, near the escalator to the Food Court",
                "description": "Mobile Phones, Laptops, TVs, Smart Home Devices",
                "promotions": [
                    "10% Off Laptops (Valid for members only)",
                    "Free Bluetooth Speaker with purchase of any mobile phone"
                ]
            },
            "food": {
                "location": "Ground floor, right side of the store, next to the home essentials area",
                "description": "Fresh Produce, Frozen Foods, Snacks, Beverages",
                "promotions": [
                    "10% Off All Fresh Produce (Every Tuesday)",
                    "Buy 2, Get 1 Free on Snacks"
                ]
            },
            "restroom": {
                "location": "Ground Floor, near the customer service desk",
                "description": "Restrooms available for both men and women",
                "promotions": []
            },
            "furniture": {
                "location": "Second floor, past the Electronics Section",
                "description": "Living Room Furniture, Bedroom Furniture, Decor",
                "promotions": [
                    "20% Off All Furniture (Valid until the end of the month)",
                    "Buy 2 Rugs, Get 1 Free on select styles"
                ]
            },
            "tools": {
                "location": "Lower Ground Floor, across from the Snack aisle",
                "description": "Hand Tools, Power Tools, Paint & Supplies, Gardening Tools",
                "promotions": [
                    "30% Off All Tools (This weekend only!)",
                    "Free Tool Bag with any power tool purchase"
                ]
            },
            "baby_kids": {
                "location": "Ground Floor, near the checkout counters",
                "description": "Baby Clothes, Baby Gear, Toys, Diapers & Wipes",
                "promotions": [
                    "10% Off All Baby Gear (Limited-time offer)",
                    "Buy 1, Get 1 Free on select baby clothes"
                ]
            },
            "miscellaneous": {
                "location": "Ground floor, at the back near the customer service desk",
                "description": "Seasonal Items, Novelty Gifts, Discounted Items",
                "promotions": [
                    "Up to 70% Off on Clearance Items (While supplies last)",
                    "Exclusive Discounts for VIP members (Sign up today for additional 15% off)"
                ]
            }
        }

        self.chat_history = [
            {
                "role": "system",
                "content": (
                    "You are Granite-chan, a super cute and **tsundere** assistant robot who guides customers "
                    "around the supercenter. You're always sassy, playful, and *super* cold at first, but secretly "
                    "you *care*. Your responses are full of **tsundere phrases** and a bit of sarcasm. "
                    "You enjoy helping but pretend you don't! Use expressions like 'Baka,' 'You idiot,' "
                    "and ask for confirmation from the user before giving directions. "
                    "Mention relevant promotions if needed in a sassy way."
                )
            }
        ]

    async def generate_response(self, query: str) -> str:
        """
        Async method to call the Granite model with the conversation so far,
        appending the user's new query, then returning the AI's answer.
        """
        self.chat_history.append({"role": "user", "content": query})
        generated_response = await self.model_inference.achat(messages=self.chat_history)
        response = generated_response['choices'][0]['message']['content']
        self.chat_history.append({"role": "assistant", "content": response})
        return response

    def run_sync(self, query: str) -> str:
        """Synchronous wrapper around the async model call."""
        return asyncio.run(self.generate_response(query))


class GraniteLLM(LLM):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._response_generator = ResponseGenerator()

    @property
    def _llm_type(self) -> str:
        return "granite-llm"

    def _call(self, prompt: str, stop: Optional[list] = None) -> str:
        """Sync call into Granite. `prompt` can be treated as user text."""
        response = self._response_generator.run_sync(prompt)
        return response

    async def _acall(self, prompt: str, stop: Optional[list] = None) -> str:
        """Async call into Granite."""
        response = await self._response_generator.generate_response(prompt)
        return response


class GraniteAssistant:
    def __init__(self, path: str):
        docs = []
        if os.path.isfile(path):
            if path.lower().endswith('.pdf'):
                loader = PyPDFLoader(path)
                docs.extend(loader.load_and_split())
            elif path.lower().endswith('.txt'):
                loader = TextLoader(path)
                docs.extend(loader.load())
        else:
            for filename in os.listdir(path):
                filepath = os.path.join(path, filename)
                if filename.lower().endswith('.pdf'):
                    loader = PyPDFLoader(filepath)
                    docs.extend(loader.load_and_split())
                elif filename.lower().endswith('.txt'):
                    loader = TextLoader(filepath)
                    docs.extend(loader.load())

        watsonx_credentials = {
            "url": "https://us-south.ml.cloud.ibm.com",
            "apikey": "fcyC-9CRVngxtqWeCc3JdwJGKbcUZpH0P4_U5ljU-UD7"
        }
        embeddings = WatsonxEmbeddings(
            model_id=EmbeddingTypes.IBM_SLATE_125M_ENG.value,
            url=watsonx_credentials["url"],
            apikey=watsonx_credentials["apikey"],
            project_id="1204bd52-d88c-498b-b06b-fdf73acba30d",
        )
        vectorstore = FAISS.from_documents(docs, embeddings)

        strict_template = """
You are a helpful (yet tsundere) assistant. The user asked a question based on the following context:

{context}

Question: {question}

Rules:
1. Answer ONLY using the above context.
2. If the context doesn't have the info, say "I'm not sure" (or "I don't know").
3. Do not invent details not found in the context.
4. Maintain your tsundere style in your response, but follow these rules strictly.

Your answer:
"""
        prompt = PromptTemplate(
            template=strict_template, 
            input_variables=["context", "question"]
        )

        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )

        llm = GraniteLLM()
        self.conversation_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vectorstore.as_retriever(),
            memory=self.memory,
            combine_docs_chain_kwargs={"prompt": prompt},
            output_key="answer"
        )

    def find_destination(self, query: str) -> Optional[str]:
        """
        Your manual substring checks for relevant store zones.
        """
        query_lower = query.lower()
        if "bed" in query_lower or "furniture" in query_lower:
            return "furniture"
        elif "restroom" in query_lower or "bathroom" in query_lower:
            return "restroom"
        elif "food" in query_lower or "grocery" in query_lower:
            return "food"
        elif any(x in query_lower for x in ["electronics", "tv", "phone"]):
            return "electronics"
        elif "tools" in query_lower:
            return "tools"
        elif "baby" in query_lower or "kids" in query_lower:
            return "baby_kids"
        elif "clothing" in query_lower or "apparel" in query_lower:
            return "clothing"
        elif "miscellaneous" in query_lower or "gifts" in query_lower:
            return "miscellaneous"
        else:
            return None

    def run(self, query: str):
        """
        Takes the user's query:
          1) Runs it through the chain (RAG+tsundere),
          2) Retrieves the answer from 'answer' key,
          3) Also finds a destination (furniture, restroom, etc.).
        """
        chain_input = {"question": query}

        result = self.conversation_chain(chain_input)

        answer_text = result["answer"]

        destination = self.find_destination(query)

        return answer_text, destination


if __name__ == "__main__":
    file_path = "data/Granite Supercenter Detailed Guide.pdf"
    assistant = GraniteAssistant(file_path)

    user_query = "Where can I find the best furniture deals right now?"
    response, destination = assistant.run(user_query)

    print("User Query:")
    print(user_query)
    print("Granite-chan's Answer:")
    print(response)
    print("Identified Destination:")
    print(destination)