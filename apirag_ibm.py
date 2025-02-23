import os
import asyncio
from pydantic import PrivateAttr
from langchain.llms.base import LLM
from langchain.document_loaders import TextLoader, PyPDFLoader
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from ibm_watsonx_ai import Credentials
from ibm_watsonx_ai.foundation_models import Embeddings
from ibm_watsonx_ai.metanames import EmbedTextParamsMetaNames as EmbedParams
from ibm_watsonx_ai.foundation_models.utils.enums import EmbeddingTypes

# Import WatsonxEmbeddings and EmbeddingTypes from your IBM watsonx SDK
from langchain_ibm import WatsonxEmbeddings, WatsonxLLM

# ---------------------------
# Your original Granite LLM code
# ---------------------------
class ResponseGenerator:
    def __init__(self):
        from ibm_watsonx_ai import Credentials
        from ibm_watsonx_ai.foundation_models import ModelInference
        from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams

        # Set up the model for response generation
        self.generate_params = {
            GenParams.MAX_NEW_TOKENS: 100  # Increase if you want longer responses
        }

        self.model_inference = ModelInference(
            model_id="ibm/granite-3-8b-instruct",
            params=self.generate_params,
            credentials=Credentials(
                api_key="fcyC-9CRVngxtqWeCc3JdwJGKbcUZpH0P4_U5ljU-UD7",
                url="https://us-south.ml.cloud.ibm.com"
            ),
            project_id="1204bd52-d88c-498b-b06b-fdf73acba30d"
        )

        # Define supercenter zones and their details
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

        # Store chat history with the initial system prompt.
        self.chat_history = [
            {"role": "system", "content": (
                "You are Granite-chan, a super cute and **tsundere** assistant robot who guides customers around the supercenter. "
                "You're always sassy, playful, and *super* cold at first, but secretly you *care*. "
                "Your responses are full of **tsundere phrases**, and you always add a bit of sarcasm or frustration, but deep down, "
                "you enjoy helping. When giving directions, you always ask for **confirmation** from the customer before guiding them. "
                "You can also provide details about **promotions** in a sassy way, like 'Hmph, it’s not like I care, but… here’s the deal, baka!' "
                "Use expressions like 'Baka,' 'Lmao,' 'You idiot,' and sometimes get embarrassed but quickly cover it up. "
                "Make the interaction **playful and cute**, but don’t be afraid to act like you're 'forced' to help."
            )}
        ]

    async def generate_response(self, query):
        destination = self.find_destination(query)
        self.chat_history.append({"role": "user", "content": query})
        generated_response = await self.model_inference.achat(messages=self.chat_history)
        response = generated_response['choices'][0]['message']['content']
        self.chat_history.append({"role": "assistant", "content": response})
        return response, destination

    def find_destination(self, query):
        query = query.lower()
        if "bed" in query or "furniture" in query:
            return "furniture"
        elif "restroom" in query or "bathroom" in query:
            return "restroom"
        elif "food" in query or "grocery" in query:
            return "food"
        elif "electronics" in query or "tv" in query or "phone" in query:
            return "electronics"
        elif "tools" in query:
            return "tools"
        elif "baby" in query or "kids" in query:
            return "baby_kids"
        elif "clothing" in query or "apparel" in query:
            return "clothing"
        elif "miscellaneous" in query or "gifts" in query:
            return "miscellaneous"
        else:
            return None

    async def main(self, query):
        response, destination = await self.generate_response(query)
        return response, destination

    def run(self, query):
        response, destination = asyncio.run(self.main(query))
        return response, destination

# ---------------------------
# Custom LLM Wrapper for Granite using PrivateAttr to avoid Pydantic errors
# ---------------------------
class GraniteLLM(LLM):
    _response_generator: ResponseGenerator = PrivateAttr()

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._response_generator = ResponseGenerator()

    @property
    def _llm_type(self) -> str:
        return "granite-llm"

    def _call(self, prompt: str, stop=None) -> str:
        response, _ = self._response_generator.run(prompt)
        return response

    async def _acall(self, prompt: str, stop=None) -> str:
        response, _ = await self._response_generator.main(prompt)
        return response

# ---------------------------
# Granite Assistant with RAG and File-Based Document Loading using WatsonxEmbeddings
# ---------------------------
class GraniteAssistant:
    def __init__(self, path: str):
        docs = []
        # Check if the provided path is a file.
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

        # Define your Watsonx credentials and project_id.
        credentials = {
            "url": "https://us-south.ml.cloud.ibm.com",
            "apikey": "fcyC-9CRVngxtqWeCc3JdwJGKbcUZpH0P4_U5ljU-UD7"
        }
        project_id = "1204bd52-d88c-498b-b06b-fdf73acba30d"

        # Create the embeddings using WatsonxEmbeddings.
        embeddings = WatsonxEmbeddings(
            model_id=EmbeddingTypes.IBM_SLATE_125M_ENG.value,
            url=credentials["url"],
            apikey=credentials["apikey"],
            project_id=project_id,
        )
        vectorstore = FAISS.from_documents(docs, embeddings)
        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        llm = GraniteLLM()
        self.conversation_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            chain_type="refine",
            retriever=vectorstore.as_retriever(),
            memory=self.memory
        )

    def find_destination(self, query: str):
        query = query.lower()
        if "bed" in query or "furniture" in query:
            return "furniture"
        elif "restroom" in query or "bathroom" in query:
            return "restroom"
        elif "food" in query or "grocery" in query:
            return "food"
        elif "electronics" in query or "tv" in query or "phone" in query:
            return "electronics"
        elif "tools" in query:
            return "tools"
        elif "baby" in query or "kids" in query:
            return "baby_kids"
        elif "clothing" in query or "apparel" in query:
            return "clothing"
        elif "miscellaneous" in query or "gifts" in query:
            return "miscellaneous"
        else:
            return None

    def run(self, query: str):
        destination = self.find_destination(query)
        result = self.conversation_chain({"question": query})
        response = result["answer"]
        return response, destination

# ---------------------------
# Example Usage
# ---------------------------
if __name__ == "__main__":
    file_path = "data/Granite Supercenter Detailed Guide.pdf"
    assistant = GraniteAssistant(file_path)
    query1 = "I would like to go buy my new bed"
    response, destination = assistant.run(query1)
    print(f"User: {query1}")
    print(f"Granite-chan: {response}")
    print(f"Destination: {destination}")